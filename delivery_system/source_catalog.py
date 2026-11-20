"""Independent source-segment catalog.

Identity is ``segment_id``; ``position`` is only the current ordinal.  The
catalog publishes immutable :class:`SourceProduct` snapshots and records every
mutation as a numbered release so historical source state stays queryable.

This side knows nothing about translations, covers or packages.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterable, Iterator, Optional, Sequence

from .errors import (
    DocumentExists,
    DocumentNotFound,
    DuplicateSegment,
    InvalidPosition,
    InvalidTokenEdit,
    RevisionConflict,
    SegmentNotFound,
    TokenMaterialMissing,
    TokenRevisionConflict,
)
from .models import (
    AnnotationResolution,
    CollationAnnotation,
    SourceProduct,
    SourceSegment,
    SourceToken,
    TokenEditPlan,
)
from .collation import (
    ORIGIN_INITIAL,
    apply_batch,
    normalize_material,
    resolve_selection,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id      TEXT PRIMARY KEY,
    revision         INTEGER NOT NULL,
    created_at       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS segments (
    document_id      TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    segment_id       TEXT NOT NULL,
    position         INTEGER NOT NULL,
    revision         INTEGER NOT NULL,
    text             TEXT NOT NULL,
    PRIMARY KEY (document_id, segment_id),
    UNIQUE (document_id, position)
);
CREATE TABLE IF NOT EXISTS source_releases (
    release_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id      TEXT NOT NULL,
    document_revision INTEGER NOT NULL,
    snapshot_json    TEXT NOT NULL,
    created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_segments_doc_position
    ON segments (document_id, position);
CREATE INDEX IF NOT EXISTS idx_releases_doc
    ON source_releases (document_id, release_id);
CREATE TABLE IF NOT EXISTS segment_tokens (
    document_id      TEXT NOT NULL,
    segment_id       TEXT NOT NULL,
    revision         INTEGER NOT NULL,
    token_id         TEXT NOT NULL,
    position         INTEGER NOT NULL,
    literal          TEXT NOT NULL,
    origin           TEXT NOT NULL,
    parents_json     TEXT NOT NULL,
    PRIMARY KEY (document_id, segment_id, revision, token_id),
    UNIQUE (document_id, segment_id, revision, position)
);
CREATE INDEX IF NOT EXISTS idx_tokens_segment_rev
    ON segment_tokens (document_id, segment_id, revision);
CREATE TABLE IF NOT EXISTS token_edits (
    edit_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id      TEXT NOT NULL,
    segment_id       TEXT NOT NULL,
    base_revision    INTEGER NOT NULL,
    new_revision     INTEGER NOT NULL,
    ops_json         TEXT NOT NULL,
    final_json       TEXT NOT NULL,
    created_at       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS collation_annotations (
    annotation_id    TEXT PRIMARY KEY,
    document_id      TEXT NOT NULL,
    segment_id       TEXT NOT NULL,
    base_revision    INTEGER NOT NULL,
    selected_json    TEXT NOT NULL,
    note             TEXT NOT NULL,
    created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_annotations_segment
    ON collation_annotations (document_id, segment_id, base_revision);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect(db_path: str, *, readonly: bool = False) -> sqlite3.Connection:
    if readonly:
        uri = f"file:{os.fspath(db_path)}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
    else:
        conn = sqlite3.connect(os.fspath(db_path), isolation_level=None)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def _transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


class SourceCatalog:
    """Read/write source catalog backed by one SQLite database."""

    def __init__(self, db_path: str) -> None:
        self.db_path = os.fspath(db_path)
        parent = os.path.dirname(os.path.abspath(self.db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        conn = _connect(self.db_path)
        try:
            conn.executescript(_SCHEMA)
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # creation / structural mutations
    # ------------------------------------------------------------------
    def create_document(
        self,
        document_id: str,
        segments: Sequence[dict[str, object]],
        *,
        document_revision: int = 1,
    ) -> SourceProduct:
        if not document_id:
            raise DocumentNotFound("document_id must be non-empty")
        if document_revision < 1:
            raise RevisionConflict("document_revision must be >= 1")
        seen: set[str] = set()
        rows: list[tuple[str, str, int, int, str]] = []
        for position, seg in enumerate(segments):
            sid = str(seg["segment_id"])
            if sid in seen:
                raise DuplicateSegment(f"duplicate segment_id {sid!r}")
            seen.add(sid)
            revision = int(seg.get("revision", 1))
            if revision < 1:
                raise RevisionConflict(f"segment {sid!r} revision must be >= 1")
            rows.append((document_id, sid, position, revision, str(seg["text"])))

        conn = _connect(self.db_path)
        try:
            with _transaction(conn):
                existing = conn.execute(
                    "SELECT 1 FROM documents WHERE document_id = ?", (document_id,)
                ).fetchone()
                if existing:
                    raise DocumentExists(f"document {document_id!r} already exists")
                conn.execute(
                    "INSERT INTO documents (document_id, revision, created_at) VALUES (?, ?, ?)",
                    (document_id, document_revision, _now()),
                )
                conn.executemany(
                    "INSERT INTO segments (document_id, segment_id, position, revision, text)"
                    " VALUES (?, ?, ?, ?, ?)",
                    rows,
                )
                snapshot = [
                    {"segment_id": r[1], "position": r[2], "revision": r[3], "text": r[4]}
                    for r in rows
                ]
                self._record_release(conn, document_id, document_revision, snapshot)
        finally:
            conn.close()
        return self.published_product(document_id)

    def add_segment(
        self,
        document_id: str,
        segment_id: str,
        text: str,
        *,
        position: Optional[int] = None,
        revision: int = 1,
    ) -> SourceProduct:
        if not segment_id:
            raise SegmentNotFound("segment_id must be non-empty")
        if revision < 1:
            raise RevisionConflict("revision must be >= 1")
        conn = _connect(self.db_path)
        try:
            with _transaction(conn):
                self._require_document(conn, document_id)
                exists = conn.execute(
                    "SELECT 1 FROM segments WHERE document_id = ? AND segment_id = ?",
                    (document_id, segment_id),
                ).fetchone()
                if exists:
                    raise DuplicateSegment(f"segment {segment_id!r} already exists")
                ids = self._ordered_ids(conn, document_id)
                if position is None:
                    position = len(ids)
                if position < 0 or position > len(ids):
                    raise InvalidPosition(
                        f"position {position} outside 0..{len(ids)}"
                    )
                conn.execute(
                    "INSERT INTO segments (document_id, segment_id, position, revision, text)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (document_id, segment_id, -(len(ids) + 1), revision, text),
                )
                ids.insert(position, segment_id)
                self._set_order(conn, document_id, ids)
                self._bump_and_release(conn, document_id)
        finally:
            conn.close()
        return self.published_product(document_id)

    def revise_segment(
        self,
        document_id: str,
        segment_id: str,
        text: str,
        *,
        revision: Optional[int] = None,
    ) -> SourceProduct:
        """Publish new text and a new per-segment revision.

        The old revision is never re-labelled: translations that referenced it
        become stale immediately.
        """

        conn = _connect(self.db_path)
        try:
            with _transaction(conn):
                self._require_document(conn, document_id)
                row = conn.execute(
                    "SELECT revision FROM segments WHERE document_id = ? AND segment_id = ?",
                    (document_id, segment_id),
                ).fetchone()
                if row is None:
                    raise SegmentNotFound(
                        f"segment {segment_id!r} not found in {document_id!r}"
                    )
                current = int(row["revision"])
                new_revision = current + 1 if revision is None else int(revision)
                if new_revision <= current:
                    raise RevisionConflict(
                        f"new revision {new_revision} must be greater than current {current}"
                    )
                conn.execute(
                    "UPDATE segments SET text = ?, revision = ?"
                    " WHERE document_id = ? AND segment_id = ?",
                    (text, new_revision, document_id, segment_id),
                )
                self._bump_and_release(conn, document_id)
        finally:
            conn.close()
        return self.published_product(document_id)

    def move_segment(
        self, document_id: str, segment_id: str, new_position: int
    ) -> SourceProduct:
        """Reorder by stable identity; identities and revisions are untouched."""

        conn = _connect(self.db_path)
        try:
            with _transaction(conn):
                self._require_document(conn, document_id)
                ids = self._ordered_ids(conn, document_id)
                if segment_id not in ids:
                    raise SegmentNotFound(
                        f"segment {segment_id!r} not found in {document_id!r}"
                    )
                if new_position < 0 or new_position >= len(ids):
                    raise InvalidPosition(
                        f"position {new_position} outside 0..{len(ids) - 1}"
                    )
                ids.remove(segment_id)
                ids.insert(new_position, segment_id)
                self._set_order(conn, document_id, ids)
                self._bump_and_release(conn, document_id)
        finally:
            conn.close()
        return self.published_product(document_id)

    def remove_segment(self, document_id: str, segment_id: str) -> SourceProduct:
        conn = _connect(self.db_path)
        try:
            with _transaction(conn):
                self._require_document(conn, document_id)
                cur = conn.execute(
                    "DELETE FROM segments WHERE document_id = ? AND segment_id = ?",
                    (document_id, segment_id),
                )
                if cur.rowcount == 0:
                    raise SegmentNotFound(
                        f"segment {segment_id!r} not found in {document_id!r}"
                    )
                ids = self._ordered_ids(conn, document_id)
                self._set_order(conn, document_id, ids)
                self._bump_and_release(conn, document_id)
        finally:
            conn.close()
        return self.published_product(document_id)

    def publish_token_plan(
        self,
        plan: TokenEditPlan,
        *,
        expected_final: Optional[Sequence[tuple[str, str]]] = None,
    ) -> TokenEditPlan:
        """Publish a previously previewed plan with its *exact* minted identities.

        Re-checks that ``plan.base_revision`` is still current, so a stale
        baseline cannot overwrite newer material, and optionally re-verifies the
        approved final order.  This is the inspect-then-publish path.
        """

        final = plan.final_signature()
        if expected_final is not None:
            expected = tuple((str(t), str(l)) for t, l in expected_final)
            if final != expected:
                raise InvalidTokenEdit(
                    "verified final order does not match the approved plan"
                )
        conn = _connect(self.db_path)
        try:
            with _transaction(conn):
                row = conn.execute(
                    "SELECT revision FROM segments WHERE document_id = ? AND segment_id = ?",
                    (plan.document_id, plan.segment_id),
                ).fetchone()
                if row is None:
                    raise SegmentNotFound(
                        f"segment {plan.segment_id!r} not found in {plan.document_id!r}"
                    )
                if int(row["revision"]) != plan.base_revision:
                    raise TokenRevisionConflict(
                        f"plan is based on revision {plan.base_revision} but current is "
                        f"{row['revision']}; a stale baseline may not overwrite newer "
                        "material"
                    )
                conn.execute(
                    "UPDATE segments SET text = ?, revision = ?"
                    " WHERE document_id = ? AND segment_id = ?",
                    (plan.text, plan.new_revision, plan.document_id, plan.segment_id),
                )
                conn.executemany(
                    "INSERT INTO segment_tokens (document_id, segment_id, revision,"
                    " token_id, position, literal, origin, parents_json)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        (
                            plan.document_id,
                            plan.segment_id,
                            plan.new_revision,
                            t.token_id,
                            t.position,
                            t.literal,
                            t.origin,
                            json.dumps(list(t.parents), ensure_ascii=False),
                        )
                        for t in plan.tokens
                    ],
                )
                conn.execute(
                    "INSERT INTO token_edits (document_id, segment_id, base_revision,"
                    " new_revision, ops_json, final_json, created_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        plan.document_id,
                        plan.segment_id,
                        plan.base_revision,
                        plan.new_revision,
                        "[]",
                        json.dumps(
                            [[t.token_id, t.literal] for t in plan.tokens],
                            ensure_ascii=False,
                        ),
                        _now(),
                    ),
                )
                self._bump_and_release(conn, plan.document_id)
        finally:
            conn.close()
        return plan

    # ------------------------------------------------------------------
    # token material (identity-bearing revisions)
    # ------------------------------------------------------------------
    def set_initial_tokens(
        self,
        document_id: str,
        segment_id: str,
        tokens: Sequence[dict[str, object]],
        *,
        text: Optional[str] = None,
        revision: Optional[int] = None,
    ) -> list[SourceToken]:
        """Attach initial token material to a segment (a source revision).

        ``tokens`` is an ordered list of ``{token_id?, literal}``.  This is the
        only entry point for a segment to gain token material; every token gets
        ``origin='initial'``.  The segment text defaults to the literals joined
        by single spaces.
        """

        conn = _connect(self.db_path)
        try:
            with _transaction(conn):
                self._require_document(conn, document_id)
                row = conn.execute(
                    "SELECT revision FROM segments WHERE document_id = ? AND segment_id = ?",
                    (document_id, segment_id),
                ).fetchone()
                if row is None:
                    raise SegmentNotFound(
                        f"segment {segment_id!r} not found in {document_id!r}"
                    )
                current = int(row["revision"])
                target_revision = current if revision is None else int(revision)
                if target_revision < 1:
                    raise RevisionConflict("revision must be >= 1")
                exists = conn.execute(
                    "SELECT 1 FROM segment_tokens WHERE document_id = ? AND segment_id = ?",
                    (document_id, segment_id),
                ).fetchone()
                if exists:
                    raise InvalidTokenEdit(
                        f"segment {segment_id!r} already has token material"
                    )
                minted: set[str] = set()
                pairs = normalize_material(tokens, minted=minted)
                if not pairs:
                    raise InvalidTokenEdit("initial token material must not be empty")
                lit_text = text if text is not None else " ".join(l for _, l in pairs)
                conn.execute(
                    "UPDATE segments SET text = ?, revision = ?"
                    " WHERE document_id = ? AND segment_id = ?",
                    (lit_text, target_revision, document_id, segment_id),
                )
                rows = [
                    (document_id, segment_id, target_revision, tid, pos, literal,
                     ORIGIN_INITIAL, "[]")
                    for pos, (tid, literal) in enumerate(pairs)
                ]
                conn.executemany(
                    "INSERT INTO segment_tokens (document_id, segment_id, revision,"
                    " token_id, position, literal, origin, parents_json)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )
                self._bump_and_release(conn, document_id)
        finally:
            conn.close()
        return self.list_tokens(document_id, segment_id, target_revision)

    def list_tokens(
        self,
        document_id: str,
        segment_id: str,
        revision: Optional[int] = None,
    ) -> list[SourceToken]:
        """List token material for a segment at ``revision`` (default: current).

        Raises :class:`TokenMaterialMissing` when the segment has no token
        material at all — the plain-text exception the brief calls out.
        """

        conn = _connect(self.db_path, readonly=True)
        try:
            return self._list_tokens(conn, document_id, segment_id, revision)
        finally:
            conn.close()

    @staticmethod
    def _list_tokens(
        conn: sqlite3.Connection,
        document_id: str,
        segment_id: str,
        revision: Optional[int],
    ) -> list[SourceToken]:
        if revision is None:
            row = conn.execute(
                "SELECT revision FROM segments WHERE document_id = ? AND segment_id = ?",
                (document_id, segment_id),
            ).fetchone()
            if row is None:
                raise SegmentNotFound(
                    f"segment {segment_id!r} not found in {document_id!r}"
                )
            revision = int(row["revision"])
        rows = conn.execute(
            "SELECT token_id, literal, position, origin, parents_json FROM segment_tokens"
            " WHERE document_id = ? AND segment_id = ? AND revision = ? ORDER BY position",
            (document_id, segment_id, int(revision)),
        ).fetchall()
        if not rows:
            raise TokenMaterialMissing(
                f"segment {segment_id!r} has no token material at revision {revision}; "
                "plain-text segments use the original segment interface"
            )
        return [
            SourceToken(
                document_id=document_id,
                segment_id=segment_id,
                revision=int(revision),
                token_id=r["token_id"],
                literal=r["literal"],
                position=int(r["position"]),
                origin=r["origin"],
                parents=tuple(json.loads(r["parents_json"])),
            )
            for r in rows
        ]

    def plan_token_edit(
        self,
        document_id: str,
        segment_id: str,
        base_revision: int,
        ops: Sequence[dict[str, object]],
        *,
        text: Optional[str] = None,
    ) -> TokenEditPlan:
        """Compute the verified final token order for a batch without publishing.

        The plan carries freshly minted identities; the same ops applied later
        with ``apply_token_edit`` will reuse them *only* when they are passed
        back, so inspect-then-publish is stable.  Re-calling ``plan`` mints new
        ids for inserts/splits/merges (they are preview values).
        """

        conn = _connect(self.db_path, readonly=True)
        try:
            return self._plan_token_edit(
                conn, document_id, segment_id, base_revision, ops, text
            )
        finally:
            conn.close()

    def _plan_token_edit(
        self,
        conn: sqlite3.Connection,
        document_id: str,
        segment_id: str,
        base_revision: int,
        ops: Sequence[dict[str, object]],
        text: Optional[str],
    ) -> TokenEditPlan:
        row = conn.execute(
            "SELECT revision FROM segments WHERE document_id = ? AND segment_id = ?",
            (document_id, segment_id),
        ).fetchone()
        if row is None:
            raise SegmentNotFound(
                f"segment {segment_id!r} not found in {document_id!r}"
            )
        current = int(row["revision"])
        if int(base_revision) != current:
            raise TokenRevisionConflict(
                f"edit batch is based on revision {base_revision} but current is "
                f"{current}; a stale baseline may not overwrite newer material"
            )
        try:
            previous_tokens = self._list_tokens(conn, document_id, segment_id, base_revision)
        except TokenMaterialMissing:
            raise
        previous = [(t.token_id, t.literal) for t in previous_tokens]
        used_ids = frozenset(
            r["token_id"]
            for r in conn.execute(
                "SELECT token_id FROM segment_tokens"
                " WHERE document_id = ? AND segment_id = ?",
                (document_id, segment_id),
            ).fetchall()
        )
        tokens, new_ids = apply_batch(previous, ops, used_ids=used_ids)
        lit_text = text if text is not None else " ".join(t.literal for t in tokens)
        stamped = tuple(
            SourceToken(
                document_id=document_id,
                segment_id=segment_id,
                revision=current + 1,
                token_id=t.token_id,
                literal=t.literal,
                position=t.position,
                origin=t.origin,
                parents=t.parents,
            )
            for t in tokens
        )
        return TokenEditPlan(
            document_id=document_id,
            segment_id=segment_id,
            base_revision=int(base_revision),
            new_revision=current + 1,
            text=lit_text,
            tokens=stamped,
            new_token_ids=new_ids,
        )

    def apply_token_edit(
        self,
        document_id: str,
        segment_id: str,
        base_revision: int,
        ops: Sequence[dict[str, object]],
        *,
        text: Optional[str] = None,
        expected_final: Optional[Sequence[tuple[str, str]]] = None,
    ) -> TokenEditPlan:
        """Publish an explicit edit batch atomically.

        All validation, the revision bump, token rows and release snapshot
        happen in one transaction.  ``base_revision`` must still be current;
        ``expected_final`` (a ``(token_id, literal)`` order) rejects a batch
        whose verified order changed.  New ids are minted inside the
        transaction, so ``expected_final`` should come from a *re-submitted*
        plan whose ids are supplied explicitly in the ops.
        """

        conn = _connect(self.db_path)
        try:
            with _transaction(conn):
                plan = self._plan_token_edit(
                    conn, document_id, segment_id, base_revision, ops, text
                )
                if expected_final is not None:
                    expected = tuple((str(t), str(l)) for t, l in expected_final)
                    if plan.final_signature() != expected:
                        raise InvalidTokenEdit(
                            "verified final order does not match the approved plan; "
                            "refusing to publish a different order"
                        )
                conn.execute(
                    "UPDATE segments SET text = ?, revision = ?"
                    " WHERE document_id = ? AND segment_id = ?",
                    (plan.text, plan.new_revision, document_id, segment_id),
                )
                conn.executemany(
                    "INSERT INTO segment_tokens (document_id, segment_id, revision,"
                    " token_id, position, literal, origin, parents_json)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        (
                            document_id,
                            segment_id,
                            plan.new_revision,
                            t.token_id,
                            t.position,
                            t.literal,
                            t.origin,
                            json.dumps(list(t.parents), ensure_ascii=False),
                        )
                        for t in plan.tokens
                    ],
                )
                conn.execute(
                    "INSERT INTO token_edits (document_id, segment_id, base_revision,"
                    " new_revision, ops_json, final_json, created_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        document_id,
                        segment_id,
                        plan.base_revision,
                        plan.new_revision,
                        json.dumps(list(ops), ensure_ascii=False, sort_keys=True),
                        json.dumps(
                            [[t.token_id, t.literal] for t in plan.tokens],
                            ensure_ascii=False,
                        ),
                        _now(),
                    ),
                )
                self._bump_and_release(conn, document_id)
        finally:
            conn.close()
        return plan

    # ------------------------------------------------------------------
    # collation annotations
    # ------------------------------------------------------------------
    def add_annotation(
        self,
        annotation_id: str,
        document_id: str,
        segment_id: str,
        base_revision: int,
        selected_token_ids: Sequence[str],
        *,
        note: str = "",
    ) -> CollationAnnotation:
        """Register a collation annotation over selected token identities.

        The selection must name only tokens that exist at ``base_revision``;
        literal spelling is irrelevant, so repeated spellings are unambiguous.
        """

        ids = [str(x) for x in selected_token_ids]
        if not ids:
            raise InvalidTokenEdit("annotation selection must not be empty")
        if len(set(ids)) != len(ids):
            raise InvalidTokenEdit("annotation selection contains duplicate token ids")
        conn = _connect(self.db_path)
        try:
            with _transaction(conn):
                self._require_document(conn, document_id)
                existing = conn.execute(
                    "SELECT 1 FROM collation_annotations WHERE annotation_id = ?",
                    (annotation_id,),
                ).fetchone()
                if existing:
                    raise InvalidTokenEdit(
                        f"annotation {annotation_id!r} already exists"
                    )
                available = {
                    t.token_id
                    for t in self._list_tokens(conn, document_id, segment_id, base_revision)
                }
                unknown = [x for x in ids if x not in available]
                if unknown:
                    raise InvalidTokenEdit(
                        "annotation selects token(s) not present at revision "
                        f"{base_revision}: {','.join(unknown)}"
                    )
                conn.execute(
                    "INSERT INTO collation_annotations (annotation_id, document_id,"
                    " segment_id, base_revision, selected_json, note, created_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        annotation_id,
                        document_id,
                        segment_id,
                        int(base_revision),
                        json.dumps(ids, ensure_ascii=False),
                        note,
                        _now(),
                    ),
                )
        finally:
            conn.close()
        return CollationAnnotation(
            annotation_id=annotation_id,
            document_id=document_id,
            segment_id=segment_id,
            base_revision=int(base_revision),
            selected_token_ids=tuple(ids),
            note=note,
        )

    def get_annotation(self, annotation_id: str) -> CollationAnnotation:
        return read_annotation(self.db_path, annotation_id)

    def resolve_annotation(
        self, annotation_id: str, target_revision: int
    ) -> AnnotationResolution:
        """Follow an annotation forward to ``target_revision``.

        Returns survivors (fully sourced), mixed (partly sourced — left for a
        human), and deleted selections.  Tokens inserted inside the selection
        after the base are *not* auto-included because their ancestry is empty.
        """

        return resolve_annotation_at(self.db_path, annotation_id, target_revision)

    # ------------------------------------------------------------------
    # reads
    # ------------------------------------------------------------------
    def published_product(self, document_id: str) -> SourceProduct:
        conn = _connect(self.db_path, readonly=True)
        try:
            return self._read_product(conn, document_id)
        finally:
            conn.close()

    # alias used by the delivery side's source-reader protocol
    def get_product(self, document_id: str) -> SourceProduct:
        return self.published_product(document_id)

    def latest_release_id(self, document_id: str) -> Optional[int]:
        conn = _connect(self.db_path, readonly=True)
        try:
            row = conn.execute(
                "SELECT MAX(release_id) AS r FROM source_releases WHERE document_id = ?",
                (document_id,),
            ).fetchone()
            return None if row is None or row["r"] is None else int(row["r"])
        finally:
            conn.close()

    def product_at_release(self, release_id: int) -> SourceProduct:
        conn = _connect(self.db_path, readonly=True)
        try:
            row = conn.execute(
                "SELECT document_id, document_revision, snapshot_json"
                " FROM source_releases WHERE release_id = ?",
                (release_id,),
            ).fetchone()
            if row is None:
                raise DocumentNotFound(f"release {release_id} not found")
            data = json.loads(row["snapshot_json"])
            segments = tuple(
                SourceSegment(
                    segment_id=s["segment_id"],
                    position=int(s["position"]),
                    revision=int(s["revision"]),
                    text=s["text"],
                )
                for s in data
            )
            return SourceProduct(
                document_id=row["document_id"],
                document_revision=int(row["document_revision"]),
                segments=segments,
            )
        finally:
            conn.close()

    def get_segment(self, document_id: str, segment_id: str) -> SourceSegment:
        product = self.published_product(document_id)
        for seg in product.segments:
            if seg.segment_id == segment_id:
                return seg
        raise SegmentNotFound(f"segment {segment_id!r} not found in {document_id!r}")

    def list_documents(self) -> list[dict[str, object]]:
        conn = _connect(self.db_path, readonly=True)
        try:
            rows = conn.execute(
                "SELECT document_id, revision FROM documents ORDER BY document_id"
            ).fetchall()
            counts = dict(
                conn.execute(
                    "SELECT document_id, COUNT(*) FROM segments GROUP BY document_id"
                ).fetchall()
            )
            return [
                {
                    "document_id": r["document_id"],
                    "revision": int(r["revision"]),
                    "segments": int(counts.get(r["document_id"], 0)),
                }
                for r in rows
            ]
        finally:
            conn.close()

    def list_releases(self, document_id: str) -> list[dict[str, object]]:
        conn = _connect(self.db_path, readonly=True)
        try:
            rows = conn.execute(
                "SELECT release_id, document_revision, created_at"
                " FROM source_releases WHERE document_id = ? ORDER BY release_id",
                (document_id,),
            ).fetchall()
            return [
                {
                    "release_id": int(r["release_id"]),
                    "document_revision": int(r["document_revision"]),
                    "created_at": r["created_at"],
                }
                for r in rows
            ]
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    @staticmethod
    def _read_product(conn: sqlite3.Connection, document_id: str) -> SourceProduct:
        doc = conn.execute(
            "SELECT revision FROM documents WHERE document_id = ?", (document_id,)
        ).fetchone()
        if doc is None:
            raise DocumentNotFound(f"document {document_id!r} not found")
        rows = conn.execute(
            "SELECT segment_id, position, revision, text FROM segments"
            " WHERE document_id = ? ORDER BY position",
            (document_id,),
        ).fetchall()
        segments = tuple(
            SourceSegment(
                segment_id=r["segment_id"],
                position=int(r["position"]),
                revision=int(r["revision"]),
                text=r["text"],
            )
            for r in rows
        )
        return SourceProduct(
            document_id=document_id,
            document_revision=int(doc["revision"]),
            segments=segments,
        )

    @staticmethod
    def _require_document(conn: sqlite3.Connection, document_id: str) -> None:
        row = conn.execute(
            "SELECT 1 FROM documents WHERE document_id = ?", (document_id,)
        ).fetchone()
        if row is None:
            raise DocumentNotFound(f"document {document_id!r} not found")

    @staticmethod
    def _ordered_ids(conn: sqlite3.Connection, document_id: str) -> list[str]:
        return [
            r["segment_id"]
            for r in conn.execute(
                "SELECT segment_id FROM segments WHERE document_id = ? ORDER BY position",
                (document_id,),
            ).fetchall()
        ]

    @staticmethod
    def _set_order(
        conn: sqlite3.Connection, document_id: str, ordered_ids: Iterable[str]
    ) -> None:
        ids = list(ordered_ids)
        # Move through unique negative positions, then assign final positions.
        for index, sid in enumerate(ids):
            conn.execute(
                "UPDATE segments SET position = ? WHERE document_id = ? AND segment_id = ?",
                (-(index + 1), document_id, sid),
            )
        for index, sid in enumerate(ids):
            conn.execute(
                "UPDATE segments SET position = ? WHERE document_id = ? AND segment_id = ?",
                (index, document_id, sid),
            )

    def _bump_and_release(self, conn: sqlite3.Connection, document_id: str) -> None:
        conn.execute(
            "UPDATE documents SET revision = revision + 1 WHERE document_id = ?",
            (document_id,),
        )
        revision = int(
            conn.execute(
                "SELECT revision FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()["revision"]
        )
        rows = conn.execute(
            "SELECT segment_id, position, revision, text FROM segments"
            " WHERE document_id = ? ORDER BY position",
            (document_id,),
        ).fetchall()
        snapshot = [
            {
                "segment_id": r["segment_id"],
                "position": int(r["position"]),
                "revision": int(r["revision"]),
                "text": r["text"],
            }
            for r in rows
        ]
        self._record_release(conn, document_id, revision, snapshot)

    @staticmethod
    def _record_release(
        conn: sqlite3.Connection,
        document_id: str,
        document_revision: int,
        snapshot: Sequence[dict[str, object]],
    ) -> None:
        conn.execute(
            "INSERT INTO source_releases (document_id, document_revision, snapshot_json, created_at)"
            " VALUES (?, ?, ?, ?)",
            (
                document_id,
                document_revision,
                json.dumps(snapshot, ensure_ascii=False, sort_keys=True),
                _now(),
            ),
        )


class PublishedSourceReader:
    """Read-only handle used by the delivery side.

    It opens its own SQLite connection in ``mode=ro`` and re-reads the
    catalog's published product on every call; it never caches source state.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = os.fspath(db_path)
        if not os.path.exists(self.db_path):
            raise DocumentNotFound(f"source database {self.db_path!r} does not exist")

    def published_product(self, document_id: str) -> SourceProduct:
        conn = _connect(self.db_path, readonly=True)
        try:
            return SourceCatalog._read_product(conn, document_id)
        finally:
            conn.close()

    def latest_release_id(self, document_id: str) -> Optional[int]:
        conn = _connect(self.db_path, readonly=True)
        try:
            row = conn.execute(
                "SELECT MAX(release_id) AS r FROM source_releases WHERE document_id = ?",
                (document_id,),
            ).fetchone()
            return None if row is None or row["r"] is None else int(row["r"])
        finally:
            conn.close()

    # token material / collation reads (no writes to the source database) ---
    def list_tokens(
        self, document_id: str, segment_id: str, revision: Optional[int] = None
    ) -> list[SourceToken]:
        conn = _connect(self.db_path, readonly=True)
        try:
            return SourceCatalog._list_tokens(conn, document_id, segment_id, revision)
        finally:
            conn.close()

    def get_annotation(self, annotation_id: str) -> CollationAnnotation:
        return read_annotation(self.db_path, annotation_id)

    def resolve_annotation(
        self, annotation_id: str, target_revision: int
    ) -> AnnotationResolution:
        return resolve_annotation_at(self.db_path, annotation_id, target_revision)


# ----------------------------------------------------------------------
# module-level read helpers (also usable through PublishedSourceReader)
# ----------------------------------------------------------------------
def read_annotation(db_path: str, annotation_id: str) -> CollationAnnotation:
    conn = _connect(db_path, readonly=True)
    try:
        row = conn.execute(
            "SELECT * FROM collation_annotations WHERE annotation_id = ?",
            (annotation_id,),
        ).fetchone()
        if row is None:
            raise InvalidTokenEdit(f"annotation {annotation_id!r} not found")
        return CollationAnnotation(
            annotation_id=row["annotation_id"],
            document_id=row["document_id"],
            segment_id=row["segment_id"],
            base_revision=int(row["base_revision"]),
            selected_token_ids=tuple(json.loads(row["selected_json"])),
            note=row["note"],
        )
    finally:
        conn.close()


def resolve_annotation_at(
    db_path: str, annotation_id: str, target_revision: int
) -> AnnotationResolution:
    ann = read_annotation(db_path, annotation_id)
    conn = _connect(db_path, readonly=True)
    try:
        revisions = [
            int(r["revision"])
            for r in conn.execute(
                "SELECT DISTINCT revision FROM segment_tokens"
                " WHERE document_id = ? AND segment_id = ? ORDER BY revision",
                (ann.document_id, ann.segment_id),
            ).fetchall()
        ]
        if not revisions or ann.base_revision not in revisions:
            return AnnotationResolution(
                annotation_id=annotation_id,
                document_id=ann.document_id,
                segment_id=ann.segment_id,
                base_revision=ann.base_revision,
                target_revision=int(target_revision),
                selected_token_ids=ann.selected_token_ids,
                survivors=(),
                mixed=(),
                deleted_token_ids=ann.selected_token_ids,
                mixed_original_ids=(),
                missing_material=True,
                detail=(
                    "segment has no token material"
                    if not revisions
                    else f"base revision {ann.base_revision} has no token material"
                ),
            )
        rows: dict[int, dict[str, tuple[str, int, str, tuple[str, ...]]]] = {}
        for rev in revisions:
            rev_rows = conn.execute(
                "SELECT token_id, literal, position, origin, parents_json"
                " FROM segment_tokens WHERE document_id = ? AND segment_id = ?"
                " AND revision = ?",
                (ann.document_id, ann.segment_id, rev),
            ).fetchall()
            rows[rev] = {
                r["token_id"]: (
                    r["literal"],
                    int(r["position"]),
                    r["origin"],
                    tuple(json.loads(r["parents_json"])),
                )
                for r in rev_rows
            }
    finally:
        conn.close()

    if int(target_revision) not in rows:
        raise InvalidTokenEdit(
            f"segment has no token material at revision {target_revision}"
        )
    survivors, mixed, deleted, mixed_originals = resolve_selection(
        rows, ann.base_revision, int(target_revision), ann.selected_token_ids
    )
    return AnnotationResolution(
        annotation_id=annotation_id,
        document_id=ann.document_id,
        segment_id=ann.segment_id,
        base_revision=ann.base_revision,
        target_revision=int(target_revision),
        selected_token_ids=ann.selected_token_ids,
        survivors=tuple(survivors),
        mixed=tuple(mixed),
        deleted_token_ids=tuple(deleted),
        mixed_original_ids=tuple(mixed_originals),
        missing_material=False,
        detail="" if not mixed else "partly-sourced tokens require human review",
    )
