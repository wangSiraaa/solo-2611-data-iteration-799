"""Multilingual delivery repository: translations, covers and immutable packages.

The repository stores translation entries and packages in its *own* SQLite
database.  Source identity, order and revisions are read through a
:class:`~delivery_system.source_catalog.PublishedSourceReader`, which opens the
catalog database read-only.  The delivery side never writes source state and
never treats a position number as identity.

Publishing freezes the current source snapshot and the selected translation
entries.  Packages are immutable: reusing an id with different frozen content
is rejected, and old packages keep the source text and ids they were built
with even after the source is revised.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterable, Iterator, Optional, Protocol

from .coverage import assess_entries, solve_cover
from .errors import (
    AlignmentConflict,
    DocumentNotFound,
    InvalidAlignment,
    InvalidTranslation,
    NoCoverError,
    PackageBlockNotFound,
    PackageExistsImmutable,
    SourcePackageMismatch,
    TranslationExists,
)
from .models import (
    BlockAlignment,
    CoverResult,
    ProjectionResult,
    SourceProduct,
    SourceRef,
    TokenLineage,
    TranslationAssessment,
    TranslationEntry,
)
from .source_catalog import PublishedSourceReader

PACKAGE_FORMAT = "translation-package/1"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS translations (
    translation_id  TEXT PRIMARY KEY,
    document_id     TEXT NOT NULL,
    language        TEXT NOT NULL,
    text            TEXT NOT NULL,
    source_refs_json TEXT NOT NULL,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_translations_doc_lang
    ON translations (document_id, language);
CREATE TABLE IF NOT EXISTS packages (
    package_id       TEXT PRIMARY KEY,
    document_id      TEXT NOT NULL,
    language         TEXT NOT NULL,
    content_json     TEXT NOT NULL,
    content_sha256   TEXT NOT NULL,
    created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_packages_doc_lang
    ON packages (document_id, language);
CREATE TABLE IF NOT EXISTS alignments (
    package_id       TEXT NOT NULL,
    block_id         TEXT NOT NULL,
    version          INTEGER NOT NULL,
    document_id      TEXT NOT NULL,
    language         TEXT NOT NULL,
    payload_json     TEXT NOT NULL,
    content_sha256   TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    PRIMARY KEY (package_id, block_id, version)
);
CREATE INDEX IF NOT EXISTS idx_alignments_pkg
    ON alignments (package_id, block_id);
"""


class SourceProductReader(Protocol):
    def published_product(self, document_id: str) -> SourceProduct: ...

    def latest_release_id(self, document_id: str) -> Optional[int]: ...


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _refs_to_json(refs: Iterable[SourceRef]) -> str:
    return json.dumps(
        [{"segment_id": r.segment_id, "revision": r.revision} for r in refs],
        ensure_ascii=False,
        sort_keys=True,
    )


def _refs_from_json(raw: str) -> tuple[SourceRef, ...]:
    return tuple(
        SourceRef(segment_id=str(r["segment_id"]), revision=int(r["revision"]))
        for r in json.loads(raw)
    )


def _entry_from_row(row: sqlite3.Row) -> TranslationEntry:
    return TranslationEntry(
        translation_id=row["translation_id"],
        document_id=row["document_id"],
        language=row["language"],
        text=row["text"],
        source_refs=_refs_from_json(row["source_refs_json"]),
    )


class DeliveryRepository:
    def __init__(self, db_path: str, source: SourceProductReader) -> None:
        self.db_path = os.fspath(db_path)
        self.source = source
        parent = os.path.dirname(os.path.abspath(self.db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        conn = self._connect()
        try:
            conn.executescript(_SCHEMA)
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    @contextmanager
    def _transaction(self, conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        else:
            conn.execute("COMMIT")

    # ------------------------------------------------------------------
    # translations
    # ------------------------------------------------------------------
    def add_translation(
        self,
        translation_id: str,
        document_id: str,
        language: str,
        text: str,
        source_refs: Iterable[dict[str, object] | SourceRef],
    ) -> TranslationEntry:
        if not translation_id:
            raise InvalidTranslation("translation_id must be non-empty")
        if not language:
            raise InvalidTranslation("language must be non-empty")
        refs: list[SourceRef] = []
        for ref in source_refs:
            if isinstance(ref, SourceRef):
                refs.append(ref)
            else:
                refs.append(
                    SourceRef(
                        segment_id=str(ref["segment_id"]),
                        revision=int(ref["revision"]),
                    )
                )
        if not refs:
            raise InvalidTranslation("source_refs must not be empty")

        entry = TranslationEntry(
            translation_id=translation_id,
            document_id=document_id,
            language=language,
            text=text,
            source_refs=tuple(refs),
        )
        conn = self._connect()
        try:
            with self._transaction(conn):
                existing = conn.execute(
                    "SELECT language, text, source_refs_json FROM translations"
                    " WHERE translation_id = ?",
                    (translation_id,),
                ).fetchone()
                if existing is not None:
                    same = (
                        existing["language"] == language
                        and existing["text"] == text
                        and existing["source_refs_json"] == _refs_to_json(refs)
                    )
                    if not same:
                        raise TranslationExists(
                            f"translation {translation_id!r} already exists with different content"
                        )
                    # Idempotent re-add of an identical entry.
                    return entry
                # Make sure the document exists on the source side so we never
                # accumulate translations for unknown documents.
                try:
                    self.source.published_product(document_id)
                except DocumentNotFound:
                    raise InvalidTranslation(
                        f"document {document_id!r} does not exist in the source catalog"
                    )
                conn.execute(
                    "INSERT INTO translations (translation_id, document_id, language, text,"
                    " source_refs_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        translation_id,
                        document_id,
                        language,
                        text,
                        _refs_to_json(refs),
                        _now(),
                    ),
                )
        finally:
            conn.close()
        return entry

    def get_translation(self, translation_id: str) -> TranslationEntry:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM translations WHERE translation_id = ?",
                (translation_id,),
            ).fetchone()
            if row is None:
                raise InvalidTranslation(f"translation {translation_id!r} not found")
            return _entry_from_row(row)
        finally:
            conn.close()

    def list_translations(
        self, document_id: Optional[str] = None, language: Optional[str] = None
    ) -> list[TranslationEntry]:
        clauses: list[str] = []
        params: list[object] = []
        if document_id is not None:
            clauses.append("document_id = ?")
            params.append(document_id)
        if language is not None:
            clauses.append("language = ?")
            params.append(language)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM translations" + where + " ORDER BY translation_id",
                params,
            ).fetchall()
            return [_entry_from_row(row) for row in rows]
        finally:
            conn.close()

    def languages(self, document_id: str) -> list[str]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT DISTINCT language FROM translations WHERE document_id = ?"
                " ORDER BY language",
                (document_id,),
            ).fetchall()
            return [row["language"] for row in rows]
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # assessment / cover
    # ------------------------------------------------------------------
    def assess(
        self, document_id: str, language: str
    ) -> list[TranslationAssessment]:
        product = self.source.published_product(document_id)
        entries = self.list_translations(document_id, language)
        return assess_entries(entries, product, language)

    def plan_cover(self, document_id: str, language: str) -> CoverResult:
        product = self.source.published_product(document_id)
        entries = self.list_translations(document_id, language)
        return solve_cover(product, language, entries)

    # ------------------------------------------------------------------
    # packages
    # ------------------------------------------------------------------
    def publish_language(
        self, document_id: str, language: str, package_id: str
    ) -> dict[str, object]:
        result = self.plan_cover(document_id, language)
        if not result.ok:
            raise NoCoverError(
                f"cannot publish {language!r} for {document_id!r}: "
                f"{result.diagnosis.detail}",
                result=result,
            )
        try:
            source_release = self.source.latest_release_id(document_id)
        except Exception:  # pragma: no cover - optional reader capability
            source_release = None
        return self._freeze_package(document_id, language, package_id, result, source_release)

    def _build_content(
        self,
        document_id: str,
        language: str,
        package_id: str,
        result: CoverResult,
        source_release: Optional[int],
    ) -> dict[str, object]:
        product = result.product
        source_snapshot = product.snapshot()
        chunks = [chunk.to_package_chunk() for chunk in result.chunks]
        identity = json.dumps(
            {
                "document_id": document_id,
                "language": language,
                "source_snapshot": source_snapshot,
                "chunks": chunks,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        content_sha = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        content: dict[str, object] = {
            "format": PACKAGE_FORMAT,
            "package_id": package_id,
            "document_id": document_id,
            "language": language,
            "source_revision": product.document_revision,
            "source_release": source_release,
            "source_snapshot": source_snapshot,
            "chunks": chunks,
            "rendered_text": result.rendered_text,
            "content_sha256": content_sha,
            "created_at": _now(),
        }
        return content

    def _freeze_package(
        self,
        document_id: str,
        language: str,
        package_id: str,
        result: CoverResult,
        source_release: Optional[int],
    ) -> dict[str, object]:
        content = self._build_content(
            document_id, language, package_id, result, source_release
        )
        sha = str(content["content_sha256"])
        conn = self._connect()
        try:
            with self._transaction(conn):
                existing = conn.execute(
                    "SELECT content_sha256, content_json FROM packages WHERE package_id = ?",
                    (package_id,),
                ).fetchone()
                if existing is not None:
                    if existing["content_sha256"] != sha:
                        raise PackageExistsImmutable(
                            f"package {package_id!r} already exists with different frozen content"
                        )
                    return json.loads(existing["content_json"])
                conn.execute(
                    "INSERT INTO packages (package_id, document_id, language, content_json,"
                    " content_sha256, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        package_id,
                        document_id,
                        language,
                        json.dumps(content, ensure_ascii=False, sort_keys=True),
                        sha,
                        str(content["created_at"]),
                    ),
                )
        finally:
            conn.close()
        return content

    def get_package(self, package_id: str) -> dict[str, object]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT content_json FROM packages WHERE package_id = ?",
                (package_id,),
            ).fetchone()
            if row is None:
                raise InvalidTranslation(f"package {package_id!r} not found")
            return json.loads(row["content_json"])
        finally:
            conn.close()

    def list_packages(
        self, document_id: Optional[str] = None, language: Optional[str] = None
    ) -> list[dict[str, object]]:
        clauses: list[str] = []
        params: list[object] = []
        if document_id is not None:
            clauses.append("document_id = ?")
            params.append(document_id)
        if language is not None:
            clauses.append("language = ?")
            params.append(language)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT package_id, document_id, language, content_sha256, created_at"
                " FROM packages" + where + " ORDER BY created_at, package_id",
                params,
            ).fetchall()
            return [
                {
                    "package_id": row["package_id"],
                    "document_id": row["document_id"],
                    "language": row["language"],
                    "content_sha256": row["content_sha256"],
                    "created_at": row["created_at"],
                }
                for row in rows
            ]
        finally:
            conn.close()


    # ------------------------------------------------------------------
    # block alignments (versioned manual source -> target correspondence)
    # ------------------------------------------------------------------
    @staticmethod
    def _package_block(package: dict, block_id: str) -> dict:
        for chunk in package.get("chunks", []):
            if chunk.get("translation_id") == block_id:
                return chunk
        raise PackageBlockNotFound(
            f"block {block_id!r} does not exist in package {package.get('package_id')!r}"
        )

    @staticmethod
    def _normalize_target_text(tokens: Iterable[str]) -> str:
        return "".join(str(t) for t in tokens).replace(" ", "").replace("\n", "")

    def add_alignment(
        self,
        package_id: str,
        block_id: str,
        version: int,
        target_tokens: Iterable[str],
        mapping: dict[str, Iterable[int] | None],
        *,
        target_text: Optional[str] = None,
    ) -> BlockAlignment:
        """Register a versioned manual alignment for one immutable package block.

        ``mapping`` maps a source ``token_id`` to target positions (indices into
        ``target_tokens``).  One-to-many is a multi-position tuple, many-to-one
        is the same position for several source tokens, and an empty tuple (or
        ``None``) is an explicit "no correspondence".  Absence from the mapping
        means "not yet annotated" and is reported as such during projection.

        The block's identity, source revisions and target token string are all
        checked against the *actual frozen package*; a separate version is the
        documented way to correct a wrong alignment (versions are immutable).
        """

        package = self.get_package(package_id)
        chunk = self._package_block(package, block_id)
        block_source_ids = tuple(str(x) for x in chunk.get("source_ids", ()))
        block_revisions = tuple(int(x) for x in chunk.get("source_revisions", ()))
        if len(block_source_ids) != len(block_revisions):
            raise InvalidAlignment(
                f"package block {block_id!r} has inconsistent source_ids/revisions"
            )
        source_revision_map = dict(zip(block_source_ids, block_revisions))

        tokens = [str(t) for t in target_tokens]
        if not tokens:
            raise InvalidAlignment("target_tokens must not be empty")
        chunk_text = str(chunk.get("text", ""))
        if target_text is not None and target_text != chunk_text:
            raise InvalidAlignment(
                f"target_text does not match frozen block text for {block_id!r}"
            )
        if self._normalize_target_text(tokens) != self._normalize_target_text([chunk_text]):
            raise InvalidAlignment(
                f"target token string does not reproduce frozen block {block_id!r} text"
            )

        norm_mapping: list[tuple[str, tuple[int, ...]]] = []
        seen: set[str] = set()
        # Validate each mapped source token really exists at the block revision.
        source_tokens: dict[str, set[str]] = {}
        for seg_id, rev in source_revision_map.items():
            fetched = self.source.list_tokens(package["document_id"], seg_id, rev)  # type: ignore[attr-defined]
            source_tokens[seg_id] = {t.token_id for t in fetched}
        for token_id, positions in mapping.items():
            token_id = str(token_id)
            if token_id in seen:
                raise InvalidAlignment(f"duplicate mapping key {token_id!r}")
            seen.add(token_id)
            raw_positions = () if positions is None else tuple(positions)
            norm_positions = tuple(int(p) for p in raw_positions)
            for p in norm_positions:
                if p < 0 or p >= len(tokens):
                    raise InvalidAlignment(
                        f"target position {p} out of range for {len(tokens)} tokens"
                    )
            if len(set(norm_positions)) != len(norm_positions):
                raise InvalidAlignment(
                    f"mapping for {token_id!r} repeats a target position"
                )
            if not any(token_id in ids for ids in source_tokens.values()):
                raise InvalidAlignment(
                    f"source token {token_id!r} does not exist at the block's "
                    f"source revisions {source_revision_map!r}"
                )
            norm_mapping.append((token_id, tuple(sorted(norm_positions))))

        if int(version) < 1:
            raise InvalidAlignment("alignment version must be >= 1")
        payload = {
            "package_id": package_id,
            "block_id": block_id,
            "version": int(version),
            "document_id": package["document_id"],
            "language": package["language"],
            "source_revisions": {sid: rev for sid, rev in source_revision_map.items()},
            "target_tokens": tokens,
            "mapping": {
                tid: list(ps) for tid, ps in sorted(norm_mapping)
            },
            "target_text": chunk_text,
        }
        sha = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        conn = self._connect()
        try:
            with self._transaction(conn):
                existing = conn.execute(
                    "SELECT content_sha256 FROM alignments"
                    " WHERE package_id = ? AND block_id = ? AND version = ?",
                    (package_id, block_id, int(version)),
                ).fetchone()
                if existing is not None:
                    if existing["content_sha256"] != sha:
                        raise AlignmentConflict(
                            f"alignment {package_id}/{block_id}@v{version} already exists "
                            "with different content; add a new version to correct it"
                        )
                else:
                    conn.execute(
                        "INSERT INTO alignments (package_id, block_id, version,"
                        " document_id, language, payload_json, content_sha256, created_at)"
                        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            package_id,
                            block_id,
                            int(version),
                            package["document_id"],
                            package["language"],
                            json.dumps(payload, ensure_ascii=False, sort_keys=True),
                            sha,
                            _now(),
                        ),
                    )
        finally:
            conn.close()
        return BlockAlignment(
            package_id=package_id,
            block_id=block_id,
            version=int(version),
            document_id=package["document_id"],
            language=package["language"],
            source_revisions=tuple(sorted(source_revision_map.items())),
            target_tokens=tuple(tokens),
            mapping=tuple(norm_mapping),
            target_text=chunk_text,
            created_at=_now(),
        )

    def get_alignment(
        self, package_id: str, block_id: str, version: Optional[int] = None
    ) -> BlockAlignment:
        conn = self._connect()
        try:
            if version is None:
                row = conn.execute(
                    "SELECT * FROM alignments WHERE package_id = ? AND block_id = ?"
                    " ORDER BY version DESC LIMIT 1",
                    (package_id, block_id),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM alignments WHERE package_id = ? AND block_id = ?"
                    " AND version = ?",
                    (package_id, block_id, int(version)),
                ).fetchone()
            if row is None:
                raise InvalidAlignment(
                    f"alignment {package_id}/{block_id}@{version} not found"
                )
            payload = json.loads(row["payload_json"])
            mapping = tuple(
                (tid, tuple(int(p) for p in ps))
                for tid, ps in sorted(payload["mapping"].items())
            )
            return BlockAlignment(
                package_id=row["package_id"],
                block_id=row["block_id"],
                version=int(row["version"]),
                document_id=row["document_id"],
                language=row["language"],
                source_revisions=tuple(
                    sorted((str(k), int(v)) for k, v in payload["source_revisions"].items())
                ),
                target_tokens=tuple(payload["target_tokens"]),
                mapping=mapping,
                target_text=payload["target_text"],
                created_at=row["created_at"],
            )
        finally:
            conn.close()

    def list_alignments(
        self, package_id: str, block_id: Optional[str] = None
    ) -> list[BlockAlignment]:
        conn = self._connect()
        try:
            if block_id is None:
                rows = conn.execute(
                    "SELECT block_id, MAX(version) AS v FROM alignments"
                    " WHERE package_id = ? GROUP BY block_id ORDER BY block_id",
                    (package_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT block_id, MAX(version) AS v FROM alignments"
                    " WHERE package_id = ? AND block_id = ? GROUP BY block_id",
                    (package_id, block_id),
                ).fetchall()
        finally:
            conn.close()
        return [self.get_alignment(package_id, r["block_id"], int(r["v"])) for r in rows]

    # ------------------------------------------------------------------
    # projection: annotation survivors -> exact target intervals
    # ------------------------------------------------------------------
    def project_annotation(
        self,
        annotation_id: str,
        source_revision: int,
        package_id: str,
        alignment_version: Optional[int] = None,
        *,
        block_id: Optional[str] = None,
    ) -> ProjectionResult:
        """Project an annotation's survivors onto a package block's target order.

        Steps: resolve the annotation at ``source_revision`` (survivors, deleted,
        mixed), then map every survivor through the requested alignment version.
        Coverage is the set of target positions, so many-to-one is highlighted
        once and gaps between disjoint survivors stay unhighlighted.  Returns
        maximal contiguous half-open intervals.
        """

        resolution = self.source.resolve_annotation(annotation_id, source_revision)  # type: ignore[attr-defined]
        if resolution.missing_material:
            return ProjectionResult(
                ok=False,
                reason="missing_token_material",
                detail=resolution.detail,
                annotation_id=annotation_id,
                source_revision=int(source_revision),
                package_id=package_id,
                block_id=block_id,
                alignment_version=alignment_version,
                deleted_token_ids=resolution.deleted_token_ids,
            )

        package = self.get_package(package_id)
        if package.get("document_id") != resolution.document_id:
            raise SourcePackageMismatch(
                f"package {package_id!r} is for document {package.get('document_id')!r}, "
                f"annotation is on {resolution.document_id!r}"
            )

        # Pick the block whose recorded revision for this segment matches the
        # requested source revision.  Refuse to mix mismatched versions.
        candidates: list[dict] = []
        for chunk in package.get("chunks", []):
            ids = [str(x) for x in chunk.get("source_ids", ())]
            revs = [int(x) for x in chunk.get("source_revisions", ())]
            mapping = dict(zip(ids, revs))
            if resolution.segment_id in mapping:
                if mapping[resolution.segment_id] == int(source_revision):
                    candidates.append(chunk)
                elif block_id is not None and chunk.get("translation_id") == block_id:
                    raise SourcePackageMismatch(
                        f"package block {block_id!r} records {resolution.segment_id}"
                        f"@{mapping[resolution.segment_id]}, but the requested source "
                        f"revision is {source_revision}"
                    )
        if block_id is not None:
            candidates = [
                c for c in candidates if c.get("translation_id") == block_id
            ]
        if not candidates:
            raise SourcePackageMismatch(
                f"no block in package {package_id!r} covers {resolution.segment_id}"
                f"@{source_revision}"
            )
        if len(candidates) > 1:
            raise InvalidAlignment(
                "multiple blocks cover the annotation segment at this revision; "
                "pass block_id to disambiguate: "
                + ",".join(str(c.get("translation_id")) for c in candidates)
            )
        chosen = candidates[0]
        chosen_block_id = str(chosen["translation_id"])
        alignment = self.get_alignment(package_id, chosen_block_id, alignment_version)
        ids = [str(x) for x in chosen.get("source_ids", ())]
        revs = [int(x) for x in chosen.get("source_revisions", ())]
        block_revisions = dict(zip(ids, revs))
        if (block_revisions.get(resolution.segment_id) != int(source_revision)):
            raise SourcePackageMismatch(
                "alignment/source revision mismatch for the selected block"
            )

        mapping = alignment.mapping_dict()
        covered: set[int] = set()
        no_correspondence: list[TokenLineage] = []
        unannotated: list[TokenLineage] = []
        for survivor in resolution.survivors:
            if survivor.token_id not in mapping:
                unannotated.append(survivor)
                continue
            positions = mapping[survivor.token_id]
            if not positions:
                no_correspondence.append(survivor)
                continue
            covered.update(positions)

        ordered = sorted(covered)
        intervals: list[tuple[int, int]] = []
        for pos in ordered:
            if intervals and intervals[-1][1] == pos:
                intervals[-1] = (intervals[-1][0], pos + 1)
            else:
                intervals.append((pos, pos + 1))

        return ProjectionResult(
            ok=True,
            reason=None,
            detail="" if not resolution.mixed else "mixed-source tokens need a human",
            annotation_id=annotation_id,
            source_revision=int(source_revision),
            package_id=package_id,
            block_id=chosen_block_id,
            alignment_version=alignment.version,
            intervals=tuple(intervals),
            target_positions=tuple(ordered),
            survivors=resolution.survivors,
            mixed=resolution.mixed,
            deleted_token_ids=resolution.deleted_token_ids,
            no_correspondence=tuple(no_correspondence),
            unannotated=tuple(unannotated),
        )


def open_repository(
    delivery_db: str, source_db: str
) -> tuple[DeliveryRepository, PublishedSourceReader]:
    reader = PublishedSourceReader(source_db)
    repo = DeliveryRepository(delivery_db, reader)
    return repo, reader
