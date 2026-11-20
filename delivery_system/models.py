"""Immutable value objects shared by the catalog and the delivery side."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SourceSegment:
    """One segment of a source document.

    ``segment_id`` is the identity; ``position`` is only the current ordinal
    in the document and may change on reorder.  ``revision`` is per-segment.
    """

    segment_id: str
    position: int
    revision: int
    text: str


@dataclass(frozen=True)
class SourceProduct:
    """Immutable, ordered view of a source document as published by the catalog."""

    document_id: str
    document_revision: int
    segments: tuple[SourceSegment, ...]

    def __post_init__(self) -> None:
        positions = [seg.position for seg in self.segments]
        if positions != list(range(len(self.segments))):
            raise ValueError(f"source product positions are not 0..n-1: {positions}")
        ids = [seg.segment_id for seg in self.segments]
        if len(ids) != len(set(ids)):
            raise ValueError("source product contains duplicate segment ids")

    def index_by_segment_id(self) -> dict[str, SourceSegment]:
        return {seg.segment_id: seg for seg in self.segments}

    @property
    def segment_ids(self) -> tuple[str, ...]:
        return tuple(seg.segment_id for seg in self.segments)

    @property
    def revision_map(self) -> dict[str, int]:
        return {seg.segment_id: seg.revision for seg in self.segments}

    def snapshot(self) -> list[dict[str, object]]:
        """The package-format source snapshot: identity + revision, in order."""

        return [
            {"segment_id": seg.segment_id, "revision": seg.revision}
            for seg in self.segments
        ]


@dataclass(frozen=True)
class SourceRef:
    """One ``(segment_id, revision)`` reference inside a translation entry."""

    segment_id: str
    revision: int


@dataclass(frozen=True)
class TranslationEntry:
    "Store exactly the revision that was translated; never re-label it."""

    translation_id: str
    document_id: str
    language: str
    text: str
    source_refs: tuple[SourceRef, ...]


@dataclass(frozen=True)
class TranslationAssessment:
    """Result of checking one entry against the *current* source product.

    ``state`` is one of: ``current_usable``, ``stale_revision``,
    ``non_contiguous``, ``unknown_segment``, ``malformed``, ``empty_text``.
    ``start``/``end`` are zero-based half-open positions and are populated
    only for ``current_usable`` entries.
    """

    entry: TranslationEntry
    state: str
    issues: tuple[str, ...]
    start: Optional[int] = None
    end: Optional[int] = None

    @property
    def usable(self) -> bool:
        return self.state == "current_usable"


@dataclass(frozen=True)
class CoverChunk:
    """One selected translation entry, positioned in the source order."""

    translation_id: str
    source_ids: tuple[str, ...]
    source_revisions: tuple[int, ...]
    text: str
    start: int
    end: int  # half-open

    def to_package_chunk(self) -> dict[str, object]:
        return {
            "translation_id": self.translation_id,
            "source_ids": list(self.source_ids),
            "source_revisions": list(self.source_revisions),
            "text": self.text,
        }


@dataclass(frozen=True)
class CoverDiagnosis:
    """Why a complete delivery could not be produced."""

    reason: str
    detail: str
    blocked_boundary: Optional[int]
    uncovered_segment_ids: tuple[str, ...]
    crossing_translation_ids: tuple[str, ...]
    excluded_translations: tuple[tuple[str, str], ...]
    union_covers_all: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "reason": self.reason,
            "detail": self.detail,
            "blocked_boundary": self.blocked_boundary,
            "uncovered_segment_ids": list(self.uncovered_segment_ids),
            "crossing_translation_ids": list(self.crossing_translation_ids),
            "excluded_translations": [list(x) for x in self.excluded_translations],
            "union_covers_all": self.union_covers_all,
        }


@dataclass(frozen=True)
class CoverResult:
    ok: bool
    product: SourceProduct
    language: str
    chunks: tuple[CoverChunk, ...]
    rendered_text: Optional[str]
    diagnosis: Optional[CoverDiagnosis]


# ----------------------------------------------------------------------
# token-level source identity / collation
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SourceToken:
    """One token as it exists at one segment revision.

    ``origin`` is ``initial`` (first material for a segment), ``retained``
    (same identity carried from the previous revision), ``insert`` (brand new
    identity, no sources), ``split`` or ``merge`` (new identity derived from
    ``parents`` in the immediately previous revision).
    """

    document_id: str
    segment_id: str
    revision: int
    token_id: str
    literal: str
    position: int
    origin: str
    parents: tuple[str, ...] = ()


@dataclass(frozen=True)
class TokenEditPlan:
    """The verified final order of an edit batch, before publication.

    ``new_token_ids`` are the identities minted for this batch.  A plan can be
    inspected, then passed back as ``expected_final`` when applying so a stale
    or altered batch cannot silently publish a different order.
    """

    document_id: str
    segment_id: str
    base_revision: int
    new_revision: int
    text: str
    tokens: tuple[SourceToken, ...]
    new_token_ids: tuple[str, ...]

    def final_signature(self) -> tuple[tuple[str, str], ...]:
        return tuple((tok.token_id, tok.literal) for tok in self.tokens)


@dataclass(frozen=True)
class CollationAnnotation:
    """A collation annotation anchored to token identities on one revision."""

    annotation_id: str
    document_id: str
    segment_id: str
    base_revision: int
    selected_token_ids: tuple[str, ...]
    note: str = ""


@dataclass(frozen=True)
class TokenLineage:
    """A token at the target revision and how it relates to the annotation.

    ``base_token_ids`` are the identities from the annotation's base revision
    that this token descends from.  ``unsourced`` is true when any component
    was a pure insertion (no identity continuity from the base revision).
    ``fully_selected`` is true only when *every* component is sourced from the
    annotation's selection; partial matches are mixed and need a human.
    """

    token_id: str
    literal: str
    position: int
    base_token_ids: tuple[str, ...]
    unsourced: bool
    fully_selected: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "token_id": self.token_id,
            "literal": self.literal,
            "position": self.position,
            "base_token_ids": list(self.base_token_ids),
            "unsourced": self.unsourced,
            "fully_selected": self.fully_selected,
        }


@dataclass(frozen=True)
class AnnotationResolution:
    """Where an annotation's selected tokens are at a later revision."""

    annotation_id: str
    document_id: str
    segment_id: str
    base_revision: int
    target_revision: int
    selected_token_ids: tuple[str, ...]
    survivors: tuple[TokenLineage, ...]
    mixed: tuple[TokenLineage, ...]
    deleted_token_ids: tuple[str, ...]
    mixed_original_ids: tuple[str, ...]
    missing_material: bool = False
    detail: str = ""

    @property
    def ok(self) -> bool:
        return not self.missing_material

    def to_dict(self) -> dict[str, object]:
        return {
            "annotation_id": self.annotation_id,
            "document_id": self.document_id,
            "segment_id": self.segment_id,
            "base_revision": self.base_revision,
            "target_revision": self.target_revision,
            "selected_token_ids": list(self.selected_token_ids),
            "survivors": [t.to_dict() for t in self.survivors],
            "mixed": [t.to_dict() for t in self.mixed],
            "deleted_token_ids": list(self.deleted_token_ids),
            "mixed_original_ids": list(self.mixed_original_ids),
            "missing_material": self.missing_material,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class BlockAlignment:
    """A versioned manual alignment for one immutable package block."""

    package_id: str
    block_id: str
    version: int
    document_id: str
    language: str
    source_revisions: tuple[tuple[str, int], ...]
    target_tokens: tuple[str, ...]
    mapping: tuple[tuple[str, tuple[int, ...]], ...]
    target_text: str
    created_at: str = ""

    def mapping_dict(self) -> dict[str, tuple[int, ...]]:
        return dict(self.mapping)

    def source_revision_map(self) -> dict[str, int]:
        return dict(self.source_revisions)

    def to_dict(self) -> dict[str, object]:
        return {
            "package_id": self.package_id,
            "block_id": self.block_id,
            "version": self.version,
            "document_id": self.document_id,
            "language": self.language,
            "source_revisions": {sid: rev for sid, rev in self.source_revisions},
            "target_tokens": list(self.target_tokens),
            "mapping": {tid: list(ps) for tid, ps in self.mapping},
            "target_text": self.target_text,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class ProjectionResult:
    """The proofing view: exact target intervals plus explicit gaps."""

    ok: bool
    reason: Optional[str]
    detail: str
    annotation_id: str
    source_revision: Optional[int]
    package_id: str
    block_id: Optional[str]
    alignment_version: Optional[int]
    intervals: tuple[tuple[int, int], ...] = ()
    target_positions: tuple[int, ...] = ()
    survivors: tuple[TokenLineage, ...] = ()
    mixed: tuple[TokenLineage, ...] = ()
    deleted_token_ids: tuple[str, ...] = ()
    no_correspondence: tuple[TokenLineage, ...] = ()
    unannotated: tuple[TokenLineage, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "detail": self.detail,
            "annotation_id": self.annotation_id,
            "source_revision": self.source_revision,
            "package_id": self.package_id,
            "block_id": self.block_id,
            "alignment_version": self.alignment_version,
            "intervals": [list(iv) for iv in self.intervals],
            "target_positions": list(self.target_positions),
            "survivors": [t.to_dict() for t in self.survivors],
            "mixed": [t.to_dict() for t in self.mixed],
            "deleted_token_ids": list(self.deleted_token_ids),
            "no_correspondence": [t.to_dict() for t in self.no_correspondence],
            "unannotated": [t.to_dict() for t in self.unannotated],
        }
