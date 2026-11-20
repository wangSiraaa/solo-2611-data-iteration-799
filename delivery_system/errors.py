"""Domain errors with actionable messages."""

from __future__ import annotations

from typing import Optional


class DeliveryError(Exception):
    """Base class for expected domain failures."""


class DocumentExists(DeliveryError):
    pass


class DocumentNotFound(DeliveryError):
    pass


class SegmentNotFound(DeliveryError):
    pass


class DuplicateSegment(DeliveryError):
    pass


class InvalidPosition(DeliveryError):
    pass


class RevisionConflict(DeliveryError):
    pass


class TranslationExists(DeliveryError):
    pass


class InvalidTranslation(DeliveryError):
    pass


class PackageExistsImmutable(DeliveryError):
    """A package id was reused with different frozen content."""


class NoCoverError(DeliveryError):
    """Publishing was attempted but no exact current cover exists."""

    def __init__(self, message: str, result: Optional[object] = None) -> None:
        super().__init__(message)
        self.result = result


# ----------------------------------------------------------------------
# token identity / collation / alignment errors
# ----------------------------------------------------------------------
class TokenRevisionConflict(DeliveryError):
    """The edit batch targets a base revision that is no longer current."""


class TokenMaterialMissing(DeliveryError):
    """A collation request touches a segment that has no token material.

    This is intentionally a distinct, identifiable reason (``missing_token_material``)
    so callers can tell "old plain-text segment" apart from "bad request".
    """

    reason = "missing_token_material"


class InvalidTokenEdit(DeliveryError):
    """An explicit token edit operation is malformed or inconsistent."""


class InvalidAlignment(DeliveryError):
    """A block alignment does not match the actual package / source."""


class AlignmentConflict(DeliveryError):
    """An alignment version was reused with different immutable content."""


class PackageBlockNotFound(DeliveryError):
    """The requested block does not exist in the frozen package."""


class SourcePackageMismatch(DeliveryError):
    """The package's frozen source snapshot does not match the requested revision."""

