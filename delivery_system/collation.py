"""Token identity, lineage and explicit edit batches.

This module is deliberately free of storage: it works on plain ordered token
lists and revision rows.  The catalog persists what these functions validate.

Design rules (mirrored from the brief):

* Identity is ``token_id``; literal spelling carries no identity.
* ``keep`` / ``move`` reuse an existing identity; ``insert`` always mints a
  *fresh* identity even when the literal equals a just-deleted token.
* ``split`` / ``merge`` create new identities with explicit ``parents`` in the
  immediately previous revision.  A token with ``origin='insert'`` has no
  parents, so "sourced rewrite" and "brand-new insertion" stay distinguishable.
* An edit batch is anchored to a ``base_revision`` and must publish a verified
  final order; a stale base can never overwrite newer material.
"""

from __future__ import annotations

import secrets
from typing import Iterable, Optional

from .errors import InvalidTokenEdit
from .models import SourceToken, TokenLineage

ORIGIN_INITIAL = "initial"
ORIGIN_RETAINED = "retained"
ORIGIN_INSERT = "insert"
ORIGIN_SPLIT = "split"
ORIGIN_MERGE = "merge"


def mint_token_id(prefix: str = "t") -> str:
    return f"{prefix}_{secrets.token_hex(6)}"


def normalize_material(
    specs: Iterable[dict[str, object]],
    *,
    minted: Optional[set[str]] = None,
) -> list[tuple[str, str]]:
    """Turn ordered ``{token_id?, literal}`` specs into ``(id, literal)`` pairs.

    Identities are minted when the caller only supplies literals.  Minted ids
    are recorded in ``minted`` so a caller can report them.
    """

    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for spec in specs:
        if "literal" not in spec:
            raise InvalidTokenEdit(f"token spec missing 'literal': {spec!r}")
        literal = str(spec["literal"])
        token_id = spec.get("token_id")
        if token_id is None:
            token_id = mint_token_id()
        token_id = str(token_id)
        if not token_id:
            raise InvalidTokenEdit("token_id must be non-empty")
        if token_id in seen:
            raise InvalidTokenEdit(f"duplicate token_id {token_id!r} in material")
        seen.add(token_id)
        if minted is not None and "token_id" not in spec:
            minted.add(token_id)
        out.append((token_id, literal))
    return out


def apply_batch(
    previous: Iterable[tuple[str, str]],
    ops: Iterable[dict[str, object]],
    *,
    used_ids: frozenset[str] = frozenset(),
) -> tuple[list[SourceToken], tuple[str, ...]]:
    """Apply an explicit ops batch to the previous ordered token list.

    Returns ``(tokens, new_ids)`` where tokens carry ``origin``/``parents``.
    ``used_ids`` are identities that already exist anywhere for this segment
    (any revision); no op may reuse them.
    """

    prev = list(previous)
    prev_ids = [tid for tid, _ in prev]
    prev_literal = dict(prev)
    prev_index = {tid: i for i, tid in enumerate(prev_ids)}

    placed: set[str] = set()
    deleted: set[str] = set()
    consumed: set[str] = set()  # keep + split + merge parents + delete
    new_ids: list[str] = []
    result: list[SourceToken] = []
    minted: set[str] = set()

    def fresh(candidate: Optional[str]) -> str:
        if candidate is None:
            token_id = mint_token_id()
        else:
            token_id = str(candidate)
            if not token_id:
                raise InvalidTokenEdit("new token_id must be non-empty")
            if token_id in used_ids or token_id in prev_literal or token_id in minted:
                raise InvalidTokenEdit(
                    f"token identity {token_id!r} was already used; insert/split/merge "
                    "must mint a never-before-used identity (even for the same literal)"
                )
        minted.add(token_id)
        new_ids.append(token_id)
        return token_id

    def claim_base(token_id: str, op: str) -> str:
        token_id = str(token_id)
        if token_id not in prev_literal:
            raise InvalidTokenEdit(f"{op} references unknown base token {token_id!r}")
        if token_id in consumed:
            raise InvalidTokenEdit(f"base token {token_id!r} is used more than once")
        consumed.add(token_id)
        return token_id

    for raw in ops:
        op = str(raw.get("op", ""))
        if op in ("keep", "move"):
            token_id = claim_base(raw["token_id"], op)
            placed.add(token_id)
            result.append(
                SourceToken(
                    document_id="",
                    segment_id="",
                    revision=0,
                    token_id=token_id,
                    literal=prev_literal[token_id],
                    position=len(result),
                    origin=ORIGIN_RETAINED,
                    parents=(token_id,),
                )
            )
        elif op == "insert":
            literal = str(raw["literal"])
            token_id = fresh(raw.get("token_id"))
            result.append(
                SourceToken(
                    document_id="",
                    segment_id="",
                    revision=0,
                    token_id=token_id,
                    literal=literal,
                    position=len(result),
                    origin=ORIGIN_INSERT,
                    parents=(),
                )
            )
        elif op == "delete":
            token_id = claim_base(raw["token_id"], "delete")
            deleted.add(token_id)
        elif op == "split":
            parent = claim_base(raw["token_id"], "split")
            literals = list(raw.get("literals") or [])
            ids = list(raw.get("new_token_ids") or [])
            if len(literals) < 2:
                raise InvalidTokenEdit("split needs at least two literals")
            if len(ids) not in (0, len(literals)):
                raise InvalidTokenEdit("split new_token_ids must match literals length")
            for i, literal in enumerate(literals):
                candidate = ids[i] if ids else None
                token_id = fresh(candidate)
                result.append(
                    SourceToken(
                        document_id="",
                        segment_id="",
                        revision=0,
                        token_id=token_id,
                        literal=str(literal),
                        position=len(result),
                        origin=ORIGIN_SPLIT,
                        parents=(parent,),
                    )
                )
        elif op == "merge":
            parents = [str(x) for x in (raw.get("token_ids") or [])]
            if len(parents) < 2:
                raise InvalidTokenEdit("merge needs at least two tokens")
            for parent in parents:
                claim_base(parent, "merge")
            order = [prev_index[p] for p in parents]
            if order != list(range(order[0], order[0] + len(order))):
                raise InvalidTokenEdit(
                    "merge parents must be adjacent in the previous revision order: "
                    f"{parents!r}"
                )
            token_id = fresh(raw.get("token_id"))
            result.append(
                SourceToken(
                    document_id="",
                    segment_id="",
                    revision=0,
                    token_id=token_id,
                    literal=str(raw["literal"]),
                    position=len(result),
                    origin=ORIGIN_MERGE,
                    parents=tuple(parents),
                )
            )
        else:
            raise InvalidTokenEdit(f"unknown token op {op!r}")

    missing = [tid for tid in prev_ids if tid not in consumed]
    if missing:
        raise InvalidTokenEdit(
            "base tokens neither placed nor explicitly deleted: " + ",".join(missing)
        )
    return result, tuple(new_ids)


# ----------------------------------------------------------------------
# lineage
# ----------------------------------------------------------------------
# ``rows_by_revision`` maps revision -> {token_id: (literal, position, origin, parents)}
Rows = dict[str, dict[int, dict[str, tuple[str, int, str, tuple[str, ...]]]]]


def _resolve(
    rows: Rows,
    base_revision: int,
    revision: int,
    token_id: str,
    memo: dict[tuple[int, str], tuple[frozenset[str], bool]],
) -> tuple[frozenset[str], bool]:
    """Return (base ancestors, unsourced) for one token at ``revision``.

    ``unsourced`` is true when some branch terminates in an ``insert`` above
    the base revision (i.e. has no identity continuity from the base).
    """

    key = (revision, token_id)
    if key in memo:
        return memo[key]
    if revision == base_revision:
        ans = (frozenset({token_id}), False)
        memo[key] = ans
        return ans
    if revision < base_revision:
        raise InvalidTokenEdit("target revision precedes the annotation base revision")
    rev_rows = rows.get(revision)
    if rev_rows is None or token_id not in rev_rows:
        raise InvalidTokenEdit(
            f"token {token_id!r} not present at revision {revision}"
        )
    literal, _pos, origin, parents = rev_rows[token_id]
    if origin == ORIGIN_INSERT:
        ans = (frozenset(), True)
    elif origin == ORIGIN_MERGE or origin == ORIGIN_SPLIT:
        ancestors: set[str] = set()
        unsourced = False
        for parent in parents:
            sub, sub_unsourced = _resolve(rows, base_revision, revision - 1, parent, memo)
            ancestors.update(sub)
            unsourced = unsourced or sub_unsourced
        ans = (frozenset(ancestors), unsourced)
    elif origin == ORIGIN_RETAINED:
        if not parents:
            ans = (frozenset({token_id}), False)
        else:
            ans = _resolve(rows, base_revision, revision - 1, parents[0], memo)
    else:  # initial material introduced above the base revision
        ans = (frozenset(), True)
    memo[key] = ans
    return ans


def resolve_selection(
    rows: Rows,
    base_revision: int,
    target_revision: int,
    selected_token_ids: Iterable[str],
) -> tuple[list[TokenLineage], list[TokenLineage], list[str], list[str]]:
    """Resolve an annotation forward to ``target_revision``.

    Returns ``(survivors, mixed, deleted_ids, mixed_original_ids)``.

    A token survives automatically only when *all* of its components descend
    from the annotation's selection.  A token that merges a selected component
    with an unselected or newly-inserted one is *mixed* and is returned
    separately for a human, never auto-highlighted.
    """

    selected = frozenset(str(x) for x in selected_token_ids)
    target_rows = rows.get(target_revision)
    if target_rows is None:
        raise InvalidTokenEdit(f"revision {target_revision} has no token material")

    memo: dict[tuple[int, str], tuple[frozenset[str], bool]] = {}
    survivors: list[TokenLineage] = []
    mixed: list[TokenLineage] = []
    covered: set[str] = set()

    for token_id, (literal, position, _origin, _parents) in sorted(
        target_rows.items(), key=lambda kv: kv[1][1]
    ):
        ancestors, unsourced = _resolve(
            rows, base_revision, target_revision, token_id, memo
        )
        touched = ancestors & selected
        if not touched:
            # A pure insertion after the base has no ancestry at all; it is not
            # part of the original selection and is not surfaced here.
            continue
        covered.update(touched)
        fully = (not unsourced) and touched == ancestors
        lineage = TokenLineage(
            token_id=token_id,
            literal=literal,
            position=position,
            base_token_ids=tuple(sorted(ancestors)),
            unsourced=unsourced,
            fully_selected=fully,
        )
        if fully:
            survivors.append(lineage)
        else:
            mixed.append(lineage)

    deleted = sorted(selected - covered)
    mixed_originals: set[str] = set()
    for item in mixed:
        mixed_originals.update(set(item.base_token_ids) & selected)
    return survivors, mixed, deleted, sorted(mixed_originals)
