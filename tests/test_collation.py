"""Tests for token identity, collation annotations and bilingual projection.

Run with ``python -m unittest discover -s tests``.
"""

from __future__ import annotations

import os
import tempfile
import unittest

from delivery_system.delivery_repository import DeliveryRepository
from delivery_system.errors import (
    AlignmentConflict,
    InvalidAlignment,
    InvalidTokenEdit,
    SourcePackageMismatch,
    TokenMaterialMissing,
    TokenRevisionConflict,
)
from delivery_system.source_catalog import PublishedSourceReader, SourceCatalog


class CollationCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.source_db = os.path.join(self.tmp, "source.db")
        self.delivery_db = os.path.join(self.tmp, "delivery.db")
        self.catalog = SourceCatalog(self.source_db)
        self.repo = DeliveryRepository(
            self.delivery_db, PublishedSourceReader(self.source_db)
        )

    def tearDown(self):
        self._tmp.cleanup()

    def reopen(self):
        self.catalog = SourceCatalog(self.source_db)
        self.repo = DeliveryRepository(
            self.delivery_db, PublishedSourceReader(self.source_db)
        )

    def make_abc(self, document_id="DOC"):
        self.catalog.create_document(
            document_id, [{"segment_id": "S", "text": "甲 乙 甲"}]
        )
        self.catalog.set_initial_tokens(
            document_id,
            "S",
            [
                {"token_id": "a", "literal": "甲"},
                {"token_id": "b", "literal": "乙"},
                {"token_id": "c", "literal": "甲"},
            ],
        )

    def insert_n(self, document_id="DOC"):
        return self.catalog.apply_token_edit(
            document_id,
            "S",
            1,
            [
                {"op": "keep", "token_id": "a"},
                {"op": "keep", "token_id": "b"},
                {"op": "insert", "token_id": "n", "literal": "甲"},
                {"op": "keep", "token_id": "c"},
            ],
        )


class TokenIdentityTests(CollationCase):
    def test_identity_and_literal_are_separate(self):
        self.make_abc()
        self.catalog.add_annotation("ANN", "DOC", "S", 1, ["b", "c"])
        self.insert_n()
        tokens = {t.token_id: t.literal for t in self.catalog.list_tokens("DOC", "S", 2)}
        self.assertEqual(tokens["b"], "乙")
        self.assertEqual(tokens["c"], "甲")
        self.assertEqual(tokens["n"], "甲")  # same literal, distinct identity

    def test_keep_reuses_and_insert_mints_fresh(self):
        self.make_abc()
        plan = self.catalog.plan_token_edit(
            "DOC", "S", 1, [{"op": "keep", "token_id": "a"}, {"op": "keep", "token_id": "b"}, {"op": "insert", "literal": "甲"}, {"op": "keep", "token_id": "c"}]
        )
        new_id = plan.new_token_ids[0]
        self.assertNotIn(new_id, {"a", "b", "c"})
        self.catalog.publish_token_plan(plan)
        # retained tokens keep their ids
        ids = [t.token_id for t in self.catalog.list_tokens("DOC", "S", 2)]
        self.assertEqual(ids[:2], ["a", "b"])
        self.assertEqual(ids[3], "c")

    def test_reusing_deleted_identity_rejected(self):
        self.catalog.create_document("D", [{"segment_id": "S", "text": "甲"}])
        self.catalog.set_initial_tokens("D", "S", [{"token_id": "a", "literal": "甲"}])
        self.catalog.apply_token_edit(
            "D", "S", 1, [{"op": "delete", "token_id": "a"}, {"op": "insert", "literal": "甲"}]
        )
        # try to resurrect 'a' for a brand-new insert
        with self.assertRaises(InvalidTokenEdit):
            self.catalog.apply_token_edit(
                "D",
                "S",
                2,
                [
                    {"op": "delete", "token_id": self.catalog.list_tokens("D", "S", 2)[0].token_id},
                    {"op": "insert", "token_id": "a", "literal": "甲"},
                ],
            )

    def test_stale_baseline_cannot_overwrite(self):
        self.make_abc()
        stale = self.catalog.plan_token_edit(
            "DOC",
            "S",
            1,
            [
                {"op": "keep", "token_id": "a"},
                {"op": "keep", "token_id": "b"},
                {"op": "keep", "token_id": "c"},
                {"op": "insert", "token_id": "z", "literal": "丙"},
            ],
        )
        # someone publishes first
        self.catalog.apply_token_edit(
            "DOC",
            "S",
            1,
            [
                {"op": "keep", "token_id": "a"},
                {"op": "delete", "token_id": "b"},
                {"op": "delete", "token_id": "c"},
                {"op": "insert", "literal": "丁"},
            ],
        )
        with self.assertRaises(TokenRevisionConflict):
            self.catalog.publish_token_plan(stale)

    def test_verified_final_order_guard(self):
        self.make_abc()
        plan = self.catalog.plan_token_edit(
            "DOC",
            "S",
            1,
            [
                {"op": "keep", "token_id": "a"},
                {"op": "keep", "token_id": "b"},
                {"op": "keep", "token_id": "c"},
                {"op": "insert", "literal": "丙"},
            ],
        )
        wrong = (("a", "甲"), ("b", "乙"), ("c", "甲"), ("zzz", "丙"))
        with self.assertRaises(InvalidTokenEdit):
            self.catalog.publish_token_plan(plan, expected_final=wrong)
        self.catalog.publish_token_plan(plan, expected_final=plan.final_signature())

    def test_merge_requires_adjacency(self):
        self.make_abc()
        with self.assertRaises(InvalidTokenEdit):
            self.catalog.apply_token_edit(
                "DOC",
                "S",
                1,
                [
                    {"op": "merge", "token_ids": ["a", "c"], "literal": "甲甲"},
                    {"op": "keep", "token_id": "b"},
                ],
            )

    def test_missing_token_material_is_identifiable(self):
        self.catalog.create_document(
            "D", [{"segment_id": "PLAIN", "text": "plain text"}]
        )
        with self.assertRaises(TokenMaterialMissing) as ctx:
            self.catalog.list_tokens("D", "PLAIN")
        self.assertEqual(ctx.exception.reason, "missing_token_material")
        # the original plain-text interface still works
        self.assertEqual(self.catalog.get_segment("D", "PLAIN").text, "plain text")
        with self.assertRaises(TokenMaterialMissing):
            self.catalog.add_annotation("X", "D", "PLAIN", 1, ["a"])


class LineageTests(CollationCase):
    def test_inserted_token_not_auto_included(self):
        self.make_abc()
        self.catalog.add_annotation("ANN", "DOC", "S", 1, ["b", "c"])
        self.insert_n()
        res = self.catalog.resolve_annotation("ANN", 2)
        self.assertEqual([t.token_id for t in res.survivors], ["b", "c"])
        self.assertNotIn("n", [t.token_id for t in res.survivors])

    def test_deleted_reported(self):
        self.make_abc()
        self.catalog.add_annotation("ANN", "DOC", "S", 1, ["b", "c"])
        self.insert_n()
        self.catalog.apply_token_edit(
            "DOC",
            "S",
            2,
            [
                {"op": "keep", "token_id": "a"},
                {"op": "keep", "token_id": "b"},
                {"op": "keep", "token_id": "n"},
                {"op": "delete", "token_id": "c"},
                {"op": "insert", "token_id": "m", "literal": "甲"},
            ],
        )
        res = self.catalog.resolve_annotation("ANN", 3)
        self.assertEqual(res.deleted_token_ids, ("c",))
        self.assertNotIn("m", [t.token_id for t in res.survivors])

    def test_split_then_merge_mixed(self):
        self.make_abc()
        self.catalog.add_annotation("ANN", "DOC", "S", 1, ["b", "c"])
        self.catalog.apply_token_edit(
            "DOC",
            "S",
            1,
            [
                {"op": "keep", "token_id": "a"},
                {"op": "split", "token_id": "b", "literals": ["乙1", "乙2"], "new_token_ids": ["b1", "b2"]},
                {"op": "insert", "token_id": "n", "literal": "丙"},
                {"op": "keep", "token_id": "c"},
            ],
        )
        self.catalog.apply_token_edit(
            "DOC",
            "S",
            2,
            [
                {"op": "keep", "token_id": "a"},
                {"op": "keep", "token_id": "b1"},
                {"op": "merge", "token_ids": ["b2", "n"], "literal": "乙2丙", "token_id": "x"},
                {"op": "keep", "token_id": "c"},
            ],
        )
        res = self.catalog.resolve_annotation("ANN", 3)
        survivors = {t.token_id for t in res.survivors}
        mixed = {t.token_id for t in res.mixed}
        self.assertIn("b1", survivors)  # all components selected -> continues
        self.assertIn("x", mixed)  # part selected, part inserted -> human
        self.assertNotIn("x", survivors)
        self.assertIn("b", res.mixed_original_ids)

    def test_lineage_across_multiple_revisions(self):
        self.catalog.create_document("D", [{"segment_id": "S", "text": "x y"}])
        self.catalog.set_initial_tokens(
            "D", "S", [{"token_id": "a", "literal": "x"}, {"token_id": "b", "literal": "y"}]
        )
        self.catalog.add_annotation("A", "D", "S", 1, ["b"])
        self.catalog.apply_token_edit(
            "D", "S", 1, [{"op": "keep", "token_id": "a"}, {"op": "keep", "token_id": "b"}, {"op": "insert", "token_id": "c", "literal": "z"}]
        )
        self.catalog.apply_token_edit(
            "D",
            "S",
            2,
            [
                {"op": "keep", "token_id": "a"},
                {"op": "keep", "token_id": "b"},
                {"op": "keep", "token_id": "c"},
                {"op": "insert", "token_id": "d", "literal": "w"},
            ],
        )
        res = self.catalog.resolve_annotation("A", 3)
        self.assertEqual([t.token_id for t in res.survivors], ["b"])


class ProjectionTests(CollationCase):
    def _setup_p1(self):
        self.make_abc()
        self.catalog.add_annotation("ANN", "DOC", "S", 1, ["b", "c"])
        self.insert_n()
        self.repo.add_translation(
            "TR", "DOC", "EN", "pqrs", [{"segment_id": "S", "revision": 2}]
        )
        self.repo.publish_language("DOC", "EN", "P1")

    def test_original_alignment_intervals(self):
        self._setup_p1()
        self.repo.add_alignment(
            "P1", "TR", 1, ["p", "q", "r", "s"], {"b": [3], "c": [0], "n": [1]}
        )
        proj = self.repo.project_annotation("ANN", 2, "P1", 1)
        self.assertEqual(proj.intervals, ((0, 1), (3, 4)))

    def test_corrected_alignment_is_new_version(self):
        self._setup_p1()
        self.repo.add_alignment("P1", "TR", 1, ["p", "q", "r", "s"], {"b": [3], "c": [0], "n": [1]})
        self.repo.add_alignment("P1", "TR", 2, ["p", "q", "r", "s"], {"b": [2], "c": [3]})
        self.assertEqual(self.repo.project_annotation("ANN", 2, "P1", 2).intervals, ((2, 4),))

    def test_alignment_version_immutable(self):
        self._setup_p1()
        self.repo.add_alignment("P1", "TR", 1, ["p", "q", "r", "s"], {"b": [3], "c": [0]})
        with self.assertRaises(AlignmentConflict):
            self.repo.add_alignment("P1", "TR", 1, ["p", "q", "r", "s"], {"b": [0]})

    def test_many_to_one_highlighted_once(self):
        self._setup_p1()
        # both b and c map to position 2; the intervening position 1 is untouched
        self.repo.add_alignment("P1", "TR", 1, ["p", "q", "r", "s"], {"b": [2], "c": [2]})
        proj = self.repo.project_annotation("ANN", 2, "P1", 1)
        self.assertEqual(proj.intervals, ((2, 3),))

    def test_explicit_no_correspondence_and_unannotated(self):
        self._setup_p1()
        self.repo.add_alignment("P1", "TR", 1, ["p", "q", "r", "s"], {"b": [], "c": [1]})
        proj = self.repo.project_annotation("ANN", 2, "P1", 1)
        self.assertEqual(proj.intervals, ((1, 2),))
        self.assertEqual([t.token_id for t in proj.no_correspondence], ["b"])
        self.repo.add_alignment("P1", "TR", 2, ["p", "q", "r", "s"], {"b": [3]})
        proj2 = self.repo.project_annotation("ANN", 2, "P1", 2)
        self.assertEqual([t.token_id for t in proj2.unannotated], ["c"])

    def test_target_token_string_must_match_package(self):
        self._setup_p1()
        with self.assertRaises(InvalidAlignment):
            self.repo.add_alignment("P1", "TR", 1, ["x", "y"], {"b": [0]})

    def test_source_token_must_exist(self):
        self._setup_p1()
        with self.assertRaises(InvalidAlignment):
            self.repo.add_alignment("P1", "TR", 1, ["p", "q", "r", "s"], {"zzz": [0]})

    def test_source_package_revision_mismatch_rejected(self):
        self._setup_p1()
        self.repo.add_alignment("P1", "TR", 1, ["p", "q", "r", "s"], {"b": [3], "c": [0]})
        with self.assertRaises(SourcePackageMismatch):
            self.repo.project_annotation("ANN", 1, "P1", 1)

    def test_delete_then_reinsert_same_literal(self):
        self._setup_p1()
        self.catalog.apply_token_edit(
            "DOC",
            "S",
            2,
            [
                {"op": "keep", "token_id": "a"},
                {"op": "keep", "token_id": "b"},
                {"op": "keep", "token_id": "n"},
                {"op": "delete", "token_id": "c"},
                {"op": "insert", "token_id": "m", "literal": "甲"},
            ],
        )
        self.repo.add_translation(
            "TR3", "DOC", "EN", "pqrs", [{"segment_id": "S", "revision": 3}]
        )
        self.repo.publish_language("DOC", "EN", "P2")
        self.repo.add_alignment("P2", "TR3", 1, ["p", "q", "r", "s"], {"b": [3]})
        proj = self.repo.project_annotation("ANN", 3, "P2", 1)
        self.assertEqual(proj.intervals, ((3, 4),))
        self.assertEqual(proj.deleted_token_ids, ("c",))

    def test_reopen_reproduces_same_version_combination(self):
        self._setup_p1()
        self.repo.add_alignment("P1", "TR", 1, ["p", "q", "r", "s"], {"b": [3], "c": [0], "n": [1]})
        first = self.repo.project_annotation("ANN", 2, "P1", 1)
        self.reopen()
        second = self.repo.project_annotation("ANN", 2, "P1", 1)
        self.assertEqual(first.intervals, second.intervals)
        self.assertEqual(first.target_positions, second.target_positions)
        self.assertEqual(self.repo.get_alignment("P1", "TR").version, 1)


if __name__ == "__main__":
    unittest.main()
