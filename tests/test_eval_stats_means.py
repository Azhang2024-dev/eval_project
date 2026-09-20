"""P2.5: all-rows mean (failed→0) vs scored-only mean must both be published."""
from __future__ import annotations

import unittest

from scripts.eval_stats import (
    FAILED_PENALTY_SCORE,
    collect_all_row_scores,
    collect_weighted_scores,
    mean_report,
)


def _card(scores):
    dims = [
        "Response Relevance",
        "Citation Accuracy",
        "Scope Completeness",
        "Instruction Following",
        "Readability and Naturalness",
    ]
    return {
        dim: {"score": s, "reason": "r"} for dim, s in zip(dims, scores)
    } | {"Overall Summary": "s"}


class TestMeanHonesty(unittest.TestCase):
    def test_failed_penalty_constant_is_zero(self):
        self.assertEqual(FAILED_PENALTY_SCORE, 0.0)

    def test_all_rows_mean_includes_failed_as_zero(self):
        results = [
            {"index": 0, "status": "success", "result": _card([5, 5, 5, 5, 5])},
            {"index": 1, "status": "failed", "result": None},
            {"index": 2, "status": "success", "result": _card([5, 5, 5, 5, 5])},
        ]
        report = mean_report(results, "en")
        # two perfect 5.0 + one failed 0.0 → headline mean 10/3
        self.assertAlmostEqual(report["mean_all_rows"], 10.0 / 3.0)
        self.assertAlmostEqual(report["mean"], 10.0 / 3.0)
        self.assertAlmostEqual(report["mean_scored_only"], 5.0)
        self.assertEqual(report["n_all"], 3)
        self.assertEqual(report["n_scored"], 2)
        self.assertEqual(report["n_failed"], 1)
        self.assertIn("0.0", report["failed_penalty_policy"])

    def test_scored_only_collector_excludes_failed(self):
        results = [
            {"index": 0, "status": "success", "result": _card([5, 5, 5, 5, 5])},
            {"index": 1, "status": "failed", "result": None},
        ]
        scored = collect_weighted_scores(results, "en")
        all_rows = collect_all_row_scores(results, "en")
        self.assertEqual([i for i, _ in scored], [0])
        self.assertEqual([i for i, _ in all_rows], [0, 1])
        self.assertEqual(all_rows[1][1], 0.0)

    def test_empty_status_is_scored_not_penalized(self):
        results = [
            {"index": 0, "status": "empty", "result": _card([1, 1, 1, 1, 1])},
        ]
        report = mean_report(results, "en")
        self.assertAlmostEqual(report["mean_all_rows"], 1.0)
        self.assertAlmostEqual(report["mean_scored_only"], 1.0)
        self.assertEqual(report["n_failed"], 0)


if __name__ == "__main__":
    unittest.main()
