"""Unit tests for scripts.eval_stats — the canonical evaluation statistics module."""

from __future__ import annotations

import math
import unittest
from typing import Any

from scripts.eval_stats import (
    BADCASE_THRESHOLD,
    LANGUAGE_META,
    WEIGHTS,
    badcase_entries,
    band_distribution,
    collect_dim_values,
    collect_weighted_scores,
    parse_score,
    score_headers,
    summarize,
    valid_score_count,
    weighted_from_score_dict,
    weighted_from_values,
)


class TestParseScore(unittest.TestCase):
    """parse_score boundaries — 9+ cases matching ``result_merger._safe_score``."""

    def test_valid_int(self) -> None:
        self.assertEqual(parse_score(4), 4)

    def test_valid_float_point_zero(self) -> None:
        self.assertEqual(parse_score(4.0), 4)

    def test_valid_string_number(self) -> None:
        self.assertEqual(parse_score("4"), 4)

    def test_non_integer_string(self) -> None:
        self.assertIsNone(parse_score("4.5"))

    def test_non_numeric_string(self) -> None:
        self.assertIsNone(parse_score("四"))

    def test_bool_rejected(self) -> None:
        self.assertIsNone(parse_score(True))
        self.assertIsNone(parse_score(False))

    def test_none(self) -> None:
        self.assertIsNone(parse_score(None))

    def test_out_of_range_high(self) -> None:
        self.assertIsNone(parse_score(6))

    def test_out_of_range_low(self) -> None:
        self.assertIsNone(parse_score(0))

    def test_float_boundary_low(self) -> None:
        self.assertIsNone(parse_score(0.0))

    def test_float_boundary_high(self) -> None:
        self.assertIsNone(parse_score(5.5))

    def test_nan(self) -> None:
        self.assertIsNone(parse_score(math.nan))

    def test_empty_string(self) -> None:
        self.assertIsNone(parse_score(""))

    def test_whitespace_string(self) -> None:
        self.assertIsNone(parse_score("   "))


class TestWeightedFromValues(unittest.TestCase):
    """weighted_from_values hand calculation verification."""

    def test_known_values(self) -> None:
        # 5*0.3 + 4*0.3 + 3*0.15 + 5*0.15 + 4*0.1
        # = 1.5 + 1.2 + 0.45 + 0.75 + 0.4 = 4.3
        result = weighted_from_values([5, 4, 3, 5, 4])
        self.assertAlmostEqual(result, 4.3)

    def test_all_zero(self) -> None:
        result = weighted_from_values([0, 0, 0, 0, 0])
        self.assertEqual(result, 0.0)

    def test_all_max(self) -> None:
        result = weighted_from_values([5, 5, 5, 5, 5])
        self.assertEqual(result, 5.0)

    def test_weights_sum_to_one(self) -> None:
        self.assertAlmostEqual(sum(WEIGHTS), 1.0)


class TestWeightedFromScoreDict(unittest.TestCase):
    """weighted_from_score_dict — all-valid vs one-missing."""

    def test_all_valid(self) -> None:
        scores: dict[str, int | None] = {
            "Response Relevance": 5,
            "Citation Accuracy": 4,
            "Scope Completeness": 3,
            "Instruction Following": 5,
            "Readability and Naturalness": 4,
        }
        result = weighted_from_score_dict(scores, "en")
        # 5*0.3 + 4*0.3 + 3*0.15 + 5*0.15 + 4*0.1 = 4.3
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 4.3)

    def test_one_dimension_none(self) -> None:
        scores: dict[str, int | None] = {
            "Response Relevance": 5,
            "Citation Accuracy": 4,
            "Scope Completeness": None,
            "Instruction Following": 5,
            "Readability and Naturalness": 4,
        }
        self.assertIsNone(weighted_from_score_dict(scores, "en"))

    def test_all_none(self) -> None:
        scores: dict[str, int | None] = {
            "Response Relevance": None,
            "Citation Accuracy": None,
            "Scope Completeness": None,
            "Instruction Following": None,
            "Readability and Naturalness": None,
        }
        self.assertIsNone(weighted_from_score_dict(scores, "en"))

    def test_zh(self) -> None:
        scores: dict[str, int | None] = {
            "回答相关性": 5,
            "引用准确性": 5,
            "范围完整性": 5,
            "指令遵循": 5,
            "阅读感受与自然度": 5,
        }
        result = weighted_from_score_dict(scores, "zh")
        self.assertIsNotNone(result)
        self.assertEqual(result, 5.0)

    def test_unknown_language_falls_back_to_en(self) -> None:
        scores: dict[str, int | None] = {
            "Response Relevance": 5,
            "Citation Accuracy": 5,
            "Scope Completeness": 5,
            "Instruction Following": 5,
            "Readability and Naturalness": 5,
        }
        result = weighted_from_score_dict(scores, "fr")
        self.assertIsNotNone(result)
        self.assertEqual(result, 5.0)


class TestBandDistribution(unittest.TestCase):
    """band_distribution boundary tests."""

    def _check(self, value: float, expected_band: str) -> None:
        bands = band_distribution([value])
        for band, count in bands.items():
            if band == expected_band:
                self.assertEqual(count, 1, f"{value} should be in {expected_band}")
            else:
                self.assertEqual(count, 0, f"{value} should NOT be in {band}")

    def test_band_1_0_to_1_9(self) -> None:
        self._check(1.99, "1.0-1.9")

    def test_band_2_0_to_2_9_lower(self) -> None:
        self._check(2.0, "2.0-2.9")

    def test_band_2_0_to_2_9_upper(self) -> None:
        self._check(2.99, "2.0-2.9")

    def test_band_3_0_to_3_9_lower(self) -> None:
        self._check(3.0, "3.0-3.9")

    def test_band_3_0_to_3_9_upper(self) -> None:
        self._check(3.99, "3.0-3.9")

    def test_band_4_0_to_4_5_lower(self) -> None:
        self._check(4.0, "4.0-4.5")

    def test_band_4_0_to_4_5_upper(self) -> None:
        self._check(4.49, "4.0-4.5")

    def test_band_4_5_to_5_0_lower(self) -> None:
        self._check(4.5, "4.5-5.0")

    def test_band_4_5_to_5_0_upper(self) -> None:
        self._check(5.0, "4.5-5.0")

    def test_multiple_values(self) -> None:
        bands = band_distribution([1.5, 2.5, 3.5, 4.2, 4.8, 2.0, 4.0])
        self.assertEqual(bands["1.0-1.9"], 1)
        self.assertEqual(bands["2.0-2.9"], 2)  # 2.5, 2.0
        self.assertEqual(bands["3.0-3.9"], 1)
        self.assertEqual(bands["4.0-4.5"], 2)  # 4.2, 4.0
        self.assertEqual(bands["4.5-5.0"], 1)


class TestCollectWeightedScores(unittest.TestCase):
    """collect_weighted_scores — index-aligned collection of weighted scores."""

    @staticmethod
    def _scorecard(language: str, scores: list[int]) -> dict[str, Any]:
        meta = LANGUAGE_META[language]
        result: dict[str, Any] = {}
        for dim, value in zip(meta.dims, scores):
            result[dim] = {meta.score_key: value, meta.reason_key: "r"}
        result[meta.summary_key] = "s"
        return result

    def test_failed_rows_do_not_shift_alignment(self) -> None:
        """A failed row between successes must not shift or mask scores.

        Regression: the old ``zip(results, weighted_scores)`` pairing lost the
        score of the second success and paired the failed row with a foreign
        score. Pairs are now aligned by the explicit ``index`` field.
        """
        results = [
            {"index": 0, "status": "success", "result": self._scorecard("en", [5, 5, 5, 5, 5])},
            {"index": 3, "status": "failed", "result": {}},
            {"index": 5, "status": "success", "result": self._scorecard("en", [1, 2, 1, 2, 1])},
        ]
        pairs = collect_weighted_scores(results, "en")
        self.assertEqual([index for index, _ in pairs], [0, 5])
        self.assertAlmostEqual(pairs[1][1], 1.45)

    def test_out_of_order_results_still_aligned(self) -> None:
        results = [
            {"index": 5, "status": "success", "result": self._scorecard("en", [1, 2, 1, 2, 1])},
            {"index": 0, "status": "success", "result": self._scorecard("en", [5, 5, 5, 5, 5])},
        ]
        pairs = collect_weighted_scores(results, "en")
        self.assertEqual([index for index, _ in pairs], [5, 0])
        self.assertAlmostEqual(pairs[0][1], 1.45)
        self.assertAlmostEqual(pairs[1][1], 5.0)

    def test_empty_rows_included(self) -> None:
        results = [
            {"index": 2, "status": "empty", "result": self._scorecard("en", [1, 1, 1, 1, 1])},
        ]
        pairs = collect_weighted_scores(results, "en")
        self.assertEqual(pairs, [(2, 1.0)])

    def test_rows_without_index_are_skipped(self) -> None:
        results = [
            {"status": "success", "result": self._scorecard("en", [5, 5, 5, 5, 5])},
        ]
        self.assertEqual(collect_weighted_scores(results, "en"), [])

    def test_invalid_scores_count_as_zero(self) -> None:
        result: dict[str, Any] = {
            "Response Relevance": {"score": 5, "reason": "r"},
            "Citation Accuracy": {},
            "Scope Completeness": {"score": 3, "reason": "r"},
            "Instruction Following": {"score": 5, "reason": "r"},
            "Readability and Naturalness": {"score": 4, "reason": "r"},
            "Overall Summary": "s",
        }
        results = [{"index": 0, "status": "success", "result": result}]
        pairs = collect_weighted_scores(results, "en")
        # 5*0.3 + 0*0.3 + 3*0.15 + 5*0.15 + 4*0.1 = 1.5+0+0.45+0.75+0.4 = 3.1
        self.assertAlmostEqual(pairs[0][1], 3.1)


class TestBadcaseEntries(unittest.TestCase):
    """badcase_entries — aligned pairs, threshold strictly below."""

    def test_threshold_exclusive(self) -> None:
        """Exactly 3.0 is NOT a badcase (threshold is strict <)."""
        pairs = [(0, 2.99), (1, 3.0)]
        self.assertEqual(badcase_entries(pairs), [(0, 2.99)])

    def test_aligned_pairs_with_gaps(self) -> None:
        """Failed rows leave gaps in indices; scores must not shift across them."""
        pairs = [(2, 1.0), (5, 1.45)]
        self.assertEqual(badcase_entries(pairs), [(2, 1.0), (5, 1.45)])

    def test_no_badcases(self) -> None:
        self.assertEqual(badcase_entries([(0, 4.0), (1, 4.5)]), [])

    def test_custom_threshold(self) -> None:
        self.assertEqual(badcase_entries([(0, 3.5), (1, 4.5)], threshold=4.0), [(0, 3.5)])

    def test_default_threshold_constant(self) -> None:
        self.assertEqual(BADCASE_THRESHOLD, 3.0)


class TestSummarize(unittest.TestCase):
    """summarize — statistics computation."""

    def test_basic_stats(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        s = summarize(values)
        self.assertAlmostEqual(s["mean"], 3.0)
        self.assertEqual(s["median"], 3.0)
        self.assertEqual(s["min"], 1.0)
        self.assertEqual(s["max"], 5.0)
        self.assertEqual(s["range"], 4.0)
        self.assertAlmostEqual(s["stdev"], 1.5811388300841898)

    def test_single_value(self) -> None:
        s = summarize([4.0])
        self.assertEqual(s["mean"], 4.0)
        self.assertEqual(s["median"], 4.0)
        self.assertEqual(s["min"], 4.0)
        self.assertEqual(s["max"], 4.0)
        self.assertEqual(s["range"], 0.0)
        self.assertEqual(s["stdev"], 0.0)

    def test_empty_list(self) -> None:
        s = summarize([])
        self.assertEqual(s["mean"], 0.0)
        self.assertEqual(s["median"], 0.0)
        self.assertEqual(s["min"], 0.0)
        self.assertEqual(s["max"], 0.0)
        self.assertEqual(s["range"], 0.0)
        self.assertEqual(s["stdev"], 0.0)

    def test_two_values(self) -> None:
        s = summarize([3.0, 7.0])
        self.assertEqual(s["mean"], 5.0)
        self.assertEqual(s["median"], 5.0)
        self.assertEqual(s["min"], 3.0)
        self.assertEqual(s["max"], 7.0)
        self.assertEqual(s["range"], 4.0)
        self.assertAlmostEqual(s["stdev"], 2.8284271247461903)


class TestValidScoreCount(unittest.TestCase):
    """valid_score_count — counting non-None values."""

    def test_all_none(self) -> None:
        scores: dict[str, int | None] = {
            "a": None, "b": None, "c": None, "d": None, "e": None,
        }
        self.assertEqual(valid_score_count(scores), 0)

    def test_some_none(self) -> None:
        scores: dict[str, int | None] = {
            "a": 5, "b": None, "c": 4, "d": None, "e": 3,
        }
        self.assertEqual(valid_score_count(scores), 3)

    def test_all_valid(self) -> None:
        scores: dict[str, int | None] = {
            "a": 5, "b": 4, "c": 3, "d": 4, "e": 5,
        }
        self.assertEqual(valid_score_count(scores), 5)

    def test_empty_dict(self) -> None:
        self.assertEqual(valid_score_count({}), 0)


class TestCollectDimValues(unittest.TestCase):
    """collect_dim_values — extracting scores from result dicts."""

    def test_all_valid(self) -> None:
        result: dict[str, Any] = {
            "回答相关性": {"分数": 5, "理由": "good"},
            "引用准确性": {"分数": 4, "理由": "ok"},
            "范围完整性": {"分数": 3, "理由": "partial"},
            "指令遵循": {"分数": 5, "理由": "yes"},
            "阅读感受与自然度": {"分数": 4, "理由": "fine"},
            "综合评价": "summary",
        }
        values = collect_dim_values(result, "zh")
        self.assertEqual(values, [5, 4, 3, 5, 4])

    def test_missing_score_defaults_to_zero(self) -> None:
        result: dict[str, Any] = {
            "回答相关性": {"分数": 5, "理由": "good"},
            "引用准确性": {},  # missing score
            "范围完整性": {"分数": 3, "理由": "partial"},
            "指令遵循": {"分数": 5, "理由": "yes"},
            "阅读感受与自然度": {"分数": 4, "理由": "fine"},
            "综合评价": "summary",
        }
        values = collect_dim_values(result, "zh")
        self.assertEqual(values, [5, 0, 3, 5, 4])

    def test_missing_dimension_defaults_to_zero(self) -> None:
        result: dict[str, Any] = {
            "回答相关性": {"分数": 5, "理由": "good"},
            "范围完整性": {"分数": 3, "理由": "partial"},
            "指令遵循": {"分数": 5, "理由": "yes"},
            "阅读感受与自然度": {"分数": 4, "理由": "fine"},
            "综合评价": "summary",
        }
        values = collect_dim_values(result, "zh")
        self.assertEqual(values, [5, 0, 3, 5, 4])

    def test_en_scores(self) -> None:
        result: dict[str, Any] = {
            "Response Relevance": {"score": 5, "reason": "good"},
            "Citation Accuracy": {"score": 4, "reason": "ok"},
            "Scope Completeness": {"score": 3, "reason": "partial"},
            "Instruction Following": {"score": 5, "reason": "yes"},
            "Readability and Naturalness": {"score": 4, "reason": "fine"},
            "Overall Summary": "summary",
        }
        values = collect_dim_values(result, "en")
        self.assertEqual(values, [5, 4, 3, 5, 4])

    def test_unknown_language_falls_back_to_en(self) -> None:
        result: dict[str, Any] = {
            "Response Relevance": {"score": 5, "reason": "yes"},
            "Citation Accuracy": {"score": 4, "reason": "ok"},
            "Scope Completeness": {"score": 3, "reason": "partial"},
            "Instruction Following": {"score": 5, "reason": "yes"},
            "Readability and Naturalness": {"score": 4, "reason": "fine"},
            "Overall Summary": "summary",
        }
        values = collect_dim_values(result, "ar")
        self.assertEqual(values, [5, 4, 3, 5, 4])


class TestScoreHeaders(unittest.TestCase):
    """score_headers — verify tuple format matches ``result_merger.LANGUAGE_CONFIG``."""

    def test_zh(self) -> None:
        headers = score_headers("zh")
        expected = [
            ("回答相关性-分数", "回答相关性-理由"),
            ("引用准确性-分数", "引用准确性-理由"),
            ("范围完整性-分数", "范围完整性-理由"),
            ("指令遵循-分数", "指令遵循-理由"),
            ("阅读感受与自然度-分数", "阅读感受与自然度-理由"),
        ]
        self.assertEqual(headers, expected)

    def test_en(self) -> None:
        headers = score_headers("en")
        expected = [
            ("Response Relevance-Score", "Response Relevance-Reason"),
            ("Citation Accuracy-Score", "Citation Accuracy-Reason"),
            ("Scope Completeness-Score", "Scope Completeness-Reason"),
            ("Instruction Following-Score", "Instruction Following-Reason"),
            ("Readability and Naturalness-Score", "Readability and Naturalness-Reason"),
        ]
        self.assertEqual(headers, expected)

    def test_unknown_language_falls_back_to_en_headers(self) -> None:
        headers = score_headers("fr")
        expected = score_headers("en")
        self.assertEqual(headers, expected)

    def test_unknown_language_falls_back_to_en(self) -> None:
        headers = score_headers("de")
        expected = score_headers("en")
        self.assertEqual(headers, expected)


class TestDrift(unittest.TestCase):
    """Consistency tests between ``LANGUAGE_META`` and the prompt templates.

    The evaluator prompt's JSON output keys must match ``LANGUAGE_META``: a
    drifted key would make every parsed response fail validation. These tests
    assert the templates embed exactly the keys metadata declares.
    """

    # ── prompt-template consistency ─────────────────────────────────────

    def _load_prompt(self, lang: str) -> str:
        if lang == "zh":
            from zh_cn import EVAL_PROMPT
        else:
            from en_us import EVAL_PROMPT
        return EVAL_PROMPT

    def _check_prompt_keys(self, lang: str) -> None:
        meta = LANGUAGE_META.get(lang, LANGUAGE_META["en"])
        prompt = self._load_prompt(lang)
        for dim in meta.dims:
            self.assertIn(dim, prompt, f"{lang}: dimension name {dim!r} missing from prompt")
        self.assertIn(meta.summary_key, prompt, f"{lang}: summary key missing from prompt")
        self.assertIn(meta.score_key, prompt, f"{lang}: score key missing from prompt")
        self.assertIn(meta.reason_key, prompt, f"{lang}: reason key missing from prompt")

    def test_prompt_keys_zh(self) -> None:
        self._check_prompt_keys("zh")

    def test_prompt_keys_en(self) -> None:
        self._check_prompt_keys("en")

    def test_prompt_keys_fallback_language(self) -> None:
        self._check_prompt_keys("ar")

    # ── score_headers anchors (historical output contract) ──────────────

    def _check_merger_headers(self, lang: str) -> None:
        expected_headers = {
            "zh": [
                ("回答相关性-分数", "回答相关性-理由"),
                ("引用准确性-分数", "引用准确性-理由"),
                ("范围完整性-分数", "范围完整性-理由"),
                ("指令遵循-分数", "指令遵循-理由"),
                ("阅读感受与自然度-分数", "阅读感受与自然度-理由"),
            ],
            "en": [
                ("Response Relevance-Score", "Response Relevance-Reason"),
                ("Citation Accuracy-Score", "Citation Accuracy-Reason"),
                ("Scope Completeness-Score", "Scope Completeness-Reason"),
                ("Instruction Following-Score", "Instruction Following-Reason"),
                ("Readability and Naturalness-Score", "Readability and Naturalness-Reason"),
            ],
        }
        self.assertEqual(
            score_headers(lang),
            expected_headers[lang],
            f"{lang}: score_headers() != expected",
        )

    def test_drift_zh_merger(self) -> None:
        self._check_merger_headers("zh")

    def test_drift_en_merger(self) -> None:
        self._check_merger_headers("en")

    def test_drift_fallback_language_uses_en(self) -> None:
        self.assertEqual(score_headers("fr"), score_headers("en"))
        self.assertEqual(score_headers("id"), score_headers("en"))
        self.assertEqual(score_headers("ru"), score_headers("en"))


if __name__ == "__main__":
    unittest.main()