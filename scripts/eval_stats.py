"""Pure stdlib evaluation statistics module — single source of truth for eval stats.

This module is the canonical home for all evaluation statistics logic. It MUST
NOT import anything from ``scripts.`` or any external package — only stdlib.

Constants, data structures, and functions here are consumed by
``excel_writer.py``, ``report_generator.py``, and ``result_merger.py``.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any

# ── Constants ────────────────────────────────────────────────────────────────

WEIGHTS: tuple[float, ...] = (0.3, 0.3, 0.15, 0.15, 0.1)
"""Weight vector for the five evaluation dimensions."""

BADCASE_THRESHOLD: float = 3.0
"""Weighted scores strictly below this value are considered badcases."""

# ── Language metadata ────────────────────────────────────────────────────────


@dataclass
class LanguageMeta:
    """Language-specific dimension keys, score keys, and reason keys.

    This is the single source of truth for the three supported languages:
    dimension names, score/reason key names, sheet names, labels, and the
    empty-answer texts. All other modules import from here instead of
    redefining their own copies.
    """

    dims: list[str]        # 5 dimensions (no summary)
    score_key: str          # "分数" / "score" / "note"
    reason_key: str         # "理由" / "reason" / "raison"
    summary_key: str        # "综合评价" / "Overall Summary" / "Synthèse"
    score_label: str        # "分数" / "Score" / "Note"
    reason_label: str       # "理由" / "Reason" / "Raison"
    eval_sheet_name: str    # "评测结果" / "Evaluation"
    review_sheet_name: str  # "低分复核" / "Review"
    weighted_label: str     # "加权均分" / "Weighted Average" / "Moyenne pondérée"
    empty_reason: str       # "回答为空" / "Response is empty" / "Réponse vide"
    empty_summary: str      # "回答为空，无法评测" / "Response is empty, unable to evaluate" / "Réponse vide, évaluation impossible"


LANGUAGE_META: dict[str, LanguageMeta] = {
    "zh": LanguageMeta(
        dims=["回答相关性", "引用准确性", "范围完整性", "指令遵循", "阅读感受与自然度"],
        score_key="分数",
        reason_key="理由",
        summary_key="综合评价",
        score_label="分数",
        reason_label="理由",
        eval_sheet_name="评测结果",
        review_sheet_name="低分复核",
        weighted_label="加权均分",
        empty_reason="回答为空",
        empty_summary="回答为空，无法评测",
    ),
    "en": LanguageMeta(
        dims=[
            "Response Relevance",
            "Citation Accuracy",
            "Scope Completeness",
            "Instruction Following",
            "Readability and Naturalness",
        ],
        score_key="score",
        reason_key="reason",
        summary_key="Overall Summary",
        score_label="Score",
        reason_label="Reason",
        eval_sheet_name="Evaluation",
        review_sheet_name="Review",
        weighted_label="Weighted Average",
        empty_reason="Response is empty",
        empty_summary="Response is empty, unable to evaluate",
    ),
}
"""Mapping from language code to dimension/reason/score key metadata.

Evaluation standards are intentionally limited to ``zh`` and ``en``: all other
data languages (ar/fr/id/ru/...) fall back to the English standard via
``_meta()``. Language detection still reports the true data language code for
display, but evaluation prompts, JSON schemas, and output headers use the
``zh`` or ``en`` standard exclusively.
"""


def _meta(language: str) -> LanguageMeta:
    """Look up LanguageMeta, falling back to English."""
    return LANGUAGE_META.get(language, LANGUAGE_META["en"])


# ── Public helpers ───────────────────────────────────────────────────────────


def score_headers(language: str) -> list[tuple[str, str]]:
    """Return the 5 ``(score_header, reason_header)`` tuples for *language*.

    Each header is formed as ``\"{dim}-{label}\"`` where *label* is the
    score/reason label from ``LanguageMeta``.
    """
    meta = _meta(language)
    return [
        (f"{dim}-{meta.score_label}", f"{dim}-{meta.reason_label}")
        for dim in meta.dims
    ]


def parse_score(value: Any) -> int | None:
    """Parse a raw score value into a valid 1-5 integer or ``None``.

    Semantics match ``result_merger._safe_score``:

    * ``bool`` → ``None`` (``True``/``False`` rejected)
    * ``None`` / empty / missing → ``None``
    * ``float`` → parse, check ``is_integer()``, must be in 1..5
    * ``str`` → try ``float()``, then same check
    * NaN → ``None``
    * Non-integer or out of range → ``None``
    """
    if isinstance(value, bool):
        return None
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    if not numeric.is_integer():
        return None
    integer = int(numeric)
    return integer if 1 <= integer <= 5 else None


def collect_dim_values(result: dict[str, Any], language: str) -> list[int]:
    """Collect the 5 dimension scores from an evaluation *result* dict.

    Invalid or missing scores are replaced with ``0``, matching the behaviour
    in ``excel_writer.write_eval_sheet``.
    """
    meta = _meta(language)
    values: list[int] = []
    for dim in meta.dims:
        dim_data = result.get(dim, {})
        if not isinstance(dim_data, dict):
            dim_data = {}
        raw = dim_data.get(meta.score_key)
        try:
            values.append(int(raw))
        except (TypeError, ValueError):
            values.append(0)
    return values


def weighted_from_values(values: list[float]) -> float:
    """Compute the weighted sum from a list of 5 dimension values.

    ``sum(v * w for v, w in zip(values, WEIGHTS))`` — exact, not rounded.
    """
    return sum(v * w for v, w in zip(values, WEIGHTS))


def weighted_from_score_dict(
    scores: dict[str, int | None],
    language: str,
) -> float | None:
    """Compute the weighted average from a dimension-to-score dict.

    Returns ``None`` if any dimension is ``None``.
    ``round(sum(float(value) * weight …), 2)``
    """
    meta = _meta(language)
    values = [scores.get(dim) for dim in meta.dims]
    if any(value is None for value in values):
        return None
    return round(
        sum(float(value) * weight for value, weight in zip(values, WEIGHTS)),
        2,
    )


def summarize(values: list[float]) -> dict[str, float]:
    """Return summary statistics for *values*.

    Returns ``{\"mean\", \"median\", \"min\", \"max\", \"range\", \"stdev\"}``.
    For ``n ≤ 1``, ``stdev = 0`` (``statistics.stdev`` would raise).
    Values are exact (not rounded).
    """
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "range": 0.0, "stdev": 0.0}
    mean = statistics.mean(values)
    median = statistics.median(values)
    min_v = min(values)
    max_v = max(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "mean": mean,
        "median": median,
        "min": min_v,
        "max": max_v,
        "range": max_v - min_v,
        "stdev": stdev,
    }


def band_distribution(weighted_values: list[float]) -> dict[str, int]:
    """Bin *weighted_values* into score bands.

    Boundary rules (matching existing code):

    * ``< 2.0`` → ``1.0-1.9``
    * ``< 3.0`` → ``2.0-2.9``
    * ``< 4.0`` → ``3.0-3.9``
    * ``< 4.5`` → ``4.0-4.5``
    * else → ``4.5-5.0``
    """
    bands: dict[str, int] = {
        "1.0-1.9": 0,
        "2.0-2.9": 0,
        "3.0-3.9": 0,
        "4.0-4.5": 0,
        "4.5-5.0": 0,
    }
    for w in weighted_values:
        if w < 2.0:
            bands["1.0-1.9"] += 1
        elif w < 3.0:
            bands["2.0-2.9"] += 1
        elif w < 4.0:
            bands["3.0-3.9"] += 1
        elif w < 4.5:
            bands["4.0-4.5"] += 1
        else:
            bands["4.5-5.0"] += 1
    return bands


SCORED_STATUSES: tuple[str, ...] = ("success", "empty")
"""Result statuses that carry dimension scores."""


def collect_weighted_scores(
    results: list[dict[str, Any]],
    language: str,
) -> list[tuple[int, float]]:
    """Return ``(index, weighted_score)`` pairs for scored rows.

    Pairs are aligned by the explicit ``index`` field on each result item, so
    ``failed`` rows (which carry no scores) neither shift nor mask the scores
    of other rows — the position-based ``zip`` that caused badcase
    misalignment is gone. Only ``success`` and ``empty`` rows are included,
    matching the previous stats loops. Missing/invalid dimension scores are
    treated as ``0`` (same semantics as the legacy loops).
    """
    pairs: list[tuple[int, float]] = []
    for item in results:
        if item.get("status") not in SCORED_STATUSES:
            continue
        index = item.get("index")
        if index is None:
            continue
        result = item.get("result") or {}
        if not isinstance(result, dict):
            result = {}
        values = collect_dim_values(result, language)
        pairs.append((int(index), weighted_from_values(values)))
    return pairs


def badcase_entries(
    weighted_pairs: list[tuple[int, float]],
    threshold: float = BADCASE_THRESHOLD,
) -> list[tuple[int, float]]:
    """Return ``(index, weighted_score)`` for badcase entries.

    *weighted_pairs* must come from ``collect_weighted_scores`` so that each
    score stays aligned with the row that produced it. Rows without scores
    (``failed``) are simply absent and cannot shift or mask other rows.
    """
    return [
        (index, score)
        for index, score in weighted_pairs
        if score < threshold
    ]


def valid_score_count(scores: dict[str, int | None]) -> int:
    """Count the number of non-``None`` values in *scores*."""
    return sum(value is not None for value in scores.values())