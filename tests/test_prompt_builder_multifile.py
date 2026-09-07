"""Tests for multi-file prompt building: chunk file labels and multi-file header."""

import unittest

from scripts.prompt_builder import build_chunks_text, build_prompt


class ChunkLabelTest(unittest.TestCase):
    def test_chunk_header_includes_file(self):
        chunks = [{"text": "hello", "score": 0.9, "filename": "a.pdf"}]
        self.assertIn(
            "--- Chunk 1 (file: a.pdf, score: 0.9000) ---",
            build_chunks_text(chunks),
        )

    def test_chunk_header_missing_file_shows_unknown(self):
        chunks = [{"text": "hello", "score": 0.5, "filename": ""}]
        self.assertIn(
            "--- Chunk 1 (file: unknown, score: 0.5000) ---",
            build_chunks_text(chunks),
        )

    def test_legacy_chunk_without_filename_key(self):
        chunks = [{"text": "hello", "score": 0.5}]
        self.assertIn("file: unknown", build_chunks_text(chunks))


class PromptMultiFileTest(unittest.TestCase):
    CHUNKS = [{"text": "t", "score": 0.9, "filename": "a.pdf"}]

    def test_zh_prompt_lists_all_files(self):
        p = build_prompt("q", ["a.pdf", "b.pdf"], self.CHUNKS, "ans", "zh")
        self.assertIn("a.pdf", p)
        self.assertIn("b.pdf", p)
        self.assertNotIn("{files_text}", p)

    def test_zh_prompt_accepts_single_string(self):
        p = build_prompt("q", "a.pdf", self.CHUNKS, "ans", "zh")
        self.assertIn("a.pdf", p)
        self.assertNotIn("{files_text}", p)

    def test_empty_file_list_falls_back_to_unknown(self):
        p = build_prompt("q", [], self.CHUNKS, "ans", "zh")
        self.assertIn("Unknown", p)
        self.assertNotIn("{files_text}", p)

    def test_all_languages_render_with_file_labels(self):
        for lang in ("zh", "en", "fr", "ar", "id", "ru"):
            p = build_prompt("q", ["a.pdf"], self.CHUNKS, "ans", lang)
            self.assertIn("file: a.pdf", p, f"missing chunk file label for {lang}")
            self.assertNotIn("{files_text}", p, f"unsubstituted placeholder for {lang}")
            self.assertNotIn("{filename}", p, f"stale placeholder for {lang}")
            self.assertNotIn("{chunks_text}", p, f"unsubstituted placeholder for {lang}")


class ChunkBudgetTest(unittest.TestCase):
    CHUNKS = [
        {"text": "first chunk text", "score": 0.9, "filename": "a.pdf"},
        {"text": "second chunk text", "score": 0.8, "filename": "a.pdf"},
        {"text": "third chunk text", "score": 0.7, "filename": "a.pdf"},
    ]

    def test_no_budget_keeps_all_chunks(self):
        p = build_prompt("q", ["a.pdf"], self.CHUNKS, "ans", "en")
        self.assertIn("first chunk text", p)
        self.assertIn("third chunk text", p)

    def test_max_chunks_keeps_top_scored_only(self):
        p = build_prompt("q", ["a.pdf"], self.CHUNKS, "ans", "en", max_chunks=2)
        self.assertIn("first chunk text", p)
        self.assertIn("second chunk text", p)
        self.assertNotIn("third chunk text", p)

    def test_max_chunk_chars_truncates_with_marker(self):
        chunks = [{"text": "x" * 100, "score": 0.9, "filename": "a.pdf"}]
        p = build_prompt("q", ["a.pdf"], chunks, "ans", "en", max_chunk_chars=20)
        self.assertIn("x" * 20, p)
        self.assertNotIn("x" * 21, p)
        self.assertIn("[truncated: 100 chars]", p)

    def test_max_chunk_chars_zero_means_unlimited(self):
        chunks = [{"text": "y" * 100, "score": 0.9, "filename": "a.pdf"}]
        p = build_prompt("q", ["a.pdf"], chunks, "ans", "en", max_chunk_chars=0)
        self.assertIn("y" * 100, p)
        self.assertNotIn("[truncated", p)


class PromptLanguageRuleTest(unittest.TestCase):
    """Reason output language must be pinned: Chinese for zh, English for
    every other data language (en/fr/ar/id/ru). Without this, evaluators mix
    the source language with English for non-zh/en documents."""

    CHUNKS = [{"text": "t", "score": 0.9, "filename": "a.pdf"}]

    EN_MARKER = "must be written in **English**"
    ZH_MARKER = "一律使用**简体中文**"

    def test_non_zh_eval_prompts_enforce_english_reasons(self):
        for lang in ("en", "fr", "ar", "id", "ru"):
            p = build_prompt("q", ["a.pdf"], self.CHUNKS, "ans", lang)
            self.assertIn(self.EN_MARKER, p, f"{lang}: reason language rule missing")

    def test_zh_eval_prompt_enforces_chinese_reasons(self):
        p = build_prompt("q", ["a.pdf"], self.CHUNKS, "ans", "zh")
        self.assertIn(self.ZH_MARKER, p, "zh: reason language rule missing")

    def test_review_prompts_enforce_reason_language(self):
        from prompts.review_prompt import REVIEW_PROMPT_EN, REVIEW_PROMPT_ZH

        self.assertIn(self.EN_MARKER, REVIEW_PROMPT_EN, "review en: language rule missing")
        self.assertIn(self.ZH_MARKER, REVIEW_PROMPT_ZH, "review zh: language rule missing")


if __name__ == "__main__":
    unittest.main()
