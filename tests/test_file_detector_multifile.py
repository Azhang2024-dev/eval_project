"""Tests for multi-file grouping: chunk-level filename attribution and group file list."""

import os
import tempfile
import unittest

import pandas as pd

from scripts.file_detector import FileDetector, count_skipped_query_rows, group_and_merge

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "multi_file_grouped.csv")


class MultiFileGroupingTest(unittest.TestCase):
    def setUp(self):
        self.detector = FileDetector(FIXTURE)
        self.df = self.detector.load()
        self.columns = self.detector.identify_columns()
        self.layout = self.detector.identify_layout()

    def test_layout_is_grouped(self):
        self.assertEqual(self.layout, "grouped")

    def test_chunks_carry_per_row_filename(self):
        items = group_and_merge(self.df, self.columns, self.layout)
        self.assertEqual(len(items), 1)
        by_text = {c["text"]: c["filename"] for c in items[0]["chunks"]}
        self.assertEqual(by_text["chunk text from a"], "a.pdf")
        self.assertEqual(by_text["chunk text from b"], "b.pdf")
        self.assertEqual(by_text["chunk text from c"], "c.pdf")

    def test_chunks_sorted_by_score_desc(self):
        items = group_and_merge(self.df, self.columns, self.layout)
        self.assertEqual(
            [c["text"] for c in items[0]["chunks"]],
            ["chunk text from a", "chunk text from b", "chunk text from c"],
        )

    def test_group_filenames_dedup_in_row_order(self):
        items = group_and_merge(self.df, self.columns, self.layout)
        self.assertEqual(items[0]["filenames"], ["a.pdf", "b.pdf", "c.pdf"])
        self.assertEqual(items[0]["filename"], "a.pdf")  # backward compat

    def test_missing_filename_falls_back_to_unknown(self):
        df = self.df.copy()
        df[self.columns["filename"]] = None
        items = group_and_merge(df, self.columns, self.layout)
        self.assertEqual(items[0]["filenames"], ["Unknown"])
        self.assertTrue(all(c["filename"] == "" for c in items[0]["chunks"]))


_COLUMNS = {
    "query": "query",
    "filename": "filename",
    "chunks": "chunks",
    "score": "score",
    "answer": "answer",
}


class RowByRowSemanticsTest(unittest.TestCase):
    """Layout A (row_by_row): each row is one independent evaluation entry."""

    def test_duplicate_queries_are_evaluated_independently(self):
        """Regression: identical query text must NOT merge rows in layout A."""
        df = pd.DataFrame([
            {"query": "same question", "filename": "a.pdf", "chunks": "chunk1", "score": 0.9, "answer": "answer 1"},
            {"query": "same question", "filename": "a.pdf", "chunks": "chunk2", "score": 0.8, "answer": "answer 2"},
        ])
        items = group_and_merge(df, _COLUMNS, "row_by_row")
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["index"], 0)
        self.assertEqual(items[0]["answer"], "answer 1")
        self.assertEqual(items[1]["index"], 1)
        self.assertEqual(items[1]["answer"], "answer 2")
        self.assertEqual(items[1]["chunks"][0]["text"], "chunk2")

    def test_empty_query_rows_are_skipped(self):
        df = pd.DataFrame([
            {"query": "q0", "filename": "a.pdf", "chunks": "c0", "score": 0.9, "answer": "a0"},
            {"query": None, "filename": "a.pdf", "chunks": "c1", "score": 0.8, "answer": "a1"},
            {"query": "q2", "filename": "a.pdf", "chunks": "c2", "score": 0.7, "answer": "a2"},
        ])
        items = group_and_merge(df, _COLUMNS, "row_by_row")
        self.assertEqual([item["index"] for item in items], [0, 2])
        self.assertEqual(count_skipped_query_rows(df, _COLUMNS, "row_by_row"), 1)

    def test_skipped_count_matches_merged_items(self):
        df = pd.DataFrame([
            {"query": None, "filename": "a.pdf", "chunks": "c0", "score": 0.9, "answer": "a0"},
            {"query": "q1", "filename": "a.pdf", "chunks": "c1", "score": 0.8, "answer": "a1"},
        ])
        items = group_and_merge(df, _COLUMNS, "row_by_row")
        self.assertEqual(len(items), 1)
        self.assertEqual(count_skipped_query_rows(df, _COLUMNS, "row_by_row"), 1)


class GroupedAnswerTest(unittest.TestCase):
    """Layout B (grouped): first non-empty answer wins, no cross-group leaks."""

    def test_answer_on_later_row_is_found(self):
        """Regression: group whose first row has no answer must still get the answer."""
        df = pd.DataFrame([
            {"query": "q0", "filename": "a.pdf", "chunks": "c0", "score": 0.9, "answer": None},
            {"query": None, "filename": "a.pdf", "chunks": "c1", "score": 0.8, "answer": "answer here"},
        ])
        items = group_and_merge(df, _COLUMNS, "grouped")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["answer"], "answer here")

    def test_answer_does_not_leak_across_groups(self):
        """Regression: an empty answer in a later group must not inherit the previous group's answer."""
        df = pd.DataFrame([
            {"query": "q1", "filename": "a.pdf", "chunks": "c0", "score": 0.9, "answer": "answer one"},
            {"query": "q2", "filename": "a.pdf", "chunks": "c1", "score": 0.8, "answer": None},
        ])
        items = group_and_merge(df, _COLUMNS, "grouped")
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["answer"], "answer one")
        self.assertEqual(items[1]["answer"], "")

    def test_grouped_skipped_count_counts_leading_empty_rows(self):
        df = pd.DataFrame([
            {"query": None, "filename": "a.pdf", "chunks": "c0", "score": 0.9, "answer": None},
            {"query": "q1", "filename": "a.pdf", "chunks": "c1", "score": 0.8, "answer": "a1"},
        ])
        items = group_and_merge(df, _COLUMNS, "grouped")
        self.assertEqual(len(items), 1)
        self.assertEqual(count_skipped_query_rows(df, _COLUMNS, "grouped"), 1)


class CsvEncodingTest(unittest.TestCase):
    """CSV must tolerate legacy Chinese encodings (GB18030 fallback)."""

    def test_utf8_csv_loads_and_records_encoding(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "utf8.csv")
            pd.DataFrame([{"query": "问题一", "chunks": "内容"}]).to_csv(path, index=False, encoding="utf-8")
            detector = FileDetector(path)
            df = detector.load()
            self.assertEqual(detector.csv_encoding, "utf-8-sig")
            self.assertEqual(df.iloc[0]["query"], "问题一")

    def test_gbk_csv_falls_back_to_gb18030(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "gbk.csv")
            payload = "query,chunks\n问题一,内容一\n".encode("gb18030")
            with open(path, "wb") as f:
                f.write(payload)
            detector = FileDetector(path)
            df = detector.load()
            self.assertEqual(detector.csv_encoding, "gb18030")
            self.assertEqual(df.iloc[0]["query"], "问题一")


class ColumnMappingTest(unittest.TestCase):
    """One column must never map to two fields; specific patterns win."""

    def _detect(self, columns):
        df = pd.DataFrame([{name: None for name in columns}])
        detector = FileDetector.__new__(FileDetector)
        detector.df = df
        return detector.identify_columns()

    def test_chunk_score_column_maps_to_score_only(self):
        """Regression: 'chunk_score' (a retrieval-score export) must not map
        to both 'chunks' and 'score', which crashed chunk parsing."""
        cols = self._detect(["query", "chunk", "chunk_score", "answer"])
        self.assertEqual(cols["chunks"], "chunk")
        self.assertEqual(cols["score"], "chunk_score")
        self.assertEqual(cols["query"], "query")
        self.assertEqual(cols["answer"], "answer")

    def test_chunk_score_pair_parses_without_crash(self):
        df = pd.DataFrame([
            {"query": "q", "chunk": "chunk text", "chunk_score": 0.9, "answer": "a"},
        ])
        detector = FileDetector.__new__(FileDetector)
        detector.df = df
        cols = detector.identify_columns()
        items = group_and_merge(df, cols, "row_by_row")
        self.assertEqual(items[0]["chunks"][0]["text"], "chunk text")
        self.assertEqual(items[0]["chunks"][0]["score"], 0.9)

    def test_plain_headers_keep_mapping(self):
        cols = self._detect(["query", "filename", "搜索结果", "score", "大模型问答结果"])
        self.assertEqual(cols["query"], "query")
        self.assertEqual(cols["filename"], "filename")
        self.assertEqual(cols["chunks"], "搜索结果")
        self.assertEqual(cols["score"], "score")
        self.assertEqual(cols["answer"], "大模型问答结果")


if __name__ == "__main__":
    unittest.main()
