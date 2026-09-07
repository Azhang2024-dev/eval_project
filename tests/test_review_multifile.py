"""Tests for multi-file review: doc discovery, budgeted multi-doc reading, placeholders."""

import os
import tempfile
import unittest

import pandas as pd

from scripts.review_module import ReviewModule


def _mk_module(doc_dir):
    df = pd.DataFrame([{"query": "q", "answer": "a", "filename": "one.md"}])
    return ReviewModule(
        eval_df=df,
        results=[],
        doc_dirs=[doc_dir],
        review_models=["m"],
        language="en",
    )


class FindDocsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        for name, body in (("one.md", "doc one"), ("two.txt", "doc two"), ("three.pdf", "binary")):
            with open(os.path.join(self.tmp.name, name), "w", encoding="utf-8") as f:
                f.write(body)
        self.rm = _mk_module(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_multiple_files_found_in_order(self):
        paths = self.rm.find_original_docs(["one.md", "two.txt", "nope.md"])
        self.assertEqual([os.path.basename(p) for p in paths], ["one.md", "two.txt"])

    def test_missing_files_skipped(self):
        paths = self.rm.find_original_docs(["nope.md"])
        self.assertEqual(paths, [])

    def test_budget_keeps_all_file_headers(self):
        text = self.rm._read_docs_with_budget(
            [os.path.join(self.tmp.name, "one.md"), os.path.join(self.tmp.name, "two.txt")],
            budget=30,
        )
        self.assertIn("one.md", text)
        self.assertIn("two.txt", text)

    def test_binary_file_placeholder(self):
        text = self.rm._read_docs_with_budget([os.path.join(self.tmp.name, "three.pdf")], budget=1000)
        self.assertIn("[Binary file: three.pdf]", text)

    def test_full_budget_includes_content(self):
        text = self.rm._read_docs_with_budget(
            [os.path.join(self.tmp.name, "one.md"), os.path.join(self.tmp.name, "two.txt")],
            budget=128000,
        )
        self.assertIn("doc one", text)
        self.assertIn("doc two", text)

    def test_review_single_zero_docs_returns_no_original_doc(self):
        from unittest.mock import patch

        item = {
            "filename": "nope.md",
            "filenames": ["nope.md"],
            "query": "q",
            "answer": "a",
            "result": {},
        }
        with patch("scripts.review_module._call_api") as api_mock:
            result = self.rm.review_single(item, "m")
        self.assertEqual(result["error"], "no_original_doc")
        self.assertIsNone(result["parsed"])
        api_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
