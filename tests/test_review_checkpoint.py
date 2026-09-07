"""Tests for review enhancements: checkpoint resume, suggested-score columns, binary docs."""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from scripts.review_module import ReviewModule


def _eval_df():
    return pd.DataFrame(
        [
            {"query": "q0", "answer": "a0", "filename": "one.md"},
            {"query": "q1", "answer": "a1", "filename": "one.md"},
        ]
    )


def _results():
    def low(index):
        return {
            "index": index,
            "status": "success",
            "result": {
                "Response Relevance": {"score": 3, "reason": "low"},
                "Citation Accuracy": {"score": 5, "reason": "ok"},
                "Scope Completeness": {"score": 5, "reason": "ok"},
                "Instruction Following": {"score": 5, "reason": "ok"},
                "Readability and Naturalness": {"score": 5, "reason": "ok"},
            },
        }

    return [low(0), low(1)]


def _review_parsed(verdict="Reasonable"):
    return {
        "Original Found": "Yes",
        "Response Relevance": {"original_score": 3, "suggested_score": 3, "reason": "r"},
        "Citation Accuracy": {"original_score": 5, "suggested_score": 5, "reason": "r"},
        "Scope Completeness": {"original_score": 5, "suggested_score": 5, "reason": "r"},
        "Instruction Following": {"original_score": 5, "suggested_score": 5, "reason": "r"},
        "Readability and Naturalness": {"original_score": 5, "suggested_score": 5, "reason": "r"},
        "Summary": {
            "score_reasonable": verdict,
            "main_issues": "Citation Accuracy — source says X, answer says Y [Critical Error]",
            "note": "Critical Error",
        },
    }


def _mk_module(tmp_dir, doc_dir, review_models=("model-one", "model-two")):
    return ReviewModule(
        eval_df=_eval_df(),
        results=_results(),
        doc_dirs=[doc_dir],
        review_models=list(review_models),
        language="en",
        checkpoint_dir=tmp_dir,
        input_fingerprint="fp",
        eval_model="eval-model",
    )


class ReviewCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.doc_dir = os.path.join(self.tmp.name, "docs")
        os.makedirs(self.doc_dir)
        with open(os.path.join(self.doc_dir, "one.md"), "w", encoding="utf-8") as f:
            f.write("doc one")

    def tearDown(self):
        self.tmp.cleanup()

    def test_resume_reuses_cached_reviews_without_api_calls(self):
        module = _mk_module(self.tmp.name, self.doc_dir)
        with patch.object(
            module,
            "review_single",
            side_effect=lambda item, model: {
                "model": model, "error": None, "parsed": _review_parsed(),
            },
        ) as review_mock:
            df = module.review_all(max_workers=2, threshold=5)
        self.assertEqual(len(df), 2)
        self.assertEqual(review_mock.call_count, 4)

        checkpoint_files = [
            name for name in os.listdir(self.tmp.name)
            if name.startswith("_review_checkpoint_")
        ]
        self.assertEqual(len(checkpoint_files), 1)

        module2 = _mk_module(self.tmp.name, self.doc_dir)
        with patch.object(module2, "review_single", side_effect=AssertionError("must not call")):
            df2 = module2.review_all(max_workers=2, threshold=5)
        self.assertEqual(len(df2), 2)
        pd.testing.assert_frame_equal(df, df2)

    def test_stale_checkpoint_with_changed_doc_dir_is_ignored(self):
        module = _mk_module(self.tmp.name, self.doc_dir)
        with patch.object(
            module,
            "review_single",
            side_effect=lambda item, model: {
                "model": model, "error": None, "parsed": _review_parsed(),
            },
        ) as review_mock:
            module.review_all(max_workers=2, threshold=5)
        self.assertEqual(review_mock.call_count, 4)

        other_doc_dir = os.path.join(self.tmp.name, "other-docs")
        os.makedirs(other_doc_dir)
        with open(os.path.join(other_doc_dir, "one.md"), "w", encoding="utf-8") as f:
            f.write("different content")

        module2 = _mk_module(self.tmp.name, other_doc_dir)
        with patch.object(
            module2,
            "review_single",
            side_effect=lambda item, model: {
                "model": model, "error": None, "parsed": _review_parsed(),
            },
        ) as review_mock2:
            module2.review_all(max_workers=2, threshold=5)
        self.assertEqual(review_mock2.call_count, 4)

    def test_checkpoint_meta_contains_identity_fields(self):
        module = _mk_module(self.tmp.name, self.doc_dir)
        module.review_all = lambda *a, **k: pd.DataFrame()  # avoid review work
        meta = module._checkpoint_meta()
        self.assertEqual(meta["eval_model"], "eval-model")
        self.assertEqual(meta["language"], "en")
        self.assertEqual(meta["review_models"], ["model-one", "model-two"])
        self.assertEqual(meta["input_fingerprint"], "fp")
        self.assertTrue(meta["doc_dirs_fingerprint"])


class ReviewSheetColumnsTest(unittest.TestCase):
    def test_suggested_scores_and_main_issues_are_exported(self):
        with tempfile.TemporaryDirectory() as tmp:
            doc_dir = os.path.join(tmp, "docs")
            os.makedirs(doc_dir)
            with open(os.path.join(doc_dir, "one.md"), "w", encoding="utf-8") as f:
                f.write("doc one")
            module = _mk_module(tmp, doc_dir)
            with patch.object(
                module,
                "review_single",
                side_effect=lambda item, model: {
                    "model": model, "error": None, "parsed": _review_parsed(),
                },
            ):
                df = module.review_all(max_workers=2, threshold=5)

            self.assertIn("model-one-建议分", df.columns)
            self.assertIn("model-two-主要问题", df.columns)
            self.assertEqual(df.iloc[0]["model-one-建议分"], "RR 3 | CA 5 | SC 5 | IF 5 | RN 5")
            self.assertIn("Citation Accuracy", df.iloc[0]["model-one-主要问题"])
            self.assertEqual(df.iloc[0]["最终结论"], "reasonable")

    def test_error_result_yields_empty_suggested_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            doc_dir = os.path.join(tmp, "docs")
            os.makedirs(doc_dir)
            module = _mk_module(tmp, doc_dir)
            with patch.object(
                module,
                "review_single",
                side_effect=lambda item, model: {"model": model, "error": "boom", "parsed": None},
            ):
                df = module.review_all(max_workers=2, threshold=5)
            self.assertEqual(df.iloc[0]["model-one-建议分"], "")
            self.assertEqual(df.iloc[0]["最终结论"], "manual_review")


class BinaryDocExtractionTest(unittest.TestCase):
    def test_pdf_text_extracted(self):
        import fitz  # pymupdf

        document = fitz.open()
        page = document.new_page()
        page.insert_text((72, 72), "Hello PDF text")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "doc.pdf")
            document.save(path)
            document.close()
            module = _mk_module(tmp, tmp)
            text = module._read_doc_content(path)
        self.assertIn("Hello PDF text", text)

    def test_blank_pdf_falls_back_to_placeholder(self):
        import fitz

        document = fitz.open()
        document.new_page()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "blank.pdf")
            document.save(path)
            document.close()
            module = _mk_module(tmp, tmp)
            text = module._read_doc_content(path)
        self.assertIn("[Binary file: blank.pdf (no text layer)]", text)

    def test_docx_text_extracted(self):
        import zipfile

        document_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            "<w:body>"
            "<w:p><w:r><w:t>Hello docx</w:t></w:r></w:p>"
            "<w:p><w:r><w:t>Second paragraph</w:t></w:r></w:p>"
            "</w:body></w:document>"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "doc.docx")
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("word/document.xml", document_xml)
            module = _mk_module(tmp, tmp)
            text = module._read_doc_content(path)
        self.assertEqual(text, "Hello docx\nSecond paragraph")

    def test_corrupt_docx_falls_back_to_placeholder(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "broken.docx")
            with open(path, "wb") as f:
                f.write(b"not a zip")
            module = _mk_module(tmp, tmp)
            text = module._read_doc_content(path)
        self.assertIn("[Binary file: broken.docx]", text)


class ReviewArbitrateTest(unittest.TestCase):
    def test_missing_verdict_is_manual_review_not_adjustment(self):
        """Regression: a parseable-but-malformed review (no verdict) must be
        manual_review, not an agreed needs_adjustment with empty notes."""
        with tempfile.TemporaryDirectory() as tmp:
            doc_dir = os.path.join(tmp, "docs")
            os.makedirs(doc_dir)
            module = _mk_module(tmp, doc_dir)
            result = module.arbitrate({
                "m1": {"model": "m1", "error": None, "parsed": {"Summary": {}}},
            })
            self.assertEqual(result["conclusion"], "manual_review")
            self.assertEqual(result["consistency"], "malformed")


if __name__ == "__main__":
    unittest.main()
