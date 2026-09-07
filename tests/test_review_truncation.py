from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from scripts.review_module import ReviewModule


class ReviewTruncationTest(unittest.TestCase):
    def test_review_single_caps_original_text_at_128000_chars(self) -> None:
        module = ReviewModule(
            eval_df=pd.DataFrame(),
            results=[],
            doc_dirs=[],
            review_models=["azure/gpt-5.4"],
            language="en",
        )
        module.prompt_template = "ORIGINAL_START{original_text}ORIGINAL_END"

        item = {
            "filename": "doc.md",
            "query": "What happened?",
            "answer": "An answer.",
            "result": {},
        }
        long_text = "§" * 200000

        with (
            patch.object(module, "_find_original_doc_single", return_value="C:/fake/doc.md"),
            patch.object(module, "_read_doc_content", return_value=long_text),
            patch("scripts.review_module._call_api", return_value={"Summary": {}}) as api_mock,
        ):
            result = module.review_single(item, "azure/gpt-5.4")

        self.assertIsNone(result["error"])
        prompt = api_mock.call_args.args[0]
        original_text = prompt.removeprefix("ORIGINAL_START").removesuffix("ORIGINAL_END")

        prefix = "=== Original file: doc.md ===\n"
        self.assertTrue(original_text.startswith(prefix))
        body = original_text[len(prefix):]
        self.assertEqual(body, long_text[:128000])
        self.assertEqual(len(body), 128000)


if __name__ == "__main__":
    unittest.main()
