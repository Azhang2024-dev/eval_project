from __future__ import annotations

import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts.eval_config import ConfigError, resolve_run_config
from scripts.evaluate import DEFAULT_MODEL_PROTOCOLS, main
from scripts.file_detector import FileDetector
from scripts.review_module import ReviewModule
from scripts.settings import APISettingsError, load_api_settings


class Args:
    config = None
    file = None
    models = None
    language = None
    output = None
    test = None
    workers = None
    checkpoint_dir = None
    thinking = None
    thinking_effort = None
    model_protocols = None
    model_response_modes = None
    review = None
    doc_dirs = None
    review_models = None
    threshold = None
    merge = None
    merge_output = None


class HeadlessCliTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.input_path = self.root / "input.xlsx"
        pd.DataFrame(
            [
                {
                    "query": "What is X?",
                    "filename": "doc.md",
                    "搜索结果": "Evidence",
                    "score": 0.9,
                    "大模型问答结果": "Answer",
                }
            ]
        ).to_excel(self.input_path, index=False)

    def tearDown(self):
        self.temp.cleanup()

    def test_environment_api_settings_are_loaded_and_required(self):
        with patch.dict(
            os.environ,
            {
                "KB_QA_API_KEY": "env-test-key",
                "KB_QA_API_BASE_URL": "https://env.example/v1",
            },
            clear=False,
        ):
            settings = load_api_settings()
        self.assertEqual(settings.api_key, "env-test-key")
        self.assertEqual(settings.base_url, "https://env.example/v1")

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(APISettingsError):
                load_api_settings()

    def test_missing_file_fails_without_interaction(self):
        with self.assertRaises(SystemExit) as context:
            main(["--models", "test-model", "--plan"])
        self.assertEqual(context.exception.code, 2)

    def test_review_requires_explicit_directories_and_models(self):
        with self.assertRaises(SystemExit) as context:
            main(["--file", str(self.input_path), "--models", "test-model", "--review", "--plan"])
        self.assertEqual(context.exception.code, 2)

    def test_plan_is_json_and_does_not_call_api_or_write_outputs(self):
        stdout = StringIO()
        with patch("scripts.evaluate.APIClient") as api_client, redirect_stdout(stdout):
            result = main(
                [
                    "--file",
                    str(self.input_path),
                    "--models",
                    "azure/gpt-5.4",
                    "--language",
                    "en",
                    "--plan",
                    "--plan-output",
                    str(self.root / "plan.json"),
                ]
            )

        self.assertEqual(result, 0)
        api_client.assert_not_called()
        manifest = json.loads(stdout.getvalue())
        self.assertTrue(manifest["valid"])
        self.assertEqual(manifest["unique_queries"], 1)
        self.assertEqual(manifest["models"][0]["protocol"], "chat_completions")
        self.assertTrue((self.root / "plan.json").exists())
        self.assertFalse((self.root / "input_evaluated_azure_gpt-5.4.xlsx").exists())
        self.assertFalse((self.root / "_checkpoint_azure_gpt-5.4_chat_completions.json").exists())

    def test_config_relative_paths_and_cli_override(self):
        config_path = self.root / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "file": "input.xlsx",
                    "models": ["azure/gpt-5.4"],
                    "model_protocols": {"azure/gpt-5.4": "chat_completions"},
                    "workers": 2,
                }
            ),
            encoding="utf-8",
        )
        args = Args()
        args.config = str(config_path)
        args.models = "alibaba/qwen3.7-max"
        args.model_protocols = "alibaba/qwen3.7-max=chat_completions"
        args.workers = 3

        resolved = resolve_run_config(args, DEFAULT_MODEL_PROTOCOLS)
        self.assertEqual(resolved.file_path, str(self.input_path.resolve()))
        self.assertEqual(resolved.eval_models, ["alibaba/qwen3.7-max"])
        self.assertEqual(resolved.model_protocols["alibaba/qwen3.7-max"], "chat_completions")
        self.assertEqual(resolved.workers, 3)

    def test_explicit_no_review_overrides_config_without_validating_review_dirs(self):
        config_path = self.root / "review-config.json"
        config_path.write_text(
            json.dumps(
                {
                    "file": "input.xlsx",
                    "models": ["test-model"],
                    "review": True,
                    "doc_dirs": ["missing-doc-dir"],
                    "review_models": ["reviewer"],
                }
            ),
            encoding="utf-8",
        )
        args = Args()
        args.config = str(config_path)
        args.review = False
        resolved = resolve_run_config(args, DEFAULT_MODEL_PROTOCOLS)
        self.assertFalse(resolved.review_enabled)
        self.assertEqual(resolved.doc_dirs, [])
        self.assertEqual(resolved.review_models, [])

    def test_execution_passes_environment_settings_to_api_client(self):
        valid_result = {
            "Response Relevance": {"score": 5, "reason": "Relevant"},
            "Citation Accuracy": {"score": 5, "reason": "Supported"},
            "Scope Completeness": {"score": 5, "reason": "Complete"},
            "Instruction Following": {"score": 5, "reason": "Followed"},
            "Readability and Naturalness": {"score": 5, "reason": "Clear"},
            "Overall Summary": "Good",
        }
        plan_path = self.root / "execution-plan.json"
        self.assertEqual(
            main(
                [
                    "--file",
                    str(self.input_path),
                    "--models",
                    "test-model",
                    "--language",
                    "en",
                    "--workers",
                    "1",
                    "--plan",
                    "--plan-output",
                    str(plan_path),
                ]
            ),
            0,
        )
        with (
            patch.dict(
                os.environ,
                {
                    "KB_QA_API_KEY": "env-test-key",
                    "KB_QA_API_BASE_URL": "https://env.example/v1",
                },
                clear=False,
            ),
            patch("scripts.evaluate.APIClient") as api_client,
            redirect_stdout(StringIO()),
        ):
            api_client.return_value.evaluate_batch.return_value = [
                {"index": 0, "result": valid_result, "status": "success"}
            ]
            result = main(["--execute-plan", str(plan_path)])

        self.assertEqual(result, 0)
        self.assertEqual(api_client.call_args.kwargs["base_url"], "https://env.example/v1")
        self.assertEqual(api_client.call_args.kwargs["api_key"], "env-test-key")
        self.assertTrue((self.root / "input_evaluated_test-model.xlsx").exists())

    def test_execute_plan_rejects_changed_input(self):
        plan_path = self.root / "changed-input-plan.json"
        with redirect_stdout(StringIO()):
            self.assertEqual(
                main(
                    [
                        "--file",
                        str(self.input_path),
                        "--models",
                        "test-model",
                        "--language",
                        "en",
                        "--plan",
                        "--plan-output",
                        str(plan_path),
                    ]
                ),
                0,
            )
        changed = pd.read_excel(self.input_path)
        changed.loc[len(changed)] = changed.iloc[0]
        changed.to_excel(self.input_path, index=False)

        with self.assertRaises(SystemExit) as context:
            main(["--execute-plan", str(plan_path)])
        self.assertEqual(context.exception.code, 2)

    def test_execute_plan_rejects_layout_override(self):
        """--execute-plan must reject layout/budget overrides like every other override."""
        plan_path = self.root / "plan.json"
        with redirect_stdout(StringIO()):
            self.assertEqual(
                main(
                    [
                        "--file",
                        str(self.input_path),
                        "--models",
                        "test-model",
                        "--language",
                        "en",
                        "--plan",
                        "--plan-output",
                        str(plan_path),
                    ]
                ),
                0,
            )
        with self.assertRaises(SystemExit) as context:
            main(["--execute-plan", str(plan_path), "--layout", "grouped"])
        self.assertEqual(context.exception.code, 2)
        with self.assertRaises(SystemExit) as context:
            main(["--execute-plan", str(plan_path), "--max-chunks", "5"])
        self.assertEqual(context.exception.code, 2)

    def test_direct_execution_requires_a_plan(self):
        with self.assertRaises(SystemExit) as context:
            main(["--file", str(self.input_path), "--models", "test-model"])
        self.assertEqual(context.exception.code, 2)

    def test_plan_warns_on_skipped_rows_and_missing_answer_column(self):
        input_path = self.root / "partial.xlsx"
        pd.DataFrame(
            [
                {"query": "q1", "filename": "a.md", "搜索结果": "c1", "score": 0.9},
                {"query": "q2", "filename": "a.md", "搜索结果": "c2", "score": 0.8},
                {"query": "q3", "filename": "a.md", "搜索结果": "c3", "score": 0.7},
                {"query": None, "filename": "a.md", "搜索结果": "c4", "score": 0.6},
            ]
        ).to_excel(input_path, index=False)

        stdout = StringIO()
        with redirect_stdout(stdout):
            result = main(
                [
                    "--file",
                    str(input_path),
                    "--models",
                    "test-model",
                    "--language",
                    "en",
                    "--plan",
                ]
            )
        self.assertEqual(result, 0)
        manifest = json.loads(stdout.getvalue())
        self.assertEqual(manifest["skipped_query_rows"], 1)
        warnings_text = "\n".join(manifest["warnings"])
        self.assertIn("empty query were skipped", warnings_text)
        self.assertIn("No answer column detected", warnings_text)

    def test_layout_override_is_used_and_recorded_in_plan(self):
        """Regression: --layout must override auto-detection and be audit-trailed."""
        input_path = self.root / "override.xlsx"
        pd.DataFrame(
            [
                {"query": "q", "filename": "a.md", "搜索结果": "c1", "score": 0.9, "大模型问答结果": "a1"},
                {"query": None, "filename": "a.md", "搜索结果": "c2", "score": 0.8, "大模型问答结果": None},
            ]
        ).to_excel(input_path, index=False)

        stdout = StringIO()
        with redirect_stdout(stdout):
            result = main(
                [
                    "--file",
                    str(input_path),
                    "--models",
                    "test-model",
                    "--language",
                    "en",
                    "--layout",
                    "row_by_row",
                    "--plan",
                ]
            )
        self.assertEqual(result, 0)
        manifest = json.loads(stdout.getvalue())
        self.assertEqual(manifest["layout"], "row_by_row")
        self.assertEqual(manifest["layout_source"], "manual")
        self.assertEqual(manifest["skipped_query_rows"], 1)
        self.assertEqual(manifest["unique_queries"], 1)

    def test_auto_layout_reports_source_auto(self):
        stdout = StringIO()
        with redirect_stdout(stdout):
            main(
                [
                    "--file",
                    str(self.input_path),
                    "--models",
                    "test-model",
                    "--language",
                    "en",
                    "--plan",
                ]
            )
        manifest = json.loads(stdout.getvalue())
        self.assertEqual(manifest["layout"], "row_by_row")
        self.assertEqual(manifest["layout_source"], "auto")

    def test_no_gui_helpers_are_called_in_plan_mode(self):
        self.assertFalse(hasattr(FileDetector, "select_file_dialog"))
        self.assertFalse(hasattr(FileDetector, "select_directories_dialog"))
        with (
            patch("scripts.evaluate.APIClient") as api_client,
            redirect_stdout(StringIO()),
        ):
            result = main(
                [
                    "--file",
                    str(self.input_path),
                    "--models",
                    "test-model",
                    "--plan",
                ]
            )
        self.assertEqual(result, 0)
        api_client.assert_not_called()


class ReviewThresholdTest(unittest.TestCase):
    def _module(self, results):
        return ReviewModule(
            eval_df=pd.DataFrame([{"query": "q", "answer": "a", "filename": "doc.md"}]),
            results=results,
            doc_dirs=[],
            review_models=["reviewer"],
            language="en",
        )

    @staticmethod
    def _result(rr=3, ca=5, sc=5, iff=5, rn=5):
        return {
            "index": 0,
            "status": "success",
            "result": {
                "Response Relevance": {"score": rr, "reason": "low"},
                "Citation Accuracy": {"score": ca, "reason": "ok"},
                "Scope Completeness": {"score": sc, "reason": "ok"},
                "Instruction Following": {"score": iff, "reason": "ok"},
                "Readability and Naturalness": {"score": rn, "reason": "ok"},
            },
        }

    def test_wa_below_badcase_line_triggers_review(self):
        # RR=3 others 5 -> WA = 4.4: not a badcase, no dim scored 1
        module = self._module([self._result(rr=3)])
        self.assertEqual(len(module.filter_low_scores(threshold=3.0)), 0)
        self.assertEqual(len(module.filter_low_scores(threshold=5.0)), 1)

    def test_any_dimension_one_triggers_regardless_of_wa(self):
        # RR=1 others 5 -> WA = 3.8 >= badcase line, but dim == 1 must trigger
        module = self._module([self._result(rr=1)])
        self.assertEqual(len(module.filter_low_scores(threshold=3.0)), 1)

    def test_review_all_passes_threshold_to_filter(self):
        module = self._module([self._result(rr=3)])
        with patch.object(module, "review_with_models", return_value=[]) as review_mock:
            module.review_all(max_workers=1, threshold=3.0)
        review_mock.assert_not_called()
        module2 = self._module([self._result(rr=2, ca=2, sc=2)])
        with patch.object(module2, "review_with_models", return_value=[]) as review_mock2:
            module2.review_all(max_workers=1, threshold=3.0)
        review_mock2.assert_called_once()


if __name__ == "__main__":
    unittest.main()
