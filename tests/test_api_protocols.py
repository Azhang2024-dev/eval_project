from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from scripts.api_client import (
    APIClient,
    PROTOCOL_ANTHROPIC_MESSAGES,
    PROTOCOL_CHAT_COMPLETIONS,
    PROTOCOL_RESPONSES,
    RESPONSE_MODE_JSON_SCHEMA,
    build_request_payload,
    call_model_json,
    extract_response_text,
    _parse_response_json,
    _validate_scores,
)
from scripts.file_detector import group_and_merge
from scripts.evaluate import DEFAULT_MODEL_PROTOCOLS
from scripts.eval_config import parse_model_mapping
from scripts.review_module import ReviewModule


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self.payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise __import__("requests").exceptions.HTTPError(
                f"status={self.status_code}", response=self
            )


VALID_EVALUATION = {
    "Response Relevance": {"score": 5, "reason": "Relevant"},
    "Citation Accuracy": {"score": 5, "reason": "Supported"},
    "Scope Completeness": {"score": 5, "reason": "Complete"},
    "Instruction Following": {"score": 5, "reason": "Followed"},
    "Readability and Naturalness": {"score": 5, "reason": "Clear"},
    "Overall Summary": "Good",
}


class ApiProtocolTest(unittest.TestCase):
    @patch("scripts.api_client.requests.post")
    def test_chat_completions_payload_and_response(self, post):
        post.return_value = FakeResponse({
            "choices": [{"message": {"content": json.dumps(VALID_EVALUATION)}}]
        })

        client = APIClient("https://example.test/v1", "key", "chat-model")
        result = client.evaluate_single("evaluate this")

        self.assertEqual(result, VALID_EVALUATION)
        args, kwargs = post.call_args
        self.assertEqual(args[0], "https://example.test/v1/chat/completions")
        self.assertIn("messages", kwargs["json"])
        self.assertNotIn("input", kwargs["json"])

    @patch("scripts.api_client.requests.post")
    def test_responses_payload_and_output_text(self, post):
        post.return_value = FakeResponse({"output_text": json.dumps(VALID_EVALUATION)})

        client = APIClient(
            "https://example.test/v1",
            "key",
            "responses-model",
            protocol=PROTOCOL_RESPONSES,
        )
        result = client.evaluate_single("evaluate this")

        self.assertEqual(result, VALID_EVALUATION)
        args, kwargs = post.call_args
        self.assertEqual(args[0], "https://example.test/v1/responses")
        self.assertEqual(kwargs["json"]["input"], "evaluate this")
        self.assertNotIn("messages", kwargs["json"])

    @patch("scripts.api_client.requests.post")
    def test_anthropic_messages_payload_headers_and_response(self, post):
        post.return_value = FakeResponse({
            "id": "msg_123",
            "content": [{"type": "text", "text": json.dumps(VALID_EVALUATION)}],
        })

        client = APIClient(
            "https://example.test/v1",
            "key",
            "anthropic-model",
            protocol=PROTOCOL_ANTHROPIC_MESSAGES,
        )
        result = client.evaluate_single("evaluate this")

        self.assertEqual(result, VALID_EVALUATION)
        args, kwargs = post.call_args
        self.assertEqual(args[0], "https://example.test/v1/messages")
        self.assertEqual(kwargs["headers"]["x-api-key"], "key")
        self.assertEqual(kwargs["headers"]["anthropic-version"], "2023-06-01")
        self.assertNotIn("Authorization", kwargs["headers"])
        self.assertEqual(kwargs["json"]["model"], "anthropic-model")
        self.assertEqual(kwargs["json"]["max_tokens"], 4096)
        self.assertIn("messages", kwargs["json"])
        self.assertNotIn("input", kwargs["json"])

    def test_anthropic_messages_rejects_json_schema_mode(self):
        with self.assertRaisesRegex(ValueError, "not supported by the Anthropic Messages adapter"):
            build_request_payload(
                "anthropic-model",
                "evaluate this",
                protocol=PROTOCOL_ANTHROPIC_MESSAGES,
                response_mode=RESPONSE_MODE_JSON_SCHEMA,
            )

    @patch("scripts.api_client.requests.post")
    def test_responses_nested_output_content(self, post):
        post.return_value = FakeResponse({
            "output": [{
                "type": "message",
                "content": [{"type": "output_text", "text": json.dumps(VALID_EVALUATION)}],
            }]
        })

        result = call_model_json(
            "https://example.test/v1",
            "key",
            "responses-model",
            "evaluate this",
            protocol=PROTOCOL_RESPONSES,
        )
        self.assertEqual(result, VALID_EVALUATION)

    def test_response_text_extraction_errors_with_top_level_keys(self):
        with self.assertRaisesRegex(ValueError, "top-level keys"):
            extract_response_text({"id": "resp_123", "output": []}, PROTOCOL_RESPONSES)

    def test_chat_completions_accepts_gemini_native_candidates_shape(self):
        content = json.dumps(VALID_EVALUATION)
        self.assertEqual(
            extract_response_text(
                {"candidates": [{"content": {"parts": [{"text": content}]}}]},
                PROTOCOL_CHAT_COMPLETIONS,
            ),
            content,
        )

    def test_parser_accepts_prose_wrapped_json_and_python_dict_style(self):
        self.assertEqual(_parse_response_json("Here is the result:\n```json\n{\"ok\": true}\n```"), {"ok": True})
        self.assertEqual(_parse_response_json("{'ok': True,}"), {"ok": True})

    def test_parser_converts_complete_plain_text_scorecard(self):
        content = (
            "1. Response Relevance (5): Directly answers the question.\n"
            "2. Citation Accuracy: score 4 - One detail differs from the source.\n"
            "3. Scope Completeness: 5/5 - Covers all required information.\n"
            "4. Instruction Following: 5 - Follows the requested format.\n"
            "5. Readability and Naturalness: 4 - Clear but slightly verbose.\n"
            "Overall Summary: Mostly correct with one citation detail to verify."
        )
        parsed = _parse_response_json(content, language="en")
        self.assertEqual(parsed["Response Relevance"]["score"], 5)
        self.assertEqual(parsed["Citation Accuracy"]["score"], 4)
        self.assertIn("citation detail", parsed["Overall Summary"])

    @patch("scripts.api_client.time.sleep")
    @patch("scripts.api_client.requests.post")
    def test_parser_retry_adds_strict_json_instruction_after_plain_text(self, post, sleep):
        post.side_effect = [
            FakeResponse({"choices": [{"message": {"content": "I evaluated the answer and it is good."}}]}),
            FakeResponse({"choices": [{"message": {"content": json.dumps(VALID_EVALUATION)}}]}),
        ]

        result = call_model_json(
            "https://example.test/v1",
            "key",
            "gemini-model",
            "evaluate this",
            protocol=PROTOCOL_CHAT_COMPLETIONS,
            retries=2,
        )

        self.assertEqual(result, VALID_EVALUATION)
        self.assertEqual(post.call_count, 2)
        second_prompt = post.call_args_list[1].kwargs["json"]["messages"][0]["content"]
        self.assertIn("FORMAT RETRY", second_prompt)
        sleep.assert_called_once_with(2)

    def test_max_thinking_effort_is_forwarded(self):
        chat_payload = build_request_payload("model", "prompt", thinking_effort="max")
        self.assertEqual(chat_payload["reasoning_effort"], "max")
        self.assertTrue(chat_payload["enable_thinking"])
        responses_payload = build_request_payload("model", "prompt", protocol=PROTOCOL_RESPONSES, thinking_effort="max")
        self.assertEqual(responses_payload["reasoning"], {"effort": "max"})

    def test_missing_retrieval_is_scored_at_minimum_without_api_call(self):
        columns = {
            "query": "query",
            "filename": "filename",
            "chunks": "搜索结果",
            "score": "score",
            "answer": "大模型问答结果",
        }
        df = pd.DataFrame([
            {"query": "q0", "filename": None, "搜索结果": None, "score": None, "大模型问答结果": "answer"},
            {"query": "q1", "filename": "doc.md", "搜索结果": "context", "score": 1.0, "大模型问答结果": "answer"},
        ])
        grouped = group_and_merge(df, columns, "layout_a")
        self.assertEqual(grouped[0]["retrieval_status"], "missing_filename_and_search_result")
        self.assertEqual(grouped[1]["retrieval_status"], "available")

    def test_incomplete_scorecard_is_rejected(self):
        with self.assertRaises(ValueError):
            _validate_scores({"分数": 5, "理由": "partial"}, "zh")

    def test_validate_scores_rejects_out_of_range_integers(self):
        """Regression: integer scores outside 1-5 must fail, not clamp silently."""
        for bad_score in (0, 6, -1, 7):
            result = dict(VALID_EVALUATION)
            result["Response Relevance"] = {"score": bad_score, "reason": "r"}
            with self.assertRaisesRegex(ValueError, "out-of-range"):
                _validate_scores(result, "en")

    def test_validate_scores_rounds_fractional_scores(self):
        """Fractional scores are tolerated and rounded (4.6 -> 5)."""
        result = dict(VALID_EVALUATION)
        result["Response Relevance"] = {"score": 4.6, "reason": "r"}
        _validate_scores(result, "en")
        self.assertEqual(result["Response Relevance"]["score"], 5)
        result["Response Relevance"] = {"score": 4.4, "reason": "r"}
        _validate_scores(result, "en")
        self.assertEqual(result["Response Relevance"]["score"], 4)

    def test_validate_scores_rejects_fraction_rounding_out_of_range(self):
        result = dict(VALID_EVALUATION)
        result["Response Relevance"] = {"score": 5.5, "reason": "r"}
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            _validate_scores(result, "en")

    @patch("scripts.api_client.time.sleep")
    @patch("scripts.api_client.requests.post")
    def test_schema_failure_retries_with_format_instruction(self, post, sleep):
        """Regression: a validator failure must reuse the format-retry channel."""
        bad = dict(VALID_EVALUATION)
        bad["Response Relevance"] = {"score": 6, "reason": "r"}
        post.side_effect = [
            FakeResponse({"choices": [{"message": {"content": json.dumps(bad)}}]}),
            FakeResponse({"choices": [{"message": {"content": json.dumps(VALID_EVALUATION)}}]}),
        ]

        result = call_model_json(
            "https://example.test/v1",
            "key",
            "model",
            "evaluate this",
            validator=lambda parsed: _validate_scores(parsed, "en"),
        )

        self.assertEqual(result, VALID_EVALUATION)
        self.assertEqual(post.call_count, 2)
        second_prompt = post.call_args_list[1].kwargs["json"]["messages"][0]["content"]
        self.assertIn("FORMAT RETRY", second_prompt)

    @patch("scripts.api_client.time.sleep")
    @patch("scripts.api_client.requests.post")
    def test_schema_failure_marks_item_failed_after_retries(self, post, sleep):
        bad = dict(VALID_EVALUATION)
        bad["Response Relevance"] = {"score": 7, "reason": "r"}
        post.side_effect = [
            FakeResponse({"choices": [{"message": {"content": json.dumps(bad)}}]}),
            FakeResponse({"choices": [{"message": {"content": json.dumps(bad)}}]}),
            FakeResponse({"choices": [{"message": {"content": json.dumps(bad)}}]}),
        ]

        with self.assertRaisesRegex(RuntimeError, "after 3 retries"):
            call_model_json(
                "https://example.test/v1",
                "key",
                "model",
                "evaluate this",
                validator=lambda parsed: _validate_scores(parsed, "en"),
            )

    def test_gateway_aliases_max_to_xhigh_for_gpt54_mini(self):
        chat_payload = build_request_payload(
            "azure/gpt-5.4-mini",
            "prompt",
            thinking_effort="max",
        )
        self.assertEqual(chat_payload["reasoning_effort"], "xhigh")
        responses_payload = build_request_payload(
            "azure/gpt-5.4-mini",
            "prompt",
            protocol=PROTOCOL_RESPONSES,
            thinking_effort="max",
        )
        self.assertEqual(responses_payload["reasoning"], {"effort": "xhigh"})

    def test_responses_thinking_and_json_schema_payload(self):
        payload = build_request_payload(
            "responses-model",
            "evaluate this",
            protocol=PROTOCOL_RESPONSES,
            thinking_effort="high",
            response_mode=RESPONSE_MODE_JSON_SCHEMA,
        )
        self.assertEqual(payload["reasoning"], {"effort": "high"})
        self.assertEqual(payload["text"]["format"]["type"], "json_schema")
        self.assertNotIn("enable_thinking", payload)
        self.assertNotIn("reasoning_effort", payload)

    def test_default_model_protocols_are_chat_completions(self):
        self.assertEqual(DEFAULT_MODEL_PROTOCOLS["azure/gpt-5.4"], PROTOCOL_CHAT_COMPLETIONS)
        self.assertEqual(DEFAULT_MODEL_PROTOCOLS["alibaba/qwen3.7-max"], PROTOCOL_CHAT_COMPLETIONS)
        self.assertEqual(DEFAULT_MODEL_PROTOCOLS["bytedance/deepseek-v4-pro"], PROTOCOL_CHAT_COMPLETIONS)

    def test_model_mapping_validation(self):
        result = parse_model_mapping(
            "new-model=responses,old-model=chat_completions",
            "--model-protocols",
            lambda value: value,
        )
        self.assertEqual(result["new-model"], "responses")
        with self.assertRaises(ValueError):
            parse_model_mapping("new-model", "--model-protocols", lambda value: value)


class ReviewProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._api_env = patch.dict("os.environ", {"KB_QA_API_KEY": "test-key"}, clear=False)
        cls._api_env.start()

    @classmethod
    def tearDownClass(cls):
        cls._api_env.stop()

    @patch("scripts.api_client.requests.post")
    def test_review_uses_responses_protocol(self, post):
        post.return_value = FakeResponse({
            "output_text": json.dumps({"Summary": {"score_reasonable": "Reasonable"}})
        })
        module = ReviewModule(
            eval_df=None,
            results=[],
            doc_dirs=[],
            review_models=["responses-reviewer"],
            language="en",
            model_protocols={"responses-reviewer": PROTOCOL_RESPONSES},
        )
        module.prompt_template = "{query}|{filenames}|{original_text}|{answer}|{current_scores}"
        item = {
            "filename": "doc.md",
            "query": "What happened?",
            "answer": "An answer.",
            "result": {},
        }

        with (
            patch.object(module, "_find_original_doc_single", return_value="C:/fake/doc.md"),
            patch.object(module, "_read_doc_content", return_value="Original text."),
        ):
            result = module.review_single(item, "responses-reviewer")

        self.assertIsNone(result["error"])
        self.assertEqual(post.call_args.args[0], "https://maragate.aibotplatform.com/v1/responses")
        payload = post.call_args.kwargs["json"]
        self.assertIn("input", payload)
        self.assertNotIn("messages", payload)

    @patch("scripts.api_client.requests.post")
    def test_review_uses_anthropic_messages_protocol(self, post):
        post.return_value = FakeResponse({
            "content": [{"type": "text", "text": json.dumps({"Summary": {}})}]
        })
        module = ReviewModule(
            eval_df=None,
            results=[],
            doc_dirs=[],
            review_models=["anthropic-model"],
            language="en",
            model_protocols={"anthropic-model": PROTOCOL_ANTHROPIC_MESSAGES},
        )
        module.prompt_template = "{query}|{filenames}|{original_text}|{answer}|{current_scores}"
        item = {"filename": "doc.md", "query": "q", "answer": "a", "result": {}}

        with (
            patch.object(module, "_find_original_doc_single", return_value="C:/fake/doc.md"),
            patch.object(module, "_read_doc_content", return_value="Original text."),
        ):
            result = module.review_single(item, "anthropic-model")

        self.assertIsNone(result["error"])
        self.assertEqual(post.call_args.args[0], "https://maragate.aibotplatform.com/v1/messages")
        self.assertEqual(post.call_args.kwargs["headers"]["anthropic-version"], "2023-06-01")

    @patch("scripts.api_client.requests.post")
    def test_review_json_schema_mode_uses_review_schema(self, post):
        post.return_value = FakeResponse({
            "output_text": json.dumps({"Summary": {"score_reasonable": "Reasonable", "main_issues": "", "note": ""}})
        })
        module = ReviewModule(
            eval_df=None,
            results=[],
            doc_dirs=[],
            review_models=["responses-reviewer"],
            language="en",
            model_protocols={"responses-reviewer": PROTOCOL_RESPONSES},
            model_response_modes={"responses-reviewer": RESPONSE_MODE_JSON_SCHEMA},
        )
        module.prompt_template = "{query}|{filenames}|{original_text}|{answer}|{current_scores}"
        item = {"filename": "doc.md", "query": "q", "answer": "a", "result": {}}

        with (
            patch.object(module, "_find_original_doc_single", return_value="C:/fake/doc.md"),
            patch.object(module, "_read_doc_content", return_value="Original text."),
        ):
            result = module.review_single(item, "responses-reviewer")

        self.assertIsNone(result["error"])
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["text"]["format"]["type"], "json_schema")
        self.assertIn("Summary", payload["text"]["format"]["schema"]["properties"])


class CheckpointProtocolTest(unittest.TestCase):
    def _client(self, protocol=PROTOCOL_CHAT_COMPLETIONS, input_fingerprint=None):
        return APIClient(
            "https://example.test/v1",
            "key",
            "model",
            max_workers=1,
            protocol=protocol,
            language="en",
            input_fingerprint=input_fingerprint,
        )

    def test_checkpoint_contains_protocol_metadata_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = str(Path(tmp) / "checkpoint.json")
            client = self._client(PROTOCOL_CHAT_COMPLETIONS)
            client.evaluate_single = Mock(return_value=VALID_EVALUATION)
            results = client.evaluate_batch(
                [{"index": 0, "query": "q", "filename": "f", "chunks": [], "answer": "a"}],
                lambda *args: "prompt",
                "en",
                checkpoint_file=checkpoint,
            )
            self.assertEqual(results[0]["status"], "success")
            saved = json.loads(Path(checkpoint).read_text(encoding="utf-8"))
            self.assertEqual(saved["meta"]["protocol"], PROTOCOL_CHAT_COMPLETIONS)
            self.assertEqual(saved["meta"]["model"], "model")

            with self.assertRaisesRegex(RuntimeError, "metadata mismatch"):
                self._client(PROTOCOL_RESPONSES).evaluate_batch(
                    [], lambda *args: "prompt", "en", checkpoint_file=checkpoint
                )

    def test_fingerprint_is_written_and_legacy_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = str(Path(tmp) / "fingerprinted.json")
            client = self._client(input_fingerprint="input-sha")
            client.evaluate_single = Mock(return_value=VALID_EVALUATION)
            result = client.evaluate_batch(
                [{"index": 0, "query": "q", "filename": "f", "chunks": [], "answer": "a"}],
                lambda *args: "prompt",
                "en",
                checkpoint_file=checkpoint,
            )
            self.assertEqual(result[0]["status"], "success")
            saved = json.loads(Path(checkpoint).read_text(encoding="utf-8"))
            self.assertEqual(saved["meta"]["input_fingerprint"], "input-sha")

            legacy = str(Path(tmp) / "legacy.json")
            Path(legacy).write_text(
                json.dumps({"completed": {"0": VALID_EVALUATION}, "failed": []}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "input fingerprint"):
                self._client(input_fingerprint="input-sha").evaluate_batch(
                    [], lambda *args: "prompt", "en", checkpoint_file=legacy
                )

    def test_responses_batch_smoke_writes_responses_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = str(Path(tmp) / "responses.json")
            client = self._client(PROTOCOL_RESPONSES)
            client.evaluate_single = Mock(return_value=VALID_EVALUATION)
            results = client.evaluate_batch(
                [{"index": 0, "query": "q", "filename": "f", "chunks": [], "answer": "a"}],
                lambda *args: "prompt",
                "en",
                checkpoint_file=checkpoint,
            )
            self.assertEqual(results[0]["status"], "success")
            saved = json.loads(Path(checkpoint).read_text(encoding="utf-8"))
            self.assertEqual(saved["meta"]["protocol"], PROTOCOL_RESPONSES)

    def test_legacy_chat_checkpoint_is_accepted_but_responses_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = str(Path(tmp) / "legacy.json")
            Path(checkpoint).write_text(
                json.dumps({"completed": {"0": VALID_EVALUATION}, "failed": []}),
                encoding="utf-8",
            )
            with self.assertWarns(RuntimeWarning):
                result = self._client(PROTOCOL_CHAT_COMPLETIONS).evaluate_batch(
                    [], lambda *args: "prompt", "en", checkpoint_file=checkpoint
                )
            self.assertEqual(result[0]["status"], "success")
            with self.assertRaisesRegex(RuntimeError, "Legacy checkpoint|metadata mismatch"):
                self._client(PROTOCOL_RESPONSES).evaluate_batch(
                    [], lambda *args: "prompt", "en", checkpoint_file=checkpoint
                )


if __name__ == "__main__":
    unittest.main()
