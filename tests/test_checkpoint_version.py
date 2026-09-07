"""Tests for checkpoint version gating: schema/prompt changes must break resume."""

import json
import os
import tempfile
import unittest
from unittest.mock import Mock

from scripts.api_client import APIClient, CHECKPOINT_SCHEMA_VERSION, PROMPT_VERSION

VALID_EVALUATION = {
    "Response Relevance": {"score": 5, "reason": "good"},
    "Citation Accuracy": {"score": 5, "reason": "good"},
    "Scope Completeness": {"score": 5, "reason": "good"},
    "Instruction Following": {"score": 5, "reason": "good"},
    "Readability and Naturalness": {"score": 5, "reason": "good"},
    "Overall Summary": "summary",
}


def _mk_client():
    return APIClient(
        base_url="http://example.invalid/v1",
        api_key="k",
        model="m",
        input_fingerprint="fp",
    )


def _mk_meta(**overrides):
    meta = {
        "model": "m",
        "protocol": "chat_completions",
        "response_mode": "text",
        "language": "en",
        "input_fingerprint": "fp",
        "prompt_version": PROMPT_VERSION,
        "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
    }
    meta.update(overrides)
    return meta


class CheckpointVersionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "cp.json")

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, meta, completed=None, failed=None):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "completed": completed or {}, "failed": failed or []}, f)

    def test_same_version_accepted(self):
        self._write(_mk_meta())
        completed, failed, legacy = _mk_client()._load_checkpoint(self.path, "en")
        self.assertEqual(completed, {})
        self.assertEqual(failed, set())
        self.assertFalse(legacy)

    def test_missing_prompt_version_rejected(self):
        meta = _mk_meta()
        del meta["prompt_version"]
        self._write(meta)
        with self.assertRaises(RuntimeError) as ctx:
            _mk_client()._load_checkpoint(self.path, "en")
        self.assertIn("prompt_version", str(ctx.exception))

    def test_wrong_prompt_version_rejected(self):
        self._write(_mk_meta(prompt_version=PROMPT_VERSION + 1))
        with self.assertRaises(RuntimeError):
            _mk_client()._load_checkpoint(self.path, "en")

    def test_missing_checkpoint_schema_rejected(self):
        """Old-schema checkpoints (no status field) must not silently resume."""
        meta = _mk_meta()
        del meta["checkpoint_schema"]
        self._write(meta)
        with self.assertRaises(RuntimeError) as ctx:
            _mk_client()._load_checkpoint(self.path, "en")
        self.assertIn("checkpoint_schema", str(ctx.exception))

    def test_wrong_checkpoint_schema_rejected(self):
        self._write(_mk_meta(checkpoint_schema=CHECKPOINT_SCHEMA_VERSION - 1))
        with self.assertRaises(RuntimeError):
            _mk_client()._load_checkpoint(self.path, "en")

    def test_response_mode_mismatch_rejected(self):
        """Checkpoints must not be reused across response modes (text vs json_schema)."""
        self._write(_mk_meta())
        client = APIClient(
            base_url="http://example.invalid/v1",
            api_key="k",
            model="m",
            input_fingerprint="fp",
            response_mode="json_schema",
        )
        with self.assertRaisesRegex(RuntimeError, "response_mode"):
            client._load_checkpoint(self.path, "en")

    def test_empty_status_survives_checkpoint_roundtrip(self):
        """Regression: resumed runs must keep 'empty' status instead of losing it to 'success'."""
        client = _mk_client()
        client.evaluate_single = Mock(return_value=VALID_EVALUATION)
        items = [
            {"index": 0, "query": "q0", "filename": "f", "chunks": [{"text": "c"}], "answer": "answer text"},
            {"index": 1, "query": "q1", "filename": "f", "chunks": [{"text": "c"}], "answer": "   "},
        ]
        results = client.evaluate_batch(items, lambda *a: "prompt", "en", checkpoint_file=self.path)
        statuses = {r["index"]: r["status"] for r in results}
        self.assertEqual(statuses, {0: "success", 1: "empty"})

        saved = json.loads(open(self.path, encoding="utf-8").read())
        self.assertEqual(saved["meta"]["checkpoint_schema"], CHECKPOINT_SCHEMA_VERSION)
        self.assertEqual(saved["completed"]["1"]["status"], "empty")

        resumed = _mk_client().evaluate_batch([], lambda *a: "prompt", "en", checkpoint_file=self.path)
        resumed_statuses = {r["index"]: r["status"] for r in resumed}
        self.assertEqual(resumed_statuses, {0: "success", 1: "empty"})

    def test_old_schema_checkpoint_rejected_on_resume(self):
        """A checkpoint written in the previous schema is refused with a clear error."""
        self._write(_mk_meta(checkpoint_schema=1), completed={"0": VALID_EVALUATION})
        with self.assertRaisesRegex(RuntimeError, "checkpoint_schema"):
            _mk_client().evaluate_batch([], lambda *a: "prompt", "en", checkpoint_file=self.path)

    def test_checkpoint_entries_outside_eval_set_are_dropped(self):
        """Regression: resuming with a truncated eval set (--test) must not
        leak stale completed entries into the results."""
        client = _mk_client()
        client.evaluate_single = Mock(return_value=VALID_EVALUATION)
        items = [
            {"index": 0, "query": "q0", "filename": "f", "chunks": [{"text": "c"}], "answer": "a0"},
            {"index": 1, "query": "q1", "filename": "f", "chunks": [{"text": "c"}], "answer": "a1"},
        ]
        client.evaluate_batch(items, lambda *a: "prompt", "en", checkpoint_file=self.path)

        resumed = _mk_client().evaluate_batch(
            [items[0]], lambda *a: "prompt", "en", checkpoint_file=self.path
        )
        self.assertEqual([r["index"] for r in resumed], [0])
        self.assertEqual(len(resumed), 1)


if __name__ == "__main__":
    unittest.main()
