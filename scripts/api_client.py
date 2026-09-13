"""API client: parallel evaluation with Chat Completions/Responses support."""

from __future__ import annotations

import ast
import json
import os
import re
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests


PROTOCOL_CHAT_COMPLETIONS = "chat_completions"
PROTOCOL_RESPONSES = "responses"
PROTOCOL_ANTHROPIC_MESSAGES = "anthropic_messages"
SUPPORTED_PROTOCOLS = frozenset({
    PROTOCOL_CHAT_COMPLETIONS,
    PROTOCOL_RESPONSES,
    PROTOCOL_ANTHROPIC_MESSAGES,
})

RESPONSE_MODE_TEXT = "text"
RESPONSE_MODE_JSON_SCHEMA = "json_schema"
SUPPORTED_RESPONSE_MODES = frozenset({RESPONSE_MODE_TEXT, RESPONSE_MODE_JSON_SCHEMA})

# Bump when EVAL_PROMPT/REVIEW_PROMPT text or scoring rules change; old checkpoints then cannot resume.
PROMPT_VERSION = 3

# Bump when the on-disk checkpoint structure changes (e.g. completed entries
# gain a status field); older checkpoints are rejected so resumed runs never
# mix incompatible result shapes or lose status information.
CHECKPOINT_SCHEMA_VERSION = 2

# The gateway exposes the highest reasoning level as xhigh for this model,
# while the skill configuration keeps the provider-neutral semantic value max.
_GATEWAY_THINKING_EFFORT_ALIASES = {
    "azure/gpt-5.4-mini": {"max": "xhigh"},
}


def _resolve_wire_thinking_effort(model: str, thinking_effort: str | None) -> str | None:
    if thinking_effort is None:
        return None
    return _GATEWAY_THINKING_EFFORT_ALIASES.get(model, {}).get(thinking_effort, thinking_effort)


def _meta(language: str):
    from scripts.eval_stats import LANGUAGE_META
    return LANGUAGE_META.get(language, LANGUAGE_META["en"])


def _dims_all(language: str) -> list[str]:
    meta = _meta(language)
    return meta.dims + [meta.summary_key]


def validate_protocol(protocol: str) -> str:
    if protocol not in SUPPORTED_PROTOCOLS:
        supported = ", ".join(sorted(SUPPORTED_PROTOCOLS))
        raise ValueError(f"Unsupported model protocol {protocol!r}; expected one of: {supported}")
    return protocol


def validate_response_mode(response_mode: str) -> str:
    if response_mode not in SUPPORTED_RESPONSE_MODES:
        supported = ", ".join(sorted(SUPPORTED_RESPONSE_MODES))
        raise ValueError(f"Unsupported response mode {response_mode!r}; expected one of: {supported}")
    return response_mode


def build_evaluation_json_schema(language: str = "en") -> dict[str, Any]:
    """Build the optional structured-output schema for evaluator responses."""
    meta = _meta(language)
    dimensions = meta.dims
    score_key = meta.score_key
    reason_key = meta.reason_key
    summary_key = meta.summary_key

    properties: dict[str, Any] = {}
    required = []
    for dimension in dimensions:
        properties[dimension] = {
            "type": "object",
            "properties": {
                score_key: {"type": "integer", "minimum": 1, "maximum": 5},
                reason_key: {"type": "string"},
            },
            "required": [score_key, reason_key],
            "additionalProperties": False,
        }
        required.append(dimension)

    properties[summary_key] = {"type": "string"}
    required.append(summary_key)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _collect_text(value: Any) -> str:
    """Collect text fragments from Responses message/content blocks."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(part for item in value if (part := _collect_text(item)))
    if not isinstance(value, dict):
        return ""

    if value.get("type") == "reasoning":
        return ""

    text = value.get("text")
    if isinstance(text, str):
        return text
    if isinstance(text, dict):
        nested_text = _collect_text(text)
        if nested_text:
            return nested_text

    for key in ("content", "output", "value", "parts"):
        nested = _collect_text(value.get(key))
        if nested:
            return nested
    return ""


def extract_response_text(payload: Any, protocol: str) -> str:
    """Extract assistant text from either supported API response shape."""
    validate_protocol(protocol)
    if not isinstance(payload, dict):
        raise ValueError(f"API response must be a JSON object, got {type(payload).__name__}")

    if protocol == PROTOCOL_ANTHROPIC_MESSAGES:
        content = _collect_text(payload.get("content"))
        if content:
            return content
        raise ValueError(
            "Anthropic Messages response has no text content; "
            f"top-level keys: {sorted(payload.keys())}"
        )

    if protocol == PROTOCOL_CHAT_COMPLETIONS:
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                message = choice.get("message")
                if isinstance(message, dict):
                    for value in (message.get("content"), message.get("output_text")):
                        content = _collect_text(value)
                        if content:
                            return content
                # Some OpenAI-compatible gateways return completion-style text
                # even when the endpoint is named /chat/completions.
                content = _collect_text(choice.get("text"))
                if content:
                    return content

        # A Gemini-compatible gateway may return the native candidates/content/
        # parts shape while still routing through the chat-completions adapter.
        content = _collect_text(payload.get("candidates"))
        if content:
            return content
        raise ValueError(
            "Chat Completions response has no assistant content; "
            f"top-level keys: {sorted(payload.keys())}"
        )

    output_text = payload.get("output_text")
    content = _collect_text(output_text)
    if content:
        return content

    output = payload.get("output")
    content = _collect_text(output)
    if content:
        return content

    # A few OpenAI-compatible gateways wrap the response object.
    nested_response = payload.get("response")
    if isinstance(nested_response, dict):
        return extract_response_text(nested_response, PROTOCOL_RESPONSES)

    raise ValueError(
        "Responses response has no output text; "
        f"top-level keys: {sorted(payload.keys())}"
    )


def build_request_payload(
    model: str,
    prompt: str,
    protocol: str = PROTOCOL_CHAT_COMPLETIONS,
    thinking_effort: str | None = None,
    response_mode: str = RESPONSE_MODE_TEXT,
    response_schema: dict[str, Any] | None = None,
    language: str = "en",
    max_tokens: int = 4096,
) -> dict[str, Any]:
    """Build a protocol-specific request without mixing fields between APIs."""
    validate_protocol(protocol)
    validate_response_mode(response_mode)
    wire_thinking_effort = _resolve_wire_thinking_effort(model, thinking_effort)

    if protocol == PROTOCOL_CHAT_COMPLETIONS:
        if response_mode != RESPONSE_MODE_TEXT:
            raise ValueError("json_schema response mode is only supported with the Responses protocol")
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
        if wire_thinking_effort:
            payload["enable_thinking"] = True
            payload["reasoning_effort"] = wire_thinking_effort
        return payload

    if protocol == PROTOCOL_ANTHROPIC_MESSAGES:
        if response_mode != RESPONSE_MODE_TEXT:
            raise ValueError("json_schema response mode is not supported by the Anthropic Messages adapter")
        return {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }

    payload = {
        "model": model,
        "input": prompt,
    }
    if wire_thinking_effort:
        payload["reasoning"] = {"effort": wire_thinking_effort}

    if response_mode == RESPONSE_MODE_JSON_SCHEMA:
        payload["text"] = {
            "format": {
                "type": "json_schema",
                "name": "kb_qa_evaluation",
                "strict": True,
                "schema": response_schema or build_evaluation_json_schema(language),
            }
        }
    return payload


def _strip_code_fence(content: str) -> str:
    """Remove a Markdown code fence without changing the enclosed text."""
    stripped = content.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _first_json_object(content: str) -> dict | None:
    """Find the first valid JSON object embedded in surrounding model prose."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", content):
        try:
            parsed, _ = decoder.raw_decode(content[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _clean_scorecard_line(line: str) -> str:
    """Normalize common Markdown/list decoration in non-JSON scorecards."""
    line = re.sub(r"^\s*(?:[-*•>]+\s*|\d+[.)]\s*)+", "", line)
    line = re.sub(r"[*_`#]", "", line)
    return line.strip()


def _plain_text_scorecard(content: str, language: str) -> dict | None:
    """Conservatively convert a complete human-readable scorecard to our schema.

    Gemini sometimes returns a well-formed evaluation as bullets or headings
    instead of JSON. We only convert output when all five dimensions have an
    explicit 1-5 score and a non-empty reason; arbitrary prose is rejected so
    that a formatting workaround cannot silently invent evaluation results.
    """
    keys = _meta(language)
    dimensions = keys.dims
    score_key = keys.score_key
    reason_key = keys.reason_key
    summary_key = keys.summary_key
    lines = [_clean_scorecard_line(line) for line in content.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return None

    dimension_lower = [dimension.casefold() for dimension in dimensions]
    summary_lower = summary_key.casefold()
    summary_line_index = next(
        (index for index, line in enumerate(lines) if line.casefold().startswith(summary_lower)),
        len(lines),
    )

    def starts_dimension(line: str) -> int | None:
        folded = line.casefold()
        for index, name in enumerate(dimension_lower):
            if folded.startswith(name):
                return index
        return None

    score_pattern = re.compile(
        r"(?:score|rating|points?|分数|得分|评分)\s*[:：=]?\s*\(?\s*([1-5])\s*(?:/\s*5)?\s*\)?"
        r"|\(\s*([1-5])\s*(?:/\s*5)?\s*\)"
        r"|^[\s:：\-–—]*([1-5])\s*(?:/\s*5|分|点)?(?=\s|$|[:：,.;()\-–—])",
        re.IGNORECASE,
    )

    parsed: dict[str, dict[str, Any]] = {}
    dimension_positions: list[tuple[int, int]] = []
    for line_index, line in enumerate(lines):
        dimension_index = starts_dimension(line)
        if dimension_index is not None:
            dimension_positions.append((line_index, dimension_index))

    if len(dimension_positions) < len(dimensions):
        return None

    for position_index, (line_index, dimension_index) in enumerate(dimension_positions):
        dimension = dimensions[dimension_index]
        next_header = (
            dimension_positions[position_index + 1][0]
            if position_index + 1 < len(dimension_positions)
            else len(lines)
        )
        # A heading may put its score/reason on the next one or two lines.
        block_end = min(next_header, line_index + 3, summary_line_index)
        block = " ".join(lines[line_index:max(line_index + 1, block_end)])
        remainder = block[len(dimension):].strip()
        score_match = score_pattern.search(remainder)
        if not score_match:
            return None

        score_groups = [group for group in score_match.groups() if group is not None]
        if not score_groups:
            return None
        score = int(score_groups[0])
        reason = remainder[score_match.end():].strip(" \t:：-–—,.;")
        reason = re.sub(r"^(?:reason|理由)\s*[:：-]\s*", "", reason, flags=re.IGNORECASE)
        if not reason:
            return None
        parsed[dimension] = {score_key: score, reason_key: reason}

    if len(parsed) != len(dimensions):
        return None

    summary_pattern = re.compile(
        rf"(?:^|\n)\s*{re.escape(summary_key)}\s*[:：-]?\s*(.*)$",
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    summary_match = summary_pattern.search(content)
    if summary_match:
        summary = summary_match.group(1).strip()
    else:
        summary = "[Dimension Scores]\n" + "\n".join(
            f"- {dimension}({parsed[dimension][score_key]}): {parsed[dimension][reason_key]}"
            for dimension in dimensions
        )

    if not summary:
        summary = "[Dimension Scores]"
    parsed[summary_key] = summary
    return parsed


def _parse_response_json(content: str, language: str = "en") -> dict:
    """Parse strict JSON plus safe fallbacks for common LLM formatting errors."""
    if not isinstance(content, str):
        raise ValueError(f"Model response text must be a string, got {type(content).__name__}")

    cleaned = _strip_code_fence(content)
    parse_error: Exception | None = None
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as error:
        parse_error = error
        parsed = _first_json_object(cleaned)

        # A few models emit Python-dict style quotes or trailing commas. This
        # fallback is intentionally limited to a response that starts with an
        # object, so arbitrary prose is never evaluated as Python.
        if parsed is None and cleaned.startswith("{"):
            try:
                literal = ast.literal_eval(cleaned)
            except (SyntaxError, ValueError):
                literal = None
            if isinstance(literal, dict):
                parsed = literal

        if parsed is None:
            parsed = _plain_text_scorecard(cleaned, language)
            if parsed is None:
                raise ValueError(
                    "Could not parse a JSON object from model response; model may have returned "
                    f"plain text. Response preview: {cleaned[:300]}"
                ) from parse_error

    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object from model, got {type(parsed).__name__}")
    return parsed


_JSON_RETRY_INSTRUCTION = (
    "FORMAT RETRY: Your previous response could not be parsed reliably. Return exactly one valid "
    "JSON object and nothing else. Do not use Markdown fences, bullets, headings, or explanatory "
    "prose. Use double-quoted JSON keys and string values, and include every required field from "
    "the requested output schema."
)

def call_model_json(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    protocol: str = PROTOCOL_CHAT_COMPLETIONS,
    response_mode: str = RESPONSE_MODE_TEXT,
    response_schema: dict[str, Any] | None = None,
    language: str = "en",
    retries: int = 3,
    thinking_effort: str | None = None,
    max_tokens: int = 4096,
    validator=None,
) -> dict:
    """Call a configured model and parse its response as a JSON object.

    *validator* (optional) is applied to every parsed response. A
    ``ValueError`` from the validator triggers the same format-retry channel
    as a JSON parse failure, so schema-level failures (missing fields,
    out-of-range scores) get one retry with the format instruction before the
    call is abandoned.
    """
    validate_protocol(protocol)
    validate_response_mode(response_mode)
    if protocol == PROTOCOL_CHAT_COMPLETIONS:
        endpoint = "chat/completions"
    elif protocol == PROTOCOL_RESPONSES:
        endpoint = "responses"
    else:
        endpoint = "messages"

    url = f"{base_url.rstrip('/')}/{endpoint}"
    if protocol == PROTOCOL_ANTHROPIC_MESSAGES:
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
    else:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
    timeout = 240 if thinking_effort else 120

    last_error: Exception | None = None
    format_retry = False
    for attempt in range(retries):
        attempt_prompt = f"{prompt}\n\n{_JSON_RETRY_INSTRUCTION}" if format_retry else prompt
        payload = build_request_payload(
            model=model,
            prompt=attempt_prompt,
            protocol=protocol,
            thinking_effort=thinking_effort,
            response_mode=response_mode,
            response_schema=response_schema,
            language=language,
            max_tokens=max_tokens,
        )
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            response_payload = response.json()
            content = extract_response_text(response_payload, protocol)
            parsed = _parse_response_json(content, language=language)
            if validator is not None:
                validator(parsed)
            return parsed
        except requests.exceptions.RequestException as error:
            status = getattr(getattr(error, "response", None), "status_code", None)
            body = ""
            error_response = getattr(error, "response", None)
            if error_response is not None:
                try:
                    body = error_response.text[:500]
                except Exception:
                    body = ""
            suffix = f"; response={body}" if body else ""
            last_error = RuntimeError(f"HTTP request failed{f' with status {status}' if status else ''}: {error}{suffix}")
        except (KeyError, IndexError, TypeError, json.JSONDecodeError, ValueError) as error:
            last_error = error
            format_retry = True

        if attempt < retries - 1:
            time.sleep(2 ** (attempt + 1))

    raise RuntimeError(f"API call failed after {retries} retries: {last_error}")


class APIClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        max_workers: int = 10,
        thinking_effort: str | None = None,
        protocol: str = PROTOCOL_CHAT_COMPLETIONS,
        response_mode: str = RESPONSE_MODE_TEXT,
        response_schema: dict[str, Any] | None = None,
        language: str = "en",
        max_tokens: int = 4096,
        input_fingerprint: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_workers = max_workers
        self.thinking_effort = thinking_effort
        self.protocol = validate_protocol(protocol)
        self.response_mode = validate_response_mode(response_mode)
        self.response_schema = response_schema
        self.language = language
        self.max_tokens = max_tokens
        self.input_fingerprint = input_fingerprint
        if self.protocol == PROTOCOL_ANTHROPIC_MESSAGES and self.thinking_effort:
            warnings.warn(
                "thinking_effort is not translated to Anthropic extended thinking; "
                "the Messages request will run without a thinking parameter.",
                RuntimeWarning,
                stacklevel=2,
            )

    def evaluate_single(self, prompt: str, retries: int = 3) -> dict:
        return call_model_json(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            prompt=prompt,
            protocol=self.protocol,
            response_mode=self.response_mode,
            response_schema=self.response_schema,
            language=self.language,
            retries=retries,
            thinking_effort=(None if self.protocol == PROTOCOL_ANTHROPIC_MESSAGES else self.thinking_effort),
            max_tokens=self.max_tokens,
            validator=lambda parsed: _validate_scores(parsed, self.language),
        )

    def _load_checkpoint(self, checkpoint_file: str, language: str) -> tuple[dict[int, dict], set[int], bool]:
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)

        metadata = checkpoint.get("meta")
        legacy_checkpoint = metadata is None
        if metadata is None:
            if self.input_fingerprint:
                raise RuntimeError(
                    f"Legacy checkpoint {checkpoint_file} has no input fingerprint and cannot be reused "
                    "in a fingerprint-checked execution. Generate a new plan/checkpoint."
                )
            if self.protocol != PROTOCOL_CHAT_COMPLETIONS:
                raise RuntimeError(
                    f"Legacy checkpoint {checkpoint_file} has no protocol metadata and cannot be reused "
                    f"for protocol {self.protocol!r}."
                )
            warnings.warn(
                f"Loading legacy checkpoint without metadata: {checkpoint_file}",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            expected = {
                "model": self.model,
                "protocol": self.protocol,
                "response_mode": self.response_mode,
                "language": language,
                "prompt_version": PROMPT_VERSION,
                "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
            }
            if self.input_fingerprint:
                expected["input_fingerprint"] = self.input_fingerprint
            mismatches = {
                key: (metadata.get(key), value)
                for key, value in expected.items()
                if metadata.get(key) != value
            }
            if mismatches:
                detail = ", ".join(f"{key}={actual!r} (expected {expected_value!r})" for key, (actual, expected_value) in mismatches.items())
                raise RuntimeError(f"Checkpoint metadata mismatch for {checkpoint_file}: {detail}")

        completed_raw = checkpoint.get("completed", {})
        completed: dict[int, dict] = {}
        for raw_index, entry in completed_raw.items():
            index = int(raw_index)
            if isinstance(entry, dict) and "result" in entry and "status" in entry:
                completed[index] = entry
            else:
                # Legacy checkpoint entry: a bare result dict without status.
                completed[index] = {
                    "result": entry,
                    "status": "success" if entry else "empty",
                }
        failed_indices = {int(idx) for idx in checkpoint.get("failed", [])}
        return completed, failed_indices, legacy_checkpoint

    def evaluate_batch(
        self,
        eval_data: list,
        prompt_builder_fn,
        language: str,
        checkpoint_file: str | None = None,
        progress_callback=None,
    ) -> list:
        completed: dict[int, dict] = {}
        failed_indices: set[int] = set()
        legacy_checkpoint = False

        if checkpoint_file and os.path.exists(checkpoint_file):
            completed, failed_indices, legacy_checkpoint = self._load_checkpoint(checkpoint_file, language)

        # Drop checkpoint entries that fall outside the current evaluation set
        # (e.g. a resumed run truncated with --test): stale rows must not be
        # reported or written back into the workbook. An empty eval_data means
        # the caller asked for a pure reload — keep everything then.
        if eval_data:
            eval_indices = {item["index"] for item in eval_data}
            completed = {idx: entry for idx, entry in completed.items() if idx in eval_indices}
            failed_indices = {idx for idx in failed_indices if idx in eval_indices}

        lock = threading.Lock()
        total = len(eval_data)
        done_count = len(completed)

        def _checkpoint():
            if not checkpoint_file:
                return
            with lock:
                data = {
                    "meta": {
                        "model": self.model,
                        "protocol": self.protocol,
                        "response_mode": self.response_mode,
                        "language": language,
                        "prompt_version": PROMPT_VERSION,
                        "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
                        **(
                            {"input_fingerprint": self.input_fingerprint}
                            if self.input_fingerprint
                            else {}
                        ),
                    },
                    "completed": {str(k): v for k, v in completed.items()},
                    "failed": sorted(failed_indices),
                }
                tmp = checkpoint_file + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                os.replace(tmp, checkpoint_file)

        def _save_checkpoint_periodically():
            if checkpoint_file:
                try:
                    _checkpoint()
                except (PermissionError, OSError):
                    pass

        def _worker(item):
            nonlocal done_count
            idx = item["index"]

            with lock:
                if idx in completed or idx in failed_indices:
                    return None

            query = item.get("query", "")
            filename_val = item.get("filename", "Unknown")
            chunks = item.get("chunks", [])
            answer = item.get("answer", "")

            retrieval_status = item.get("retrieval_status", "available")
            if retrieval_status != "available":
                result = _missing_retrieval_result(language, retrieval_status)
                with lock:
                    completed[idx] = {"result": result, "status": "success"}
                    done_count += 1
                    if progress_callback:
                        progress_callback(done_count, total)
                _save_checkpoint_periodically()
                return {"index": idx, "result": result, "status": "success"}

            empty_answer = not answer or not answer.strip() or answer.lower() == "nan"
            if empty_answer:
                result = _empty_answer_result(language)
                with lock:
                    completed[idx] = {"result": result, "status": "empty"}
                    done_count += 1
                    if progress_callback:
                        progress_callback(done_count, total)
                _save_checkpoint_periodically()
                return {"index": idx, "result": result, "status": "empty"}

            prompt = prompt_builder_fn(query, item.get("filenames") or [filename_val], chunks, answer, language)
            try:
                result = self.evaluate_single(prompt)
                _validate_scores(result, language)
                with lock:
                    completed[idx] = {"result": result, "status": "success"}
                    done_count += 1
                    if progress_callback:
                        progress_callback(done_count, total)
                _save_checkpoint_periodically()
                return {"index": idx, "result": result, "status": "success"}
            except Exception:
                with lock:
                    failed_indices.add(idx)
                    done_count += 1
                    if progress_callback:
                        progress_callback(done_count, total)
                _save_checkpoint_periodically()
                return {"index": idx, "result": None, "status": "failed"}

        if legacy_checkpoint:
            _save_checkpoint_periodically()

        pending = [
            item
            for item in eval_data
            if item["index"] not in completed and item["index"] not in failed_indices
        ]

        results = []
        for idx, entry in completed.items():
            entry = entry if isinstance(entry, dict) else {}
            result = entry.get("result")
            status = entry.get("status") or ("success" if result else "empty")
            results.append({"index": idx, "result": result, "status": status})
        for idx in failed_indices:
            results.append({"index": idx, "result": None, "status": "failed"})

        if pending:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(_worker, item): item for item in pending}
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        results.append(result)

        results.sort(key=lambda x: x["index"])
        return results


def _validate_scores(result: dict, language: str):
    meta = _meta(language)
    dim_keys = meta.dims
    score_key = meta.score_key
    reason_key = meta.reason_key
    summary_key = meta.summary_key
    missing = [key for key in dim_keys + [summary_key] if key not in result]
    if missing:
        raise ValueError(f"Evaluation response is missing required fields: {', '.join(missing)}")

    for key in dim_keys:
        dimension = result[key]
        if not isinstance(dimension, dict):
            raise ValueError(f"Evaluation dimension {key!r} must be an object")
        score = dimension.get(score_key)
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise ValueError(f"Evaluation dimension {key!r} has an invalid score")
        reason = dimension.get(reason_key)
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"Evaluation dimension {key!r} has an invalid reason")
        # Fractional scores are tolerated (4.6 -> 5), but the rounded result
        # must land inside 1-5. Out-of-range integers mean the model ignored
        # the scoring rules — raise so the item is retried once and then
        # marked failed instead of silently clamping a bogus score.
        rounded = int(round(score))
        if rounded < 1 or rounded > 5:
            raise ValueError(
                f"Evaluation dimension {key!r} has out-of-range score {score!r}; expected 1-5"
            )
        dimension[score_key] = rounded

    if not isinstance(result[summary_key], str):
        raise ValueError(f"Evaluation summary {summary_key!r} must be a string")


def _missing_retrieval_result(language: str, retrieval_status: str) -> dict:
    meta = _meta(language)
    if language == "zh":
        reason_map = {
            "missing_filename": "找不到文件，按规则所有维度给最低分",
            "missing_search_result": "无法找到召回 chunks，按规则所有维度给最低分",
            "missing_filename_and_search_result": "找不到文件且无法找到召回 chunks，按规则所有维度给最低分",
        }
    else:
        reason_map = {
            "missing_filename": "File not found; minimum score applied to all dimensions by rule",
            "missing_search_result": "Retrieved chunks not found; minimum score applied to all dimensions by rule",
            "missing_filename_and_search_result": "File and retrieved chunks not found; minimum score applied to all dimensions by rule",
        }
    reason_text = reason_map.get(retrieval_status, reason_map.get("missing_search_result"))
    result = {}
    for key in meta.dims:
        result[key] = {meta.score_key: 1, meta.reason_key: reason_text}
    result[meta.summary_key] = reason_text
    result["_retrieval_status"] = retrieval_status
    return result


def _empty_answer_result(language: str) -> dict:
    meta = _meta(language)
    reason_text = meta.empty_reason
    summary_text = meta.empty_summary

    result = {}
    for key in meta.dims:
        result[key] = {meta.score_key: 1, meta.reason_key: reason_text}
    result[meta.summary_key] = summary_text
    return result
