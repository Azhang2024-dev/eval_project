"""Multi-model review module: arbitrate evaluation scores with N independent reviewers."""

import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from prompts.review_prompt import REVIEW_PROMPT_ZH, REVIEW_PROMPT_EN
from scripts.api_client import (
    PROTOCOL_CHAT_COMPLETIONS,
    RESPONSE_MODE_JSON_SCHEMA,
    RESPONSE_MODE_TEXT,
    call_model_json,
    validate_protocol,
    validate_response_mode,
)
from scripts.eval_stats import LANGUAGE_META, weighted_from_score_dict
from scripts.settings import load_api_settings

# Bump when REVIEW_PROMPT_* text or review scoring rules change; review
# checkpoints from older prompts are then ignored.
REVIEW_PROMPT_VERSION = 2


def _meta(language: str):
    return LANGUAGE_META.get(language, LANGUAGE_META["en"])


def _review_dims(language: str) -> list[str]:
    return _meta(language).dims


_ORIGINAL_SCORE_KEY_MAP = {"zh": "原评分", "en": "original_score"}
_SUGGESTED_SCORE_KEY_MAP = {"zh": "建议评分", "en": "suggested_score"}
_MAIN_ISSUES_KEY_MAP = {"zh": "主要问题", "en": "main_issues"}

_DIM_ABBREV = ["RR", "CA", "SC", "IF", "RN"]

_PROMPT_MAP = {"zh": REVIEW_PROMPT_ZH, "en": REVIEW_PROMPT_EN}


def _build_review_json_schema(language: str) -> dict:
    """Build the optional structured-output schema for review responses."""
    language = language if language in LANGUAGE_META else "en"
    dimensions = _review_dims(language)

    if language == "zh":
        original_found_key = "原文是否找到"
        summary_key = "总结"
        reasonable_key = "评分合理"
        main_issues_key = "主要问题"
        note_key = "备注"
        original_score_key = "原评分"
        suggested_score_key = "建议评分"
        reasonable_values = ["合理", "需调整"]
    else:
        original_found_key = "Original Found"
        summary_key = "Summary"
        reasonable_key = "score_reasonable"
        main_issues_key = "main_issues"
        note_key = "note"
        original_score_key = "original_score"
        suggested_score_key = "suggested_score"
        reasonable_values = ["Reasonable", "Needs Adjustment"]

    reason_key = _meta(language).reason_key
    properties = {original_found_key: {"type": "string"}}
    required = [original_found_key]
    for dimension in dimensions:
        properties[dimension] = {
            "type": "object",
            "properties": {
                original_score_key: {"type": "integer", "minimum": 1, "maximum": 5},
                suggested_score_key: {"type": "integer", "minimum": 1, "maximum": 5},
                reason_key: {"type": "string"},
            },
            "required": [original_score_key, suggested_score_key, reason_key],
            "additionalProperties": False,
        }
        required.append(dimension)

    properties[summary_key] = {
        "type": "object",
        "properties": {
            reasonable_key: {"type": "string", "enum": reasonable_values},
            main_issues_key: {"type": "string"},
            note_key: {"type": "string"},
        },
        "required": [reasonable_key, main_issues_key, note_key],
        "additionalProperties": False,
    }
    required.append(summary_key)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }

def _model_short(model: str) -> str:
    """Extract short name from full model path (e.g. 'azure/gpt-5.4' -> 'gpt-5.4')."""
    if "/" in model:
        return model.rsplit("/", 1)[-1]
    return model


def _safe_filename_component(value: str) -> str:
    """Return a model name safe for checkpoint filenames."""
    invalid = '<>:"/\\|?*'
    return ''.join('_' if char in invalid else char for char in value).strip(' .') or 'model'


def _call_api(
    prompt: str,
    model: str,
    retries: int = 3,
    thinking_effort: str | None = None,
    protocol: str = PROTOCOL_CHAT_COMPLETIONS,
    response_mode: str = RESPONSE_MODE_TEXT,
    language: str = "en",
    response_schema: dict | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> dict:
    """Compatibility wrapper around the shared protocol-aware API transport."""
    if base_url is None or api_key is None:
        settings = load_api_settings(require_key=True)
        base_url = settings.base_url
        api_key = settings.api_key
    return call_model_json(
        base_url=base_url,
        api_key=api_key,
        model=model,
        prompt=prompt,
        retries=retries,
        thinking_effort=thinking_effort,
        protocol=protocol,
        response_mode=response_mode,
        language=language,
        response_schema=response_schema,
    )


def _format_current_scores(result: dict, language: str) -> str:
    dim_keys = _review_dims(language)
    score_key = _meta(language).score_key
    reason_key = _meta(language).reason_key
    lines = []
    for dim in dim_keys:
        dim_data = result.get(dim, {})
        s = dim_data.get(score_key, "-")
        r = dim_data.get(reason_key, "-")
        lines.append(f"- {dim}: {s}分 - {r}")
    return "\n".join(lines)


def _format_suggested_scores(parsed: dict, language: str, suggested_score_key: str) -> str:
    """Compact per-dimension suggested scores, e.g. 'RR 4 | CA 5 | SC 3 | IF 4 | RN 5'."""
    if not isinstance(parsed, dict):
        return ""
    dim_keys = _review_dims(language)
    parts = []
    for abbrev, dim in zip(_DIM_ABBREV, dim_keys):
        dim_data = parsed.get(dim)
        if not isinstance(dim_data, dict):
            return ""
        score = dim_data.get(suggested_score_key)
        if score is None:
            return ""
        parts.append(f"{abbrev} {score}")
    return " | ".join(parts)


class ReviewModule:
    def __init__(
        self,
        eval_df: pd.DataFrame,
        results: list,
        doc_dirs: list[str],
        review_models: list[str],
        language: str = "en",
        thinking_effort: str | None = None,
        model_protocols: dict[str, str] | None = None,
        model_response_modes: dict[str, str] | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        eval_data: list | None = None,
        checkpoint_dir: str | None = None,
        input_fingerprint: str | None = None,
        eval_model: str | None = None,
    ):
        self.eval_df = eval_df
        self.results = results
        self.doc_dirs = doc_dirs
        self.review_models = review_models
        self.language = language
        self.thinking_effort = thinking_effort
        self.base_url = base_url
        self.api_key = api_key
        self.input_fingerprint = input_fingerprint
        self.eval_model = eval_model
        self.model_protocols = {
            model: validate_protocol(protocol)
            for model, protocol in (model_protocols or {}).items()
        }
        self.model_response_modes = {
            model: validate_response_mode(response_mode)
            for model, response_mode in (model_response_modes or {}).items()
        }
        self.dim_keys = _review_dims(language)
        self.score_key = _meta(language).score_key
        self.prompt_template = _PROMPT_MAP.get(language, _PROMPT_MAP["en"])
        self.checkpoint_path: str | None = None
        if checkpoint_dir and eval_model:
            safe_model = _safe_filename_component(eval_model)
            self.checkpoint_path = os.path.join(
                checkpoint_dir,
                f"_review_checkpoint_{safe_model}_{language}.json",
            )
        # Serializes checkpoint writes: item threads save concurrently, and an
        # unlocked last-writer could otherwise persist a snapshot that misses
        # entries another thread finished moments earlier.
        self._checkpoint_lock = threading.Lock()
        self._group_files: dict[int, list[str]] = {}
        for item in eval_data or []:
            files = [f for f in (item.get("filenames") or []) if f and f != "Unknown"]
            self._group_files[item.get("index")] = files

    def _doc_dirs_fingerprint(self) -> str:
        """Hash of the document library contents (paths + size + mtime)."""
        entries = []
        for doc_dir in self.doc_dirs:
            if not os.path.isdir(doc_dir):
                continue
            for root, _, files in os.walk(doc_dir):
                for name in sorted(files):
                    path = os.path.join(root, name)
                    try:
                        st = os.stat(path)
                        entries.append((path, st.st_size, int(st.st_mtime)))
                    except OSError:
                        pass
        payload = json.dumps(sorted(entries), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _checkpoint_meta(self) -> dict:
        return {
            "eval_model": self.eval_model,
            "language": self.language,
            "review_models": list(self.review_models),
            "review_prompt_version": REVIEW_PROMPT_VERSION,
            "input_fingerprint": self.input_fingerprint or "",
            "doc_dirs_fingerprint": self._doc_dirs_fingerprint(),
        }

    def _load_review_checkpoint(self) -> dict[int, dict]:
        """Load cached per-item per-model review results.

        Returns ``{index: {model: {"model","error","parsed"}}}``. A stale
        checkpoint (different model set, doc library, prompt version, or input
        fingerprint) is ignored with a warning so a resumed run never mixes
        incompatible review contexts.
        """
        if not self.checkpoint_path or not os.path.exists(self.checkpoint_path):
            return {}
        try:
            with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}
        if data.get("meta") != self._checkpoint_meta():
            print(
                f"  Review checkpoint mismatch; restarting review: "
                f"{os.path.basename(self.checkpoint_path)}"
            )
            return {}
        cached = {}
        for raw_index, model_results in (data.get("reviews") or {}).items():
            try:
                cached[int(raw_index)] = model_results
            except (TypeError, ValueError):
                continue
        return cached

    def _save_review_checkpoint(self, cached: dict[int, dict], index: int, model_results: dict) -> None:
        if not self.checkpoint_path:
            return
        with self._checkpoint_lock:
            cached[index] = model_results
            data = {
                "meta": self._checkpoint_meta(),
                "reviews": {str(k): v for k, v in cached.items()},
            }
            tmp = self.checkpoint_path + ".tmp"
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                os.replace(tmp, self.checkpoint_path)
            except OSError:
                pass

    def _protocol_for_model(self, model: str) -> str:
        return self.model_protocols.get(model, PROTOCOL_CHAT_COMPLETIONS)

    def _response_mode_for_model(self, model: str) -> str:
        return self.model_response_modes.get(model, RESPONSE_MODE_TEXT)

    def filter_low_scores(self, threshold: float = 3.0) -> list[dict]:
        """Return items whose scores need cross-model review.

        Review triggers (minimizing human review volume):
        - any dimension scored 1 (core task failure on that dimension)
        - weighted average below *threshold* (default 3.0, the badcase line)
        """
        items = []
        for r in self.results:
            if r["status"] not in ("success", "empty"):
                continue
            result = r.get("result") or {}
            if result.get("_retrieval_status", "available") != "available":
                continue
            has_one = False
            scores: dict[str, int | float] = {}
            for dim in self.dim_keys:
                dim_data = result.get(dim, {})
                s = dim_data.get(self.score_key)
                if isinstance(s, (int, float)):
                    scores[dim] = s
                    if int(s) == 1:
                        has_one = True
            wa = weighted_from_score_dict(scores, self.language)
            if not has_one and (wa is None or wa >= threshold):
                continue
            idx = r["index"]
            row = self.eval_df.iloc[idx] if idx < len(self.eval_df) else None
            query = ""
            answer = ""
            filename = ""
            if row is not None:
                query = str(row.get("query", "") or row.get("Query", "") or "")
                answer = str(row.get("answer", "") or row.get("大模型问答结果", "") or "")
                filename = str(row.get("filename", "") or row.get("source_file", "") or "")
            if not query and isinstance(result, dict):
                query = result.get("query", "")
            if not answer:
                answer = str(row.get("大模型问答结果", "") or "") if row is not None else ""
            items.append({
                "index": idx,
                "query": query,
                "filename": filename,
                "filenames": self._group_files.get(idx) or ([filename] if filename else []),
                "answer": answer,
                "result": result,
            })
        return items

    def _find_original_doc_single(self, filename: str) -> str | None:
        if not filename or filename in ("Unknown", "nan", ""):
            return None
        name_no_ext = os.path.splitext(filename)[0].lower()
        extensions = [".md", ".txt", ".pdf", ".docx", ".doc", ".html", ".htm"]
        candidates = []
        for doc_dir in self.doc_dirs:
            if not os.path.isdir(doc_dir):
                continue
            for root, _, files in os.walk(doc_dir):
                for f in files:
                    f_lower = f.lower()
                    f_name = os.path.splitext(f)[0].lower()
                    if f_name == name_no_ext:
                        candidates.append(os.path.join(root, f))
                    elif name_no_ext in f_name or f_name in name_no_ext:
                        candidates.append(os.path.join(root, f))
        if not candidates:
            for doc_dir in self.doc_dirs:
                if not os.path.isdir(doc_dir):
                    continue
                for root, _, files in os.walk(doc_dir):
                    for f in files:
                        f_lower = f.lower()
                        f_name = os.path.splitext(f)[0].lower()
                        if any(ext in f_lower for ext in extensions):
                            if len(name_no_ext) > 3 and name_no_ext[:10] in f_name:
                                candidates.append(os.path.join(root, f))
        if candidates:
            return candidates[0]
        return None

    def _read_doc_content(self, filepath: str) -> str:
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".md":
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        elif ext == ".txt":
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        elif ext == ".pdf":
            return self._read_pdf_text(filepath)
        elif ext == ".docx":
            return self._read_docx_text(filepath)
        else:
            return f"[Binary file: {os.path.basename(filepath)}]"

    @staticmethod
    def _read_pdf_text(filepath: str) -> str:
        """Extract PDF text with pymupdf; fall back to a placeholder when the
        file has no text layer (scanned PDFs) or the reader fails."""
        try:
            import fitz  # pymupdf

            document = fitz.open(filepath)
            try:
                text = "\n".join(page.get_text() for page in document)
            finally:
                document.close()
        except Exception:
            return f"[Binary file: {os.path.basename(filepath)}]"
        if text and text.strip():
            return text
        return f"[Binary file: {os.path.basename(filepath)} (no text layer)]"

    @staticmethod
    def _read_docx_text(filepath: str) -> str:
        """Extract DOCX paragraphs via the OOXML package (stdlib-only)."""
        try:
            import zipfile
            from xml.etree import ElementTree

            with zipfile.ZipFile(filepath) as archive:
                xml = archive.read("word/document.xml")
            root = ElementTree.fromstring(xml)
            ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
            paragraphs = []
            for para in root.iter(f"{{{ns}}}p"):
                text = "".join(node.text or "" for node in para.iter(f"{{{ns}}}t"))
                paragraphs.append(text)
            content = "\n".join(paragraphs).strip()
        except Exception:
            return f"[Binary file: {os.path.basename(filepath)}]"
        if content:
            return content
        return f"[Binary file: {os.path.basename(filepath)} (no extractable text)]"

    def find_original_docs(self, filenames: list[str]) -> list[str]:
        found: list[str] = []
        seen: set[str] = set()
        for fn in filenames:
            path = self._find_original_doc_single(fn)
            if path and path not in seen:
                seen.add(path)
                found.append(path)
        return found

    def _read_docs_with_budget(self, paths: list[str], budget: int = 128000) -> str:
        parts: list[str] = []
        remaining = budget
        for p in paths:
            header = f"=== Original file: {os.path.basename(p)} ==="
            if remaining <= 0:
                parts.append(f"{header} [omitted: length budget exhausted]")
                continue
            text = self._read_doc_content(p)
            if len(text) > remaining:
                text = text[:remaining]
            parts.append(f"{header}\n{text}")
            remaining -= len(header) + len(text)
        return "\n\n".join(parts)

    def review_single(self, item: dict, model_name: str) -> dict:
        filenames = item.get("filenames") or ([item.get("filename", "")] if item.get("filename") else [])
        query = item.get("query", "")
        answer = item.get("answer", "")
        result = item.get("result", {})

        doc_paths = self.find_original_docs(filenames)
        if doc_paths:
            original_text = self._read_docs_with_budget(doc_paths, budget=128000)
        else:
            return {
                "model": model_name,
                "error": "no_original_doc",
                "parsed": None,
            }

        current_scores = _format_current_scores(result, self.language)
        files_block = "\n".join(f"- {fn}" for fn in filenames) if filenames else "- Unknown"
        prompt = self.prompt_template.format(
            query=query,
            filenames=files_block,
            original_text=original_text,
            answer=answer,
            current_scores=current_scores,
        )
        try:
            parsed = _call_api(
                prompt,
                model_name,
                thinking_effort=self.thinking_effort,
                protocol=self._protocol_for_model(model_name),
                response_mode=self._response_mode_for_model(model_name),
                language=self.language,
                response_schema=(
                    _build_review_json_schema(self.language)
                    if self._response_mode_for_model(model_name) == RESPONSE_MODE_JSON_SCHEMA
                    else None
                ),
                base_url=self.base_url,
                api_key=self.api_key,
            )
            return {"model": model_name, "error": None, "parsed": parsed}
        except Exception as e:
            return {"model": model_name, "error": str(e), "parsed": None}

    def review_with_models(self, items: list[dict], max_workers: int = 5) -> list[dict]:
        """Review items with every review model, resuming from the review
        checkpoint: only models without a usable cached result are re-called.

        Returns a list of ``{"item": item, "model_results": {model: result}}``
        where every item has a result for every review model.
        """
        cached = self._load_review_checkpoint()
        results = []

        def _review_one(item):
            idx = item["index"]
            model_results = dict(cached.get(idx, {}))
            missing_models = [
                model
                for model in self.review_models
                if model not in model_results or model_results[model].get("error")
            ]
            if missing_models:
                futures = {}
                with ThreadPoolExecutor(max_workers=len(missing_models)) as executor:
                    for model in missing_models:
                        futures[executor.submit(self.review_single, item, model)] = model
                for future in as_completed(futures):
                    model_name = futures[future]
                    model_results[model_name] = future.result()
                self._save_review_checkpoint(cached, idx, model_results)
            return item, model_results

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_item = {executor.submit(_review_one, item): item for item in items}
            for future in as_completed(future_to_item):
                item, model_results = future.result()
                results.append({"item": item, "model_results": model_results})
        results.sort(key=lambda x: x["item"]["index"])
        return results

    def arbitrate(self, model_results: dict) -> dict:
        """Arbitrate N reviewer results. model_results: {model_name: {"model":..., "error":..., "parsed":...}}."""
        errors = {m: r.get("error") for m, r in model_results.items() if r.get("error")}
        if errors:
            if len(errors) == len(model_results):
                detail = "; ".join(f"{m}: {e}" for m, e in errors.items())
                return {"consistency": "all_error", "conclusion": "manual_review",
                        "detail": f"All reviewers errored: {detail}"}
            else:
                detail = "; ".join(f"{m}: {e}" for m, e in errors.items())
                return {"consistency": "partial_error", "conclusion": "manual_review",
                        "detail": f"Partial errors: {detail}"}

        sum_key = "总结" if self.language == "zh" else "Summary"
        reasonable_key = "评分合理" if self.language == "zh" else "score_reasonable"
        note_key = "备注" if self.language == "zh" else "note"

        verdicts = {}
        notes = {}
        for model_name, result in model_results.items():
            parsed = result.get("parsed")
            if parsed is None:
                return {"consistency": "no_parsed", "conclusion": "manual_review",
                        "detail": f"{model_name} returned no parsed result"}
            summary = parsed.get(sum_key, {})
            verdict = summary.get(reasonable_key, "")
            if not verdict:
                # A parseable but malformed review (missing/empty verdict)
                # must not be treated as an agreed adjustment.
                return {"consistency": "malformed", "conclusion": "manual_review",
                        "detail": f"{model_name} returned no verdict in its review"}
            verdicts[model_name] = verdict
            notes[model_name] = summary.get(note_key, "")

        unique_verdicts = set(verdicts.values())
        reasonable_values = ("合理", "Reasonable", "Raisonnable")

        if len(unique_verdicts) == 1:
            verdict = list(unique_verdicts)[0]
            if verdict in reasonable_values:
                return {"consistency": "agree_reasonable", "conclusion": "reasonable",
                        "detail": "All reviewers agree the score is reasonable"}
            else:
                all_notes = "; ".join(f"{m}: {n}" for m, n in notes.items())
                return {"consistency": "agree_adjust", "conclusion": "needs_adjustment",
                        "detail": f"All reviewers agree adjustment is needed: {all_notes}"}
        else:
            verdict_summary = ", ".join(f"{m}: {v}" for m, v in verdicts.items())
            return {"consistency": "disagree", "conclusion": "manual_review",
                    "detail": f"Reviewers disagree: {verdict_summary}"}

    def review_all(self, max_workers: int = 5, threshold: float = 3.0) -> pd.DataFrame:
        items = self.filter_low_scores(threshold=threshold)
        if not items:
            return pd.DataFrame()
        review_results = self.review_with_models(items, max_workers=max_workers)

        sum_key = "总结" if self.language == "zh" else "Summary"
        note_key = "备注" if self.language == "zh" else "note"
        reasonable_key = "评分合理" if self.language == "zh" else "score_reasonable"
        suggested_score_key = _SUGGESTED_SCORE_KEY_MAP.get(self.language, "suggested_score")
        main_issues_key = _MAIN_ISSUES_KEY_MAP.get(self.language, "main_issues")

        rows = []
        for rr in review_results:
            item = rr["item"]
            model_results = rr.get("model_results", {})
            arbitrated = self.arbitrate(model_results)
            row = {
                "行号": item["index"],
                "Query": item["query"],
                "原文文件": ", ".join(item.get("filenames") or [item["filename"]]),
                "RR 原分": item["result"].get(self.dim_keys[0], {}).get(self.score_key, "") if len(self.dim_keys) > 0 else "",
                "CA 原分": item["result"].get(self.dim_keys[1], {}).get(self.score_key, "") if len(self.dim_keys) > 1 else "",
                "SC 原分": item["result"].get(self.dim_keys[2], {}).get(self.score_key, "") if len(self.dim_keys) > 2 else "",
                "IF 原分": item["result"].get(self.dim_keys[3], {}).get(self.score_key, "") if len(self.dim_keys) > 3 else "",
                "RN 原分": item["result"].get(self.dim_keys[4], {}).get(self.score_key, "") if len(self.dim_keys) > 4 else "",
            }

            for model_name in self.review_models:
                result = model_results.get(model_name, {})
                parsed = result.get("parsed") or {}
                summary = parsed.get(sum_key, {})
                short = _model_short(model_name)
                row[f"{short}-结论"] = summary.get(reasonable_key, "N/A")
                row[f"{short}-建议"] = summary.get(note_key, "")
                row[f"{short}-建议分"] = _format_suggested_scores(parsed, self.language, suggested_score_key)
                row[f"{short}-主要问题"] = summary.get(main_issues_key, "")

            row["一致性"] = arbitrated["consistency"]
            row["最终结论"] = arbitrated["conclusion"]
            row["备注"] = arbitrated.get("detail", "")
            rows.append(row)
        return pd.DataFrame(rows)
