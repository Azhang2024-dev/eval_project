"""Prompt builder: load templates and fill in variables."""

import os
import sys

_prompts_dir = os.path.join(os.path.dirname(__file__), "..", "prompts")
if _prompts_dir not in sys.path:
    sys.path.insert(0, _prompts_dir)


def build_chunks_text(chunks: list, max_chunk_chars: int = 0) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        score = chunk.get("score", 0)
        text = chunk.get("text", "")
        fname = chunk.get("filename") or "unknown"
        if max_chunk_chars > 0 and len(text) > max_chunk_chars:
            text = f"{text[:max_chunk_chars]}\n[truncated: {len(text)} chars]"
        parts.append(f"--- Chunk {i} (file: {fname}, score: {score:.4f}) ---\n{text}")
    return "\n\n".join(parts)


_PROMPT_MAP = {}


def _load_prompt(language: str):
    if language not in _PROMPT_MAP:
        if language == "zh":
            from zh_cn import EVAL_PROMPT as p
        else:
            from en_us import EVAL_PROMPT as p
        _PROMPT_MAP[language] = p
    return _PROMPT_MAP[language]


def build_prompt(
    query: str,
    filenames,
    chunks: list,
    answer: str,
    language: str = "en",
    max_chunks: int = 0,
    max_chunk_chars: int = 0,
) -> str:
    if isinstance(filenames, str):
        filenames = [filenames]
    file_list = [fn for fn in (filenames or []) if fn] or ["Unknown"]
    files_text = "\n".join(f"- {fn}" for fn in file_list)
    eval_prompt = _load_prompt(language)
    if max_chunks > 0:
        chunks = chunks[:max_chunks]
    chunks_text = build_chunks_text(chunks, max_chunk_chars=max_chunk_chars)
    return eval_prompt.format(
        query=query,
        files_text=files_text,
        chunks_text=chunks_text,
        answer=answer,
    )
