"""File format auto-detection, column mapping, and data layout detection."""

import os
import json
import re
import pandas as pd


class FileDetector:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df: pd.DataFrame | None = None
        self.columns: dict = {}
        self.layout: str = ""
        self.language: str = ""
        self.csv_encoding: str | None = None

    def load(self) -> pd.DataFrame:
        ext = os.path.splitext(self.file_path)[1].lower()
        if ext in (".xlsx", ".xls"):
            self.df = pd.read_excel(self.file_path)
        elif ext == ".csv":
            self.df = self._read_csv_with_encoding_fallback()
        elif ext == ".json":
            self.df = pd.read_json(self.file_path)
        elif ext == ".jsonl":
            self.df = pd.read_json(self.file_path, lines=True)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        return self.df

    def _read_csv_with_encoding_fallback(self) -> pd.DataFrame:
        """Read CSV as UTF-8 (BOM tolerated), falling back to GB18030 for
        legacy Chinese encodings. The successful encoding is kept on
        ``self.csv_encoding`` so the plan can show it for confirmation."""
        try:
            self.csv_encoding = "utf-8-sig"
            return pd.read_csv(self.file_path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            self.csv_encoding = "gb18030"
            return pd.read_csv(self.file_path, encoding="gb18030")

    def identify_columns(self) -> dict:
        if self.df is None:
            raise RuntimeError("Call load() first")

        # Field order matters: more specific patterns come first so that a
        # column like 'chunk_score' (a retrieval-score export) is captured by
        # 'score' before the looser 'chunk' pattern of 'chunks' can claim it.
        rules = {
            "query": [r"query", r"\u95ee\u9898", r"question"],
            "answer": [r"answer", r"\u56de\u7b54", r"\u95ee\u7b54\u7ed3\u679c", r"response", r"\u5927\u6a21\u578b\u95ee\u7b54\u7ed3\u679c"],
            "score": [r"score", r"\u5206\u6570", r"\u76f8\u4f3c\u5ea6", r"similarity"],
            "chunks": [r"\u641c\u7d22\u7ed3\u679c", r"chunk", r"\u68c0\u7d22", r"retrieved", r"context"],
            "filename": [r"filename", r"\u6587\u4ef6\u540d", r"source", r"\u6765\u6e90"],
            "reference": [r"\u6807\u6ce8", r"reference", r"gold", r"\u6807\u51c6\u7b54\u6848"],
        }

        result = {}
        for col in self.df.columns:
            col_lower = str(col).lower()
            for field, patterns in rules.items():
                if field in result:
                    continue
                if any(re.search(pattern, col_lower) for pattern in patterns):
                    # First match wins per field AND per column: one column
                    # must never satisfy two fields (e.g. 'chunk_score'
                    # matching both 'chunks' and 'score'), which would later
                    # crash chunk/score parsing.
                    result[field] = col
                    break

        self.columns = result
        return result

    def identify_layout(self) -> str:
        if self.df is None:
            raise RuntimeError("Call load() first")
        if "query" not in self.columns:
            raise RuntimeError("Call identify_columns() first")

        query_col = self.columns["query"]
        nan_ratio = self.df[query_col].isna().mean()
        self.layout = "grouped" if nan_ratio > 0.3 else "row_by_row"
        return self.layout

    def detect_language(self) -> str:
        if self.df is None:
            raise RuntimeError("Call load() first")
        if "query" not in self.columns:
            raise RuntimeError("Call identify_columns() first")

        query_col = self.columns["query"]
        sample_texts = self.df[query_col].dropna().astype(str).head(50)
        combined = " ".join(sample_texts)
        if re.search(r"[\u4e00-\u9fff]", combined):
            self.language = "zh"
            return self.language
        if re.search(r"[\u0600-\u06ff]", combined):
            self.language = "ar"
            return self.language
        if re.search(r"[\u0400-\u04ff]", combined):
            self.language = "ru"
            return self.language

        detected_lang: str | None = None
        try:
            from fast_langdetect import detect
            detected = detect(combined)
            if isinstance(detected, list) and detected:
                detected_lang = detected[0].get("lang") if isinstance(detected[0], dict) else str(detected[0])
            elif isinstance(detected, dict):
                detected_lang = detected.get("lang")
            else:
                detected_lang = str(detected)
        except Exception:
            detected_lang = None

        if detected_lang in {"fr", "ar", "id", "ru"}:
            self.language = detected_lang
            return self.language
        if detected_lang is not None:
            # fast-langdetect succeeded but the code is unsupported (es/de/...):
            # those datasets evaluate with the English standard.
            self.language = "en"
            return self.language

        # fast-langdetect unavailable: legacy French word-list fallback.
        fr_words = {"le", "la", "les", "un", "une", "des", "et", "ou", "est", "sont",
                     "dans", "sur", "avec", "pour", "pas", "que", "qui", "par", "plus",
                     "ce", "en", "au", "aux", "du", "ne", "se", "sa", "son", "ses"}
        words = combined.lower().split()
        total_words = len(words)
        if total_words > 0:
            fr_count = sum(1 for w in words if w in fr_words)
            if fr_count / total_words > 0.08:
                self.language = "fr"
                return self.language

        self.language = "en"
        return self.language




def _is_missing_source_value(value) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return not text or text.casefold() in {"nan", "none", "null", "[]"}


def _build_row_item(
    index: int,
    query_val: str,
    answer_val: str,
    chunks: list,
    filename_values: list,
) -> dict:
    """Build one evaluation item from a row/group's normalized values."""
    filename_missing = not filename_values
    filename_val = filename_values[0] if filename_values else "Unknown"
    seen_files: set = set()
    filenames_dedup: list = []
    for fn in filename_values:
        if fn not in seen_files:
            seen_files.add(fn)
            filenames_dedup.append(fn)
    chunks = sorted(chunks, key=lambda c: c["score"], reverse=True)
    chunks_missing = not chunks
    if filename_missing and chunks_missing:
        retrieval_status = "missing_filename_and_search_result"
    elif filename_missing:
        retrieval_status = "missing_filename"
    elif chunks_missing:
        retrieval_status = "missing_search_result"
    else:
        retrieval_status = "available"

    return {
        "index": index,
        "query": query_val,
        "filename": filename_val,
        "filenames": filenames_dedup if filenames_dedup else ["Unknown"],
        "answer": answer_val,
        "chunks": chunks,
        "retrieval_status": retrieval_status,
    }


def _first_non_empty(values) -> str:
    """Return the first non-empty string value, or '' when all are empty."""
    for value in values:
        if pd.notna(value) and not _is_missing_source_value(value):
            return str(value).strip()
    return ""


def group_and_merge(df: pd.DataFrame, columns: dict, layout: str) -> list:
    query_col = columns.get("query")
    answer_col = columns.get("answer")
    chunks_col = columns.get("chunks")
    score_col = columns.get("score")
    filename_col = columns.get("filename")

    if query_col is None or chunks_col is None:
        raise ValueError("Missing required columns: query or chunks")

    grouped = []

    if layout != "grouped":
        # Layout A (row_by_row): every row is one independent evaluation
        # entry. Rows with an empty query cell cannot be evaluated and are
        # skipped (counted via count_skipped_query_rows for plan warnings).
        for position, (_, row) in enumerate(df.iterrows()):
            query_val = row[query_col]
            if pd.isna(query_val):
                continue
            filename_values = []
            if filename_col and filename_col in df.columns:
                filename_values = [
                    str(value).strip()
                    for value in (row[filename_col],)
                    if not _is_missing_source_value(value)
                ]
            answer_val = ""
            if answer_col and answer_col in df.columns:
                answer_val = _first_non_empty((row[answer_col],))
            chunk_text = row[chunks_col]
            chunks = []
            if not _is_missing_source_value(chunk_text):
                chunk_score = float(row[score_col]) if score_col and score_col in df.columns and pd.notna(row[score_col]) else 1.0
                row_filename = ""
                if filename_col and filename_col in df.columns:
                    raw_fn = row[filename_col]
                    if not _is_missing_source_value(raw_fn):
                        row_filename = str(raw_fn).strip()
                chunks.append({"text": str(chunk_text), "score": chunk_score, "filename": row_filename})
            grouped.append(_build_row_item(
                index=position,
                query_val=str(query_val),
                answer_val=answer_val,
                chunks=chunks,
                filename_values=filename_values,
            ))
        return grouped

    # Layout B (grouped): merge all rows of the same query into one entry.
    df = df.copy()
    df[query_col] = df[query_col].ffill()
    # NOTE: the answer column is NOT forward-filled here — _first_non_empty
    # below picks the first non-empty answer inside each query group, so
    # answers cannot leak across group boundaries via ffill.

    for query_text, group in df.groupby(query_col, sort=False):
        original_index = int(group.index[0])

        query_val = str(query_text) if pd.notna(query_text) else ""
        filename_values = []
        if filename_col and filename_col in df.columns:
            filename_values = [
                str(value).strip()
                for value in group[filename_col].tolist()
                if not _is_missing_source_value(value)
            ]
        answer_val = ""
        if answer_col and answer_col in df.columns:
            answer_val = _first_non_empty(group[answer_col].tolist())

        chunks = []
        for _, row in group.iterrows():
            chunk_text = row[chunks_col]
            if _is_missing_source_value(chunk_text):
                continue
            chunk_score = float(row[score_col]) if score_col and score_col in df.columns and pd.notna(row[score_col]) else 1.0
            row_filename = ""
            if filename_col and filename_col in df.columns:
                raw_fn = row[filename_col]
                if not _is_missing_source_value(raw_fn):
                    row_filename = str(raw_fn).strip()
            chunks.append({"text": str(chunk_text), "score": chunk_score, "filename": row_filename})

        grouped.append(_build_row_item(
            index=original_index,
            query_val=query_val,
            answer_val=answer_val,
            chunks=chunks,
            filename_values=filename_values,
        ))

    return grouped


def count_skipped_query_rows(df: pd.DataFrame, columns: dict, layout: str) -> int:
    """Count rows that cannot be evaluated because the query cell is empty.

    Mirrors the skip rule in ``group_and_merge`` so plan warnings match the
    actual evaluation set:

    * ``row_by_row``: each row whose query is missing/blank is skipped.
    * ``grouped``: rows whose query is still missing after forward-fill
      (leading rows before the first query cell) belong to no group and are
      skipped.
    """
    query_col = columns.get("query")
    if query_col is None or query_col not in df.columns:
        return int(len(df))
    if layout == "grouped":
        filled = df[query_col].ffill()
        return int(filled.isna().sum())
    return int(df[query_col].isna().sum())

