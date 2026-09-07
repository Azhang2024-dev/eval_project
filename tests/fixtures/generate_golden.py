"""Generate golden reference artifacts for kb-qa-eval equivalence regression tests.

This script:
1. Reads fixture data (sample_input.csv, results_zh.json, results_en.json)
2. Creates evaluation workbooks via ExcelWriter for zh and en
3. Captures ReportGenerator console output for zh and en
4. Dumps all workbook cells to JSON for each language
5. Creates two model evaluation workbooks (test-model-a, test-model-b) for merge path
6. Merges them via merge_evaluation_files and dumps merged workbook cells
7. Saves the raw .xlsx workbooks for both models

All golden artifacts are written to tests/fixtures/golden/.
"""

import json
import os
import sys
from io import StringIO
from pathlib import Path

# ---------------------------------------------------------------------------
# Project import setup (same pattern as evaluate.py and merge_results.py)
# ---------------------------------------------------------------------------
_script_dir = Path(__file__).resolve().parent
_project_dir = _script_dir.parent.parent  # kb-qa-eval/
if str(_project_dir) not in sys.path:
    sys.path.insert(0, str(_project_dir))

import pandas as pd
from openpyxl import load_workbook

from scripts.excel_writer import ExcelWriter
from scripts.report_generator import ReportGenerator
from scripts.result_merger import merge_evaluation_files

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FIXTURES_DIR = _script_dir
GOLDEN_DIR = FIXTURES_DIR / "golden"
CSV_PATH = FIXTURES_DIR / "sample_input.csv"
RESULTS_ZH_PATH = FIXTURES_DIR / "results_zh.json"
RESULTS_EN_PATH = FIXTURES_DIR / "results_en.json"

os.makedirs(GOLDEN_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Common column mapping (matches what file_detector.FileDetector would detect)
# ---------------------------------------------------------------------------
COLUMNS = {
    "query": "query",
    "filename": "filename",
    "chunks": "搜索结果",
    "score": "score",
    "answer": "大模型问答结果",
    "reference": None,
}
COLUMNS_FOR_WRITER = {
    "query": "query",
    "filename": "filename",
    "chunks": "搜索结果",
    "score": "score",
    "answer": "大模型问答结果",
}


# ---------------------------------------------------------------------------
# Helper: dump workbook cells to a JSON-serializable dict
# ---------------------------------------------------------------------------
def _dump_workbook_cells(wb) -> dict:
    """Dump ALL cells of a workbook to a stable JSON structure.

    For each sheet: record every cell's value and column dimension widths.
    Sorted keys guarantee deterministic JSON output.
    """
    result = {}
    for ws in wb:
        sheet_name = ws.title
        cells = {}
        for row in ws.iter_rows(min_row=1, max_row=ws.max_row, max_col=ws.max_column):
            for cell in row:
                key = f"{cell.column_letter}{cell.row}"
                val = cell.value
                # Convert pd.Timestamp to string for JSON serialization
                if isinstance(val, pd.Timestamp):
                    val = val.isoformat()
                cells[key] = val

        col_dims = {}
        for col_letter, dim in ws.column_dimensions.items():
            col_dims[col_letter] = dim.width

        result[sheet_name] = {
            "cells": dict(sorted(cells.items())),
            "column_dimensions": dict(sorted(col_dims.items())),
        }
    return result


# ---------------------------------------------------------------------------
# Helper: dump workbook by path (open with data_only=True, same as merger)
# ---------------------------------------------------------------------------
def _dump_workbook_by_path(path: str) -> dict:
    wb = load_workbook(path, data_only=True)
    try:
        return _dump_workbook_cells(wb)
    finally:
        wb.close()


# ---------------------------------------------------------------------------
# Load fixture data
# ---------------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
results_zh = json.loads(RESULTS_ZH_PATH.read_text("utf-8"))
results_en = json.loads(RESULTS_EN_PATH.read_text("utf-8"))


# ===================================================================
# 1. ZH evaluation workbook + console output
# ===================================================================
print("=== Generating ZH golden artifacts ===")

writer_zh = ExcelWriter(df, results_zh, COLUMNS_FOR_WRITER, language="zh")
zh_xlsx_path = GOLDEN_DIR / "eval_zh.xlsx"
writer_zh.save(str(zh_xlsx_path))

# Capture ReportGenerator console output
zh_stdout = StringIO()
old_stdout = sys.stdout
sys.stdout = zh_stdout
try:
    ReportGenerator(results_zh, language="zh").print_summary()
finally:
    sys.stdout = old_stdout
zh_console_text = zh_stdout.getvalue()

# Write console output
(GOLDEN_DIR / "console_zh.txt").write_text(zh_console_text, encoding="utf-8")

# Dump workbook cells to JSON
zh_cells = _dump_workbook_by_path(str(zh_xlsx_path))
(GOLDEN_DIR / "eval_zh.cells.json").write_text(
    json.dumps(zh_cells, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(f"  -> eval_zh.xlsx, eval_zh.cells.json, console_zh.txt")

# ===================================================================
# 2. EN evaluation workbook + console output
# ===================================================================
print("=== Generating EN golden artifacts ===")

writer_en = ExcelWriter(df, results_en, COLUMNS_FOR_WRITER, language="en")
en_xlsx_path = GOLDEN_DIR / "eval_en.xlsx"
writer_en.save(str(en_xlsx_path))

en_stdout = StringIO()
old_stdout = sys.stdout
sys.stdout = en_stdout
try:
    ReportGenerator(results_en, language="en").print_summary()
finally:
    sys.stdout = old_stdout
en_console_text = en_stdout.getvalue()

(GOLDEN_DIR / "console_en.txt").write_text(en_console_text, encoding="utf-8")

en_cells = _dump_workbook_by_path(str(en_xlsx_path))
(GOLDEN_DIR / "eval_en.cells.json").write_text(
    json.dumps(en_cells, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(f"  -> eval_en.xlsx, eval_en.cells.json, console_en.txt")

# ===================================================================
# 3. Merge path: two model workbooks (test-model-a, test-model-b) + merge
# ===================================================================
print("=== Generating MERGE golden artifacts ===")

# Create two model evaluation workbooks from the same zh results
# but with different model labels
for model_name in ("test-model-a", "test-model-b"):
    writer = ExcelWriter(df, results_zh, COLUMNS_FOR_WRITER, language="zh")
    model_path = GOLDEN_DIR / f"eval_{model_name}.xlsx"
    writer.save(str(model_path))
    print(f"  -> eval_{model_name}.xlsx")

# Merge the two model workbooks
model_a_path = str(GOLDEN_DIR / "eval_test-model-a.xlsx")
model_b_path = str(GOLDEN_DIR / "eval_test-model-b.xlsx")
merged_path = str(GOLDEN_DIR / "merged.xlsx")

merge_evaluation_files(
    [("test-model-a", model_a_path), ("test-model-b", model_b_path)],
    merged_path,
)

# Dump merged workbook cells
merged_cells = _dump_workbook_by_path(merged_path)
(GOLDEN_DIR / "merged.cells.json").write_text(
    json.dumps(merged_cells, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(f"  -> merged.xlsx, merged.cells.json")

# ===================================================================
# Summary
# ===================================================================
print("\n=== Golden artifacts generated ===")
for f in sorted(GOLDEN_DIR.iterdir()):
    size = f.stat().st_size
    print(f"  {f.name}  ({size:,} bytes)")
