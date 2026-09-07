"""Equivalence regression tests: compare newly generated products against golden references.

Each test method rebuilds a product from the same fixture data used to generate
the golden artifacts, then compares cell-by-cell or line-by-line.  Any change
in the code paths that alters the output will be caught here.
"""

import json
import os
import sys
import tempfile
from io import StringIO
from pathlib import Path

# ---------------------------------------------------------------------------
# Project import setup (same pattern as evaluate.py and generate_golden.py)
# ---------------------------------------------------------------------------
_script_dir = Path(__file__).resolve().parent
_project_dir = _script_dir.parent  # kb-qa-eval/
if str(_project_dir) not in sys.path:
    sys.path.insert(0, str(_project_dir))

import pandas as pd
from openpyxl import load_workbook
import unittest

from scripts.excel_writer import ExcelWriter
from scripts.report_generator import ReportGenerator
from scripts.result_merger import merge_evaluation_files

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FIXTURES_DIR = _script_dir / "fixtures"
GOLDEN_DIR = FIXTURES_DIR / "golden"
CSV_PATH = FIXTURES_DIR / "sample_input.csv"
RESULTS_ZH_PATH = FIXTURES_DIR / "results_zh.json"
RESULTS_EN_PATH = FIXTURES_DIR / "results_en.json"

# ---------------------------------------------------------------------------
# Common column mapping (matches generate_golden.py)
# ---------------------------------------------------------------------------
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
    """Dump ALL cells of a workbook to a stable dict structure.

    For each sheet: record every cell's value and column dimension widths.
    Sorted keys guarantee deterministic output.
    """
    result = {}
    for ws in wb:
        sheet_name = ws.title
        cells = {}
        for row in ws.iter_rows(min_row=1, max_row=ws.max_row, max_col=ws.max_column):
            for cell in row:
                key = f"{cell.column_letter}{cell.row}"
                val = cell.value
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


def _dump_workbook_by_path(path: str) -> dict:
    """Open a workbook with data_only=True and dump its cells."""
    wb = load_workbook(path, data_only=True)
    try:
        return _dump_workbook_cells(wb)
    finally:
        wb.close()


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------
class TestRefactorEquivalence(unittest.TestCase):
    """Compare newly generated products against golden reference artifacts."""

    maxDiff = None

    @classmethod
    def setUpClass(cls):
        """Load fixture data once for all tests (avoids repeated I/O)."""
        cls.df = pd.read_csv(CSV_PATH)
        cls.results_zh = json.loads(RESULTS_ZH_PATH.read_text("utf-8"))
        cls.results_en = json.loads(RESULTS_EN_PATH.read_text("utf-8"))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_golden_cells(self, stem: str) -> dict:
        """Load a golden cells.json file and return the parsed dict."""
        path = GOLDEN_DIR / f"{stem}.cells.json"
        return json.loads(path.read_text("utf-8"))

    def _assert_sheet_cells_equal(
        self,
        actual_sheet: dict,
        golden_sheet: dict,
        label: str,
    ):
        """Compare cells and column_dimensions for a single sheet."""
        golden_cells = golden_sheet["cells"]
        actual_cells = actual_sheet["cells"]

        # Check every golden cell
        for cell_key in golden_cells:
            expected = golden_cells[cell_key]
            actual_val = actual_cells.get(cell_key)
            # Treat None / NaN / NaT as equivalent
            if expected is None and actual_val is None:
                continue
            if isinstance(expected, float) and isinstance(actual_val, float):
                if pd.isna(expected) and pd.isna(actual_val):
                    continue
            self.assertEqual(
                expected,
                actual_val,
                f"{label} / {cell_key}: expected {expected!r}, got {actual_val!r}",
            )

        # Fail on unexpected extra cells (structural regression)
        extra_cells = set(actual_cells.keys()) - set(golden_cells.keys())
        if extra_cells:
            self.fail(f"{label}: unexpected extra cells: {sorted(extra_cells)}")

        # Compare column dimensions
        self.assertEqual(
            golden_sheet["column_dimensions"],
            actual_sheet["column_dimensions"],
            f"{label}: column_dimensions differ",
        )

    def _assert_workbook_equal(
        self,
        actual_cells: dict,
        golden_cells: dict,
        label: str,
        *,
        skip_sheets: set[str] | None = None,
    ):
        """Compare two full workbook dumps sheet by sheet."""
        skip = skip_sheets or set()

        # Check sheet name parity (excluding skipped sheets)
        golden_sheets = set(golden_cells.keys())
        actual_sheets = set(actual_cells.keys())

        # Golden may have sheets the actual doesn't (if they were skipped)
        missing_in_actual = golden_sheets - actual_sheets - skip
        if missing_in_actual:
            self.fail(f"{label}: golden sheets missing from actual: {sorted(missing_in_actual)}")

        # Actual may have sheets the golden doesn't
        extra_in_actual = actual_sheets - golden_sheets
        if extra_in_actual:
            self.fail(f"{label}: unexpected extra sheets: {sorted(extra_in_actual)}")

        # Compare each sheet
        common_sheets = golden_sheets & actual_sheets
        for sheet_name in sorted(common_sheets):
            if sheet_name in skip:
                continue
            with self.subTest(sheet=sheet_name):
                self._assert_sheet_cells_equal(
                    actual_cells[sheet_name],
                    golden_cells[sheet_name],
                    f"{label} / {sheet_name}",
                )

    def _assert_console_equal(self, actual: str, golden_path: Path, label: str):
        """Compare console output, normalizing \\r\\n to \\n."""
        golden_text = golden_path.read_text("utf-8")
        actual_norm = actual.replace("\r\n", "\n").rstrip()
        golden_norm = golden_text.replace("\r\n", "\n").rstrip()
        self.assertMultiLineEqual(golden_norm, actual_norm, f"{label}: console output differs")

    # ------------------------------------------------------------------
    # Test: ZH evaluation workbook
    # ------------------------------------------------------------------

    def test_zh_eval_workbook(self):
        """Rebuild the zh evaluation workbook and compare against golden cells."""
        with tempfile.TemporaryDirectory() as tmpdir:
            xlsx_path = os.path.join(tmpdir, "eval_zh.xlsx")
            writer = ExcelWriter(self.df, self.results_zh, COLUMNS_FOR_WRITER, language="zh")
            writer.save(xlsx_path)
            actual_cells = _dump_workbook_by_path(xlsx_path)
            golden_cells = self._load_golden_cells("eval_zh")
            self._assert_workbook_equal(actual_cells, golden_cells, "zh_eval")

    # ------------------------------------------------------------------
    # Test: EN evaluation workbook
    # ------------------------------------------------------------------

    def test_en_eval_workbook(self):
        """Rebuild the en evaluation workbook and compare against golden cells."""
        with tempfile.TemporaryDirectory() as tmpdir:
            xlsx_path = os.path.join(tmpdir, "eval_en.xlsx")
            writer = ExcelWriter(self.df, self.results_en, COLUMNS_FOR_WRITER, language="en")
            writer.save(xlsx_path)
            actual_cells = _dump_workbook_by_path(xlsx_path)
            golden_cells = self._load_golden_cells("eval_en")
            self._assert_workbook_equal(actual_cells, golden_cells, "en_eval")

    # ------------------------------------------------------------------
    # Test: ZH console output
    # ------------------------------------------------------------------

    def test_zh_console_output(self):
        """Rebuild zh console output and compare against golden text."""
        zh_stdout = StringIO()
        old_stdout = sys.stdout
        sys.stdout = zh_stdout
        try:
            ReportGenerator(self.results_zh, language="zh").print_summary()
        finally:
            sys.stdout = old_stdout
        actual_text = zh_stdout.getvalue()
        self._assert_console_equal(actual_text, GOLDEN_DIR / "console_zh.txt", "zh_console")

    # ------------------------------------------------------------------
    # Test: EN console output
    # ------------------------------------------------------------------

    def test_en_console_output(self):
        """Rebuild en console output and compare against golden text."""
        en_stdout = StringIO()
        old_stdout = sys.stdout
        sys.stdout = en_stdout
        try:
            ReportGenerator(self.results_en, language="en").print_summary()
        finally:
            sys.stdout = old_stdout
        actual_text = en_stdout.getvalue()
        self._assert_console_equal(actual_text, GOLDEN_DIR / "console_en.txt", "en_console")

    # ------------------------------------------------------------------
    # Test: Merged workbook (two model workbooks → merge)
    # ------------------------------------------------------------------

    def test_merged_workbook(self):
        """Rebuild two model workbooks, merge, compare against golden merged cells.

        The 数据校验 sheet contains absolute file paths and is therefore
        excluded from comparison.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two model workbooks
            model_a_path = os.path.join(tmpdir, "eval_test-model-a.xlsx")
            model_b_path = os.path.join(tmpdir, "eval_test-model-b.xlsx")

            for model_name, path in (
                ("test-model-a", model_a_path),
                ("test-model-b", model_b_path),
            ):
                writer = ExcelWriter(self.df, self.results_zh, COLUMNS_FOR_WRITER, language="zh")
                writer.save(path)

            # Merge
            merged_path = os.path.join(tmpdir, "merged.xlsx")
            merge_evaluation_files(
                [("test-model-a", model_a_path), ("test-model-b", model_b_path)],
                merged_path,
            )

            # Dump and compare
            actual_cells = _dump_workbook_by_path(merged_path)
            golden_cells = self._load_golden_cells("merged")

            self._assert_workbook_equal(
                actual_cells,
                golden_cells,
                "merged",
                # 数据校验 sheet contains absolute paths — skip it
                skip_sheets={"数据校验"},
            )