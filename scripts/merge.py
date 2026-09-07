"""CLI for merging per-model evaluation workbooks."""

from __future__ import annotations

import argparse
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from scripts.api_client import validate_protocol, validate_response_mode
from scripts.eval_config import parse_model_mapping
from scripts.result_merger import MergeValidationError, _infer_model_label, merge_evaluation_files


def _parse_file_specs(raw_specs: list[str]) -> list[tuple[str, str]]:
    files = []
    seen = set()
    for raw in raw_specs:
        if "=" in raw:
            model, path = raw.split("=", 1)
            model = model.strip()
            path = path.strip()
        else:
            path = raw.strip()
            model = _infer_model_label(path)
        if not model or not path:
            raise ValueError(f"--file must be MODEL=PATH or PATH: {raw!r}")
        if model in seen:
            raise ValueError(f"Duplicate model label: {model}")
        seen.add(model)
        files.append((model, path))
    if not files:
        raise ValueError("At least one --file is required")
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge multiple KB QA evaluation workbooks")
    parser.add_argument("--file", action="append", required=True, help="Per-model evaluation workbook, preferably MODEL=PATH; repeat for each model")
    parser.add_argument("--output", help="Merged workbook path; defaults beside the first input workbook")
    parser.add_argument("--language", choices=["zh", "en", "fr"], help="Expected language; otherwise detect from score headers")
    parser.add_argument("--model-protocols", help="Comma-separated MODEL=PROTOCOL mappings")
    parser.add_argument("--model-response-modes", help="Comma-separated MODEL=MODE mappings")
    args = parser.parse_args()
    try:
        files = _parse_file_specs(args.file)
        protocols = parse_model_mapping(args.model_protocols, "--model-protocols", validate_protocol)
        response_modes = parse_model_mapping(args.model_response_modes, "--model-response-modes", validate_response_mode)
    except ValueError as exc:
        parser.error(str(exc))

    output = args.output
    if not output:
        first = os.path.abspath(files[0][1])
        root, ext = os.path.splitext(first)
        output = f"{root}_multi_model_merged{ext or '.xlsx'}"
    try:
        saved = merge_evaluation_files(files, output, protocols=protocols, response_modes=response_modes, language=args.language)
    except MergeValidationError as exc:
        print(f"Merge refused: {exc}", file=sys.stderr)
        for issue in exc.issues[:20]:
            print(f"  - {issue}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Merge failed: {exc}", file=sys.stderr)
        return 1
    print(f"Merged {len(files)} model workbooks into: {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
