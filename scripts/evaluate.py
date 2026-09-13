"""Headless KB QA evaluation entrypoint.

Workflow:
  1. Generate a preflight plan:
       python evaluate.py --config <config.json> --plan --plan-output <plan.json>
  2. After the Agent/user confirms the manifest, execute only that plan:
       python evaluate.py --execute-plan <plan.json>

Direct execution without a validated plan is disabled. The script does not
open GUI dialogs and does not read interactive stdin.
"""

import argparse
import json
import os
import sys
import time

_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from scripts.file_detector import FileDetector, count_skipped_query_rows, group_and_merge
from scripts.prompt_builder import build_prompt
from scripts.api_client import (
    APIClient,
    PROTOCOL_ANTHROPIC_MESSAGES,
    PROTOCOL_CHAT_COMPLETIONS,
    PROTOCOL_RESPONSES,
    RESPONSE_MODE_TEXT,
)
from scripts.excel_writer import ExcelWriter
from scripts.report_generator import ReportGenerator
from scripts.result_merger import (
    MergeValidationError,
    build_runtime_artifact,
    merge_artifacts,
    write_merged_workbook,
)
from scripts.eval_config import (
    ConfigError,
    ResolvedRunConfig,
    config_digest,
    config_from_dict,
    config_to_dict,
    file_sha256,
    load_plan,
    make_plan_manifest,
    resolve_run_config,
    write_plan,
)
from scripts.settings import APISettingsError, load_api_settings

# The current evaluation configuration uses the Chat Completions adapter for
# all three configured models. Override per model with --model-protocols or
# JSON config if a gateway exposes a different protocol.
DEFAULT_MODEL_PROTOCOLS = {
    "azure/gpt-5.4": PROTOCOL_CHAT_COMPLETIONS,
    "alibaba/qwen3.7-max": PROTOCOL_CHAT_COMPLETIONS,
    "bytedance/deepseek-v4-pro": PROTOCOL_CHAT_COMPLETIONS,
}


def _model_short(model: str) -> str:
    """Extract short name from full model path (e.g. 'azure/gpt-5.4' -> 'gpt-5.4')."""
    if "/" in model:
        return model.rsplit("/", 1)[-1]
    return model


def _safe_filename_component(value: str) -> str:
    """Return a model name safe for checkpoint and output filenames."""
    invalid = '<>:"/\\|?*'
    return ''.join('_' if char in invalid else char for char in value).strip(' .') or 'model'


def _add_optional_bool(parser: argparse.ArgumentParser, name: str, help_text: str) -> None:
    """Add --flag/--no-flag while preserving an explicit None for config merging."""
    dest = name.lstrip("-").replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(name, dest=dest, action="store_true", help=help_text)
    group.add_argument(f"--no-{dest.replace('_', '-')}", dest=dest, action="store_false")
    parser.set_defaults(**{dest: None})


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KB QA Evaluation Tool (headless)")
    parser.add_argument("--config", type=str, help="Optional JSON run configuration")
    parser.add_argument("--plan", action="store_true", help="Validate and print a JSON execution plan without API calls")
    parser.add_argument("--plan-output", type=str, default=None, help="Write the --plan manifest to this JSON path")
    parser.add_argument("--execute-plan", type=str, default=None, help="Execute only a previously generated valid plan JSON")
    parser.add_argument("--file", type=str, help="Input file path (required unless supplied by --config)")
    parser.add_argument(
        "--language",
        type=str,
        choices=["zh", "en", "fr", "ar", "id", "ru"],
        default=None,
        help="Evaluation language (default: auto-detect)",
    )
    parser.add_argument(
        "--layout",
        type=str,
        choices=["auto", "row_by_row", "grouped"],
        default=None,
        help="Data layout override (default: auto-detect)",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Limit chunks per query to the top-N by score (0 = unlimited)",
    )
    parser.add_argument(
        "--max-chunk-chars",
        type=int,
        default=None,
        help="Truncate each chunk to N characters (0 = unlimited)",
    )
    parser.add_argument("--output", type=str, default=None, help="Output file path base (model suffix appended automatically)")
    parser.add_argument("--test", type=int, default=None, help="Test mode: only evaluate first N entries")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (default: 10)")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory for checkpoint files")
    _add_optional_bool(
        parser,
        "--thinking",
        "Enable thinking mode for evaluation and review models",
    )
    parser.add_argument(
        "--thinking-effort",
        type=str,
        choices=["low", "medium", "high", "max"],
        default=None,
        help="Thinking effort level (default: high, requires --thinking)",
    )
    parser.add_argument("--models", type=str, default=None, help="Comma-separated evaluation model names")
    parser.add_argument(
        "--model-protocols",
        type=str,
        default=None,
        help="Comma-separated MODEL=PROTOCOL mappings; protocols: chat_completions,responses,anthropic_messages",
    )
    parser.add_argument(
        "--model-response-modes",
        type=str,
        default=None,
        help="Optional MODEL=MODE mappings; modes: text,json_schema (default: text)",
    )
    _add_optional_bool(
        parser,
        "--review",
        "Enable cross-model review of low-score items",
    )
    parser.add_argument("--doc-dirs", type=str, default=None, help="Comma-separated document library folders (required with --review)")
    parser.add_argument("--review-models", type=str, default=None, help="Comma-separated review model names (required with --review)")
    parser.add_argument("--threshold", type=float, default=None, help="Review trigger: weighted-average threshold (default: 3.0, the badcase line); any dimension scored 1 always triggers")
    _add_optional_bool(
        parser,
        "--merge",
        "Enable cross-model comparison workbook generation",
    )
    parser.add_argument("--merge-output", type=str, default=None, help="Output path for the cross-model comparison workbook")
    return parser


def _prepare_model_paths(
    config: ResolvedRunConfig,
    language: str,
) -> tuple[str, str, list[dict], str]:
    """Resolve output/checkpoint paths without creating files or directories."""
    file_base, file_ext = os.path.splitext(config.file_path)
    output_base = config.output_base or f"{file_base}_evaluated{file_ext}"
    out_base, out_ext = os.path.splitext(output_base)
    checkpoint_dir = config.checkpoint_dir or os.path.dirname(config.file_path) or "."
    checkpoint_dir = os.path.abspath(checkpoint_dir)

    model_paths = []
    for model in config.eval_models:
        # Include the provider in filenames so models with the same short name
        # cannot overwrite or accidentally resemble historical results.
        safe_model_id = _safe_filename_component(model)
        protocol = config.model_protocols.get(model, PROTOCOL_CHAT_COMPLETIONS)
        response_mode = config.model_response_modes.get(model, RESPONSE_MODE_TEXT)
        protocol_checkpoint_file = os.path.join(
            checkpoint_dir,
            f"_checkpoint_{safe_model_id}_{protocol}.json",
        )
        checkpoint_file = protocol_checkpoint_file

        output_suffix = safe_model_id
        if protocol != PROTOCOL_CHAT_COMPLETIONS:
            output_suffix = f"{output_suffix}_{protocol}"
        output_path = f"{out_base}_{output_suffix}{out_ext}"
        model_paths.append(
            {
                "model": model,
                "protocol": protocol,
                "response_mode": response_mode,
                "checkpoint_dir": checkpoint_dir,
                "checkpoint_path": checkpoint_file,
                "output_path": output_path,
            }
        )

    merge_output = config.merge_output or f"{out_base}_multi_model_{language}{out_ext}"
    return out_base, out_ext, model_paths, merge_output


def _build_plan_manifest(
    config: ResolvedRunConfig,
    df,
    columns: dict,
    layout: str,
    language: str,
    eval_data: list,
    model_paths: list[dict],
    merge_output: str,
    skipped_rows: int = 0,
    input_encoding: str | None = None,
) -> dict:
    warnings: list[str] = []
    checkpoint_dir = config.checkpoint_dir or os.path.dirname(config.file_path) or "."
    if not os.path.exists(checkpoint_dir):
        warnings.append(f"checkpoint_dir will be created during execution: {checkpoint_dir}")
    if config.test_count > 0 and config.test_count < len(eval_data):
        warnings.append(f"test mode limits evaluation to {config.test_count} of {len(eval_data)} queries")
    if skipped_rows > 0:
        warnings.append(
            f"{skipped_rows} row(s) with an empty query were skipped and will not be evaluated"
        )
    if "answer" not in columns:
        warnings.append(
            "No answer column detected — every entry will be scored 1 for an empty answer"
        )
    if config.max_chunks > 0:
        max_chunk_count = max((len(item.get("chunks", [])) for item in eval_data), default=0)
        if max_chunk_count > config.max_chunks:
            warnings.append(
                f"max_chunks={config.max_chunks} will truncate the largest query "
                f"({max_chunk_count} chunks) to the top-{config.max_chunks}"
            )

    return {
        "valid": True,
        "file": config.file_path,
        "language": language,
        "rows": int(len(df)),
        "skipped_query_rows": skipped_rows,
        "input_encoding": input_encoding,
        "layout": layout,
        "layout_source": "manual" if config.layout else "auto",
        "unique_queries": len(eval_data),
        "evaluated_query_count": min(config.test_count, len(eval_data)) if config.test_count > 0 else len(eval_data),
        "test_count": config.test_count,
        "columns": columns,
        "models": [
            {
                "model": item["model"],
                "protocol": item["protocol"],
                "response_mode": item["response_mode"],
                "output_path": os.path.abspath(item["output_path"]),
                "checkpoint_path": os.path.abspath(item["checkpoint_path"]),
            }
            for item in model_paths
        ],
        "review": {
            "enabled": config.review_enabled,
            "models": config.review_models,
            "doc_dirs": config.doc_dirs,
            "threshold": config.threshold,
        },
        "merge": {
            "enabled": config.merge_enabled,
            "output_path": os.path.abspath(merge_output) if config.merge_enabled else None,
        },
        "resolved_config": config_to_dict(config),
        "warnings": warnings,
    }


def _load_and_prepare_input(config: ResolvedRunConfig):
    """Load and normalize input data once for both --plan and execution."""
    detector = FileDetector(config.file_path)
    df = detector.load()
    columns = detector.identify_columns()
    layout = config.layout or detector.identify_layout()
    language = config.language or detector.detect_language()
    eval_data = group_and_merge(df, columns, layout)
    skipped_rows = count_skipped_query_rows(df, columns, layout)
    return detector, df, columns, layout, language, eval_data, skipped_rows


def _progress_bar(done: int, total: int, start_time: float):
    pct = done / total if total > 0 else 1
    bar_len = 20
    filled = int(bar_len * pct)
    bar = "\u2588" * filled + "\u2591" * (bar_len - filled)

    elapsed = time.time() - start_time
    if done > 0:
        eta = (elapsed / done) * (total - done)
        eta_str = f"ETA: {eta/60:.0f}m" if eta >= 60 else f"ETA: {eta:.0f}s"
    else:
        eta_str = "ETA: --"

    print(f"\r  [{bar}] {done}/{total} ({pct*100:.0f}%) {eta_str}", end="", flush=True)


def _absolute_path(value: str) -> str:
    return os.path.abspath(os.path.expanduser(value))


def _validate_plan_against_input(
    manifest: dict,
    config: ResolvedRunConfig,
    df,
    columns: dict,
    layout: str,
    language: str,
    all_eval_data: list,
    model_paths: list[dict],
    merge_output: str,
) -> None:
    """Reject execution when the approved plan no longer matches the input/config."""
    actual_input_fingerprint = file_sha256(config.file_path)
    if actual_input_fingerprint != manifest.get("input_fingerprint"):
        raise ConfigError(
            "Input file changed after --plan; generate a new plan before execution"
        )

    if manifest.get("config_digest") != config_digest(config):
        raise ConfigError(
            "Resolved configuration changed after --plan; generate a new plan before execution"
        )
    if manifest.get("columns") != columns:
        raise ConfigError(
            "Detected column mapping changed after --plan; generate a new plan before execution"
        )
    if manifest.get("layout") != layout:
        raise ConfigError(
            "Detected data layout changed after --plan; generate a new plan before execution"
        )
    if manifest.get("language") != language:
        raise ConfigError(
            "Detected language changed after --plan; generate a new plan before execution"
        )
    if manifest.get("unique_queries") != len(all_eval_data):
        raise ConfigError(
            "Query count changed after --plan; generate a new plan before execution"
        )

    expected_count = (
        min(config.test_count, len(all_eval_data))
        if config.test_count > 0
        else len(all_eval_data)
    )
    if manifest.get("evaluated_query_count") != expected_count:
        raise ConfigError(
            "Evaluation count changed after --plan; generate a new plan before execution"
        )

    actual_models = [
        {
            "model": item["model"],
            "protocol": item["protocol"],
            "response_mode": item["response_mode"],
            "output_path": _absolute_path(item["output_path"]),
            "checkpoint_path": _absolute_path(item["checkpoint_path"]),
        }
        for item in model_paths
    ]
    if manifest.get("models") != actual_models:
        raise ConfigError(
            "Model/protocol/output configuration changed after --plan; generate a new plan"
        )

    actual_merge = _absolute_path(merge_output) if config.merge_enabled else None
    planned_merge = (manifest.get("merge") or {}).get("output_path")
    if planned_merge != actual_merge:
        raise ConfigError(
            "Merge output configuration changed after --plan; generate a new plan"
        )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.plan and args.execute_plan:
        parser.error("--plan and --execute-plan cannot be used together")
    if args.plan_output and not args.plan:
        parser.error("--plan-output requires --plan")

    plan_manifest = None
    if args.execute_plan:
        forbidden = [
            "config",
            "file",
            "language",
            "layout",
            "max_chunks",
            "max_chunk_chars",
            "output",
            "test",
            "workers",
            "checkpoint_dir",
            "thinking",
            "thinking_effort",
            "models",
            "model_protocols",
            "model_response_modes",
            "review",
            "doc_dirs",
            "review_models",
            "threshold",
            "merge",
            "merge_output",
        ]
        supplied = [name for name in forbidden if getattr(args, name, None) is not None]
        if supplied:
            parser.error(
                "--execute-plan cannot override resolved parameters; "
                f"remove: {', '.join('--' + name.replace('_', '-') for name in supplied)}"
            )
        try:
            plan_manifest = load_plan(args.execute_plan)
            config = config_from_dict(plan_manifest["resolved_config"])
        except ConfigError as exc:
            parser.error(str(exc))
    else:
        if not args.plan:
            parser.error(
                "Direct execution is disabled. Run --plan --plan-output first, "
                "then execute with --execute-plan <plan.json>."
            )
        try:
            config = resolve_run_config(args, DEFAULT_MODEL_PROTOCOLS)
        except ConfigError as exc:
            parser.error(str(exc))

    try:
        detector, df, columns, layout, language, all_eval_data, skipped_rows = _load_and_prepare_input(config)
    except Exception as exc:
        parser.error(f"Unable to prepare input data: {exc}")

    # Keep the detected language in the resolved snapshot used by --plan output.
    config.language = language
    eval_data = all_eval_data[: config.test_count] if config.test_count > 0 else all_eval_data
    out_base, out_ext, model_paths, merge_output = _prepare_model_paths(config, language)

    if args.plan:
        manifest = make_plan_manifest(
            _build_plan_manifest(
                config,
                df,
                columns,
                layout,
                language,
                all_eval_data,
                model_paths,
                merge_output,
                skipped_rows=skipped_rows,
                input_encoding=getattr(detector, "csv_encoding", None),
            ),
            config,
        )
        if args.plan_output:
            plan_path = _absolute_path(args.plan_output)
            if plan_path == _absolute_path(config.file_path):
                parser.error("--plan-output cannot overwrite the input file")
            manifest["plan_path"] = plan_path
            try:
                write_plan(plan_path, manifest)
            except OSError as exc:
                parser.error(f"Unable to write plan file: {exc}")
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    try:
        _validate_plan_against_input(
            plan_manifest,
            config,
            df,
            columns,
            layout,
            language,
            all_eval_data,
            model_paths,
            merge_output,
        )
        input_fingerprint = file_sha256(config.file_path)
        api_settings = load_api_settings(require_key=True)
    except (ConfigError, APISettingsError) as exc:
        parser.error(str(exc))

    file_path = config.file_path
    print(f"\n  File: {os.path.basename(file_path)}")
    print(f"  Format: {os.path.splitext(file_path)[1]}")
    print(f"  Rows: {len(df)}")

    print("\n  Column mapping:")
    for field, col in columns.items():
        print(f"    {field:<12} -> {col}")

    print(f"\n  Data layout: {layout}")
    print(f"  Language: {language}")
    print(f"  Unique queries: {len(all_eval_data)}")
    if config.test_count > 0:
        print(f"  TEST MODE: only evaluating first {len(eval_data)} entries")

    if config.checkpoint_dir:
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    print(f"\n  Evaluation models: {config.eval_models}")
    if config.review_enabled and config.review_models:
        print(f"  Review models: {config.review_models}")
    print()

    model_artifacts = []
    had_runtime_failures = False
    thinking_effort = config.thinking_effort if config.thinking else None

    for mi, model_info in enumerate(model_paths):
        model = model_info["model"]
        protocol = model_info["protocol"]
        response_mode = model_info["response_mode"]
        checkpoint_file = model_info["checkpoint_path"]
        output_path = model_info["output_path"]

        print(f"\n{'='*60}")
        print(f"  [{mi+1}/{len(model_paths)}] Model: {model}")
        print(f"  Protocol: {protocol}; response mode: {response_mode}")
        print(f"{'='*60}")

        client = APIClient(
            base_url=api_settings.base_url,
            api_key=api_settings.api_key,
            model=model,
            max_workers=config.workers,
            thinking_effort=thinking_effort,
            protocol=protocol,
            response_mode=response_mode,
            language=language,
            input_fingerprint=input_fingerprint,
        )

        print(f"\n  Starting evaluation ({config.workers} parallel workers)...")
        if config.thinking:
            print(f"  Thinking mode: ON (effort: {config.thinking_effort})")
        print(f"  Checkpoint: {os.path.basename(checkpoint_file)}")
        print(f"  Output: {os.path.basename(output_path)}")
        print()

        start_time = time.time()
        results = client.evaluate_batch(
            eval_data=eval_data,
            prompt_builder_fn=lambda q, f, c, a, l: build_prompt(
                q, f, c, a, l,
                max_chunks=config.max_chunks,
                max_chunk_chars=config.max_chunk_chars,
            ),
            language=language,
            checkpoint_file=checkpoint_file,
            progress_callback=lambda d, t, st=start_time: _progress_bar(d, t, st),
        )
        if any(item.get("status") == "failed" for item in results):
            had_runtime_failures = True
        print()

        if config.merge_enabled:
            model_artifacts.append(
                build_runtime_artifact(
                    model=model,
                    source_df=df,
                    columns=columns,
                    eval_data=eval_data,
                    results=results,
                    language=language,
                    protocol=protocol,
                    response_mode=response_mode,
                    source_path=file_path,
                    output_path=output_path,
                    checkpoint_path=checkpoint_file,
                )
            )

        elapsed = time.time() - start_time
        print(f"\n  Evaluation completed in {elapsed:.0f}s")

        writer = ExcelWriter(df, results, columns, language)

        # --- Cross-model review ---
        if config.review_enabled and config.review_models:
            effective_review_models = [m for m in config.review_models if m != model]
            if not effective_review_models:
                print(
                    f"\n  WARNING: No effective review models for {model} "
                    "(all review models are the same as the eval model). Skipping review."
                )
            else:
                print("\n  Starting cross-model review...")
                print(f"  Review models: {effective_review_models}")
                print(f"  Review trigger: any dim == 1 or WA < {config.threshold}")

                from scripts.review_module import ReviewModule

                reviewer = ReviewModule(
                    df,
                    results,
                    config.doc_dirs,
                    effective_review_models,
                    language,
                    thinking_effort=thinking_effort,
                    model_protocols=config.model_protocols,
                    model_response_modes=config.model_response_modes,
                    base_url=api_settings.base_url,
                    api_key=api_settings.api_key,
                    eval_data=eval_data,
                    checkpoint_dir=config.checkpoint_dir,
                    input_fingerprint=input_fingerprint,
                    eval_model=model,
                )
                review_df = reviewer.review_all(
                    max_workers=config.workers,
                    threshold=config.threshold,
                )
                writer.write_review_sheet(review_df)

                if not review_df.empty:
                    reasonable_count = (review_df["最终结论"] == "reasonable").sum()
                    adjust_count = (review_df["最终结论"] == "needs_adjustment").sum()
                    manual_count = (review_df["最终结论"] == "manual_review").sum()
                    print("\n  Review summary:")
                    print(f"    Reviewed items: {len(review_df)}")
                    print(f"    Score reasonable: {reasonable_count}")
                    print(f"    Needs adjustment: {adjust_count}")
                    print(f"    Manual review needed: {manual_count}")
                else:
                    print("\n  No low-score items to review.")

        saved_path = writer.save(output_path)
        print(f"\n  Results saved to: {saved_path}")

        reporter = ReportGenerator(results, language)
        reporter.print_summary()

    if config.merge_enabled:
        try:
            merged_results = merge_artifacts(model_artifacts)
            merged_path = write_merged_workbook(merged_results, merge_output)
            print(f"\n  Multi-model comparison saved to: {merged_path}")
        except MergeValidationError as exc:
            had_runtime_failures = True
            print(f"\n  Multi-model comparison refused: {exc}")
            for issue in exc.issues[:20]:
                print(f"    - {issue}")

    print(f"\n{'='*60}")
    print("  All evaluations completed.")
    print(f"  Evaluated models: {[_model_short(m) for m in config.eval_models]}")
    if config.review_enabled and config.review_models:
        print(f"  Review models: {[_model_short(m) for m in config.review_models]}")
    print(f"{'='*60}")
    return 1 if had_runtime_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
