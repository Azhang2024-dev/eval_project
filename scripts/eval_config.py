"""Headless configuration resolution for the KB QA evaluation CLI."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from scripts.api_client import (
    PROTOCOL_CHAT_COMPLETIONS,
    RESPONSE_MODE_TEXT,
    validate_protocol,
    validate_response_mode,
)


class ConfigError(ValueError):
    """Raised when a CLI/config-file run cannot be resolved safely."""


@dataclass
class ResolvedRunConfig:
    file_path: str
    eval_models: list[str]
    language: str | None
    model_protocols: dict[str, str]
    model_response_modes: dict[str, str]
    output_base: str | None
    test_count: int
    workers: int
    checkpoint_dir: str | None
    thinking: bool
    thinking_effort: str
    review_enabled: bool
    doc_dirs: list[str]
    review_models: list[str]
    threshold: float
    merge_enabled: bool
    merge_output: str | None
    layout: str | None = None
    max_chunks: int = 0
    max_chunk_chars: int = 0
    config_path: str | None = None

    @property
    def all_models(self) -> list[str]:
        return list(dict.fromkeys(self.eval_models + self.review_models))


def parse_model_mapping(
    raw: str | None,
    option_name: str,
    validator: Callable[[str], str],
) -> dict[str, str]:
    """Parse comma-separated MODEL=VALUE CLI mappings."""
    if not raw:
        return {}

    mapping: dict[str, str] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"{option_name} entries must use MODEL=VALUE: {item!r}")
        model, value = item.split("=", 1)
        model = model.strip()
        value = value.strip()
        if not model or not value:
            raise ValueError(f"{option_name} entries must use MODEL=VALUE: {item!r}")
        if model in mapping:
            raise ValueError(f"Duplicate model in {option_name}: {model!r}")
        try:
            mapping[model] = validator(value)
        except ValueError as exc:
            raise ValueError(f"Invalid {option_name} value for {model!r}: {exc}") from exc
    return mapping


def _load_config(config_path: str | None) -> tuple[dict[str, Any], Path | None, str | None]:
    if not config_path:
        return {}, None, None

    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    if not path.is_file():
        raise ConfigError(f"Config path is not a file: {path}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON config {path}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Unable to read config {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError("Config root must be a JSON object")

    allowed = {
        "file",
        "language",
        "models",
        "model_protocols",
        "model_response_modes",
        "output",
        "test",
        "workers",
        "checkpoint_dir",
        "thinking",
        "thinking_effort",
        "review",
        "doc_dirs",
        "review_models",
        "threshold",
        "merge",
        "merge_output",
        "layout",
        "max_chunks",
        "max_chunk_chars",
    }
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigError(f"Unknown config field(s): {', '.join(unknown)}")
    return data, path.parent, str(path)


def _value(args: Any, config: Mapping[str, Any], name: str, default: Any = None) -> Any:
    cli_value = getattr(args, name, None)
    if cli_value is not None:
        return cli_value
    if name in config:
        return config[name]
    return default


def _path_value(
    args: Any,
    config: Mapping[str, Any],
    name: str,
    *,
    config_dir: Path | None,
    must_exist: bool = False,
    directory: bool = False,
) -> str | None:
    cli_value = getattr(args, name, None)
    if cli_value is not None:
        raw = cli_value
        base_dir = Path.cwd()
    elif name in config and config[name] is not None:
        raw = config[name]
        base_dir = config_dir or Path.cwd()
    else:
        return None

    if not isinstance(raw, str) or not raw.strip():
        raise ConfigError(f"{name} must be a non-empty path string")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if must_exist and not path.exists():
        raise ConfigError(f"{name} does not exist: {path}")
    if directory and path.exists() and not path.is_dir():
        raise ConfigError(f"{name} is not a directory: {path}")
    if not directory and path.exists() and path.is_dir():
        raise ConfigError(f"{name} is a directory, expected a file: {path}")
    return str(path)


def _list_value(
    args: Any,
    config: Mapping[str, Any],
    name: str,
    *,
    required: bool = False,
) -> list[str]:
    cli_value = getattr(args, name, None)
    if cli_value is not None:
        if not isinstance(cli_value, str):
            raise ConfigError(f"{name} must be a comma-separated string on the CLI")
        values = [item.strip() for item in cli_value.split(",") if item.strip()]
    elif name in config and config[name] is not None:
        raw = config[name]
        if isinstance(raw, str):
            values = [item.strip() for item in raw.split(",") if item.strip()]
        elif isinstance(raw, list):
            if not all(isinstance(item, str) for item in raw):
                raise ConfigError(f"{name} must contain only strings")
            values = [item.strip() for item in raw if item.strip()]
        else:
            raise ConfigError(f"{name} must be a JSON array of strings")
    else:
        values = []

    if required and not values:
        raise ConfigError(f"Missing required parameter: {name}")
    return values


def _mapping_value(
    args: Any,
    config: Mapping[str, Any],
    name: str,
    option_name: str,
    validator: Callable[[str], str],
) -> dict[str, str]:
    config_mapping: dict[str, str] = {}
    if name in config and config[name] is not None:
        raw = config[name]
        if not isinstance(raw, dict):
            raise ConfigError(f"{name} must be a JSON object")
        for model, value in raw.items():
            if not isinstance(model, str) or not isinstance(value, str):
                raise ConfigError(f"{name} keys and values must be strings")
            try:
                config_mapping[model.strip()] = validator(value.strip())
            except ValueError as exc:
                raise ConfigError(f"Invalid {name} value for {model!r}: {exc}") from exc

    cli_raw = getattr(args, name, None)
    if cli_raw is None:
        return config_mapping

    try:
        cli_mapping = parse_model_mapping(cli_raw, option_name, validator)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    config_mapping.update(cli_mapping)
    return config_mapping


def _bool_value(args: Any, config: Mapping[str, Any], name: str, default: bool) -> bool:
    value = _value(args, config, name, default)
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be true or false")
    return value


def _int_value(args: Any, config: Mapping[str, Any], name: str, default: int) -> int:
    value = _value(args, config, name, default)
    if isinstance(value, bool):
        raise ConfigError(f"{name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be an integer") from exc


def _float_value(args: Any, config: Mapping[str, Any], name: str, default: float) -> float:
    value = _value(args, config, name, default)
    if isinstance(value, bool):
        raise ConfigError(f"{name} must be a number")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a number") from exc


def resolve_run_config(args: Any, default_model_protocols: Mapping[str, str]) -> ResolvedRunConfig:
    """Resolve CLI and JSON config values into one validated headless config."""
    config, config_dir, config_path = _load_config(getattr(args, "config", None))

    file_path = _path_value(
        args,
        config,
        "file",
        config_dir=config_dir,
        must_exist=True,
    )
    if file_path is None:
        raise ConfigError("Missing required parameter: file")

    eval_models = _list_value(args, config, "models", required=True)
    review_enabled = _bool_value(args, config, "review", False)
    if review_enabled:
        doc_dirs_raw = _list_value(args, config, "doc_dirs", required=True)
        review_models = _list_value(args, config, "review_models", required=True)
    else:
        doc_dirs_raw = []
        review_models = []

    doc_dirs: list[str] = []
    if review_enabled:
        for index, raw_dir in enumerate(doc_dirs_raw, 1):
            cli_value = getattr(args, "doc_dirs", None)
            base_dir = Path.cwd() if cli_value is not None else (config_dir or Path.cwd())
            directory = Path(raw_dir).expanduser()
            if not directory.is_absolute():
                directory = base_dir / directory
            directory = directory.resolve()
            if not directory.exists():
                raise ConfigError(f"doc_dirs[{index}] does not exist: {directory}")
            if not directory.is_dir():
                raise ConfigError(f"doc_dirs[{index}] is not a directory: {directory}")
            doc_dirs.append(str(directory))

    language = _value(args, config, "language", None)
    if language is not None and language not in {"zh", "en", "fr", "ar", "id", "ru"}:
        raise ConfigError("language must be one of: zh, en, fr, ar, id, ru")

    layout = _value(args, config, "layout", None)
    if layout is not None:
        if layout not in {"auto", "row_by_row", "grouped"}:
            raise ConfigError("layout must be one of: auto, row_by_row, grouped")
        layout = None if layout == "auto" else layout

    max_chunks = _int_value(args, config, "max_chunks", 0)
    max_chunk_chars = _int_value(args, config, "max_chunk_chars", 0)
    if max_chunks < 0:
        raise ConfigError("max_chunks must be >= 0 (0 = unlimited)")
    if max_chunk_chars < 0:
        raise ConfigError("max_chunk_chars must be >= 0 (0 = unlimited)")

    test_count = _int_value(args, config, "test", 0)
    workers = _int_value(args, config, "workers", 10)
    threshold = _float_value(args, config, "threshold", 3.0)
    if test_count < 0:
        raise ConfigError("test must be >= 0")
    if workers < 1:
        raise ConfigError("workers must be >= 1")

    thinking = _bool_value(args, config, "thinking", False)
    thinking_effort = _value(args, config, "thinking_effort", "high")
    if thinking_effort not in {"low", "medium", "high", "max"}:
        raise ConfigError("thinking_effort must be one of: low, medium, high, max")

    model_protocols = {
        model: default_model_protocols.get(model, PROTOCOL_CHAT_COMPLETIONS)
        for model in list(dict.fromkeys(eval_models + review_models))
    }
    model_protocols.update(
        _mapping_value(
            args,
            config,
            "model_protocols",
            "--model-protocols",
            validate_protocol,
        )
    )
    model_response_modes = {
        model: RESPONSE_MODE_TEXT
        for model in list(dict.fromkeys(eval_models + review_models))
    }
    model_response_modes.update(
        _mapping_value(
            args,
            config,
            "model_response_modes",
            "--model-response-modes",
            validate_response_mode,
        )
    )

    invalid_response_modes = [
        model
        for model in list(dict.fromkeys(eval_models + review_models))
        if model_response_modes.get(model, RESPONSE_MODE_TEXT) != RESPONSE_MODE_TEXT
        and model_protocols.get(model) != "responses"
    ]
    if invalid_response_modes:
        raise ConfigError(
            "json_schema response mode requires the Responses protocol; "
            f"invalid models: {', '.join(invalid_response_modes)}"
        )

    output_base = _path_value(
        args,
        config,
        "output",
        config_dir=config_dir,
    )
    checkpoint_dir = _path_value(
        args,
        config,
        "checkpoint_dir",
        config_dir=config_dir,
        directory=True,
    )
    merge_output = _path_value(
        args,
        config,
        "merge_output",
        config_dir=config_dir,
    )
    merge_enabled = _bool_value(args, config, "merge", False)
    if merge_output:
        merge_enabled = True

    return ResolvedRunConfig(
        file_path=file_path,
        eval_models=eval_models,
        language=language,
        model_protocols=model_protocols,
        model_response_modes=model_response_modes,
        output_base=output_base,
        test_count=test_count,
        workers=workers,
        checkpoint_dir=checkpoint_dir,
        thinking=thinking,
        thinking_effort=thinking_effort,
        review_enabled=review_enabled,
        doc_dirs=doc_dirs,
        review_models=review_models,
        threshold=threshold,
        merge_enabled=merge_enabled,
        merge_output=merge_output,
        layout=layout,
        max_chunks=max_chunks,
        max_chunk_chars=max_chunk_chars,
        config_path=config_path,
    )


def config_to_dict(config: ResolvedRunConfig) -> dict[str, Any]:
    """Return a JSON-serializable resolved configuration snapshot."""
    data = asdict(config)
    data["all_models"] = config.all_models
    return data


PLAN_SCHEMA_VERSION = 1


def file_sha256(path: str) -> str:
    """Calculate a streaming SHA-256 digest for a plan input file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_digest(config: ResolvedRunConfig) -> str:
    """Return a stable digest for a resolved run configuration."""
    payload = json.dumps(
        config_to_dict(config),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def make_plan_manifest(
    manifest: dict[str, Any],
    config: ResolvedRunConfig,
) -> dict[str, Any]:
    """Add versioning and integrity metadata to a preflight manifest."""
    result = dict(manifest)
    result["plan_schema_version"] = PLAN_SCHEMA_VERSION
    result["plan_id"] = uuid.uuid4().hex
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    result["input_fingerprint"] = file_sha256(config.file_path)
    result["config_digest"] = config_digest(config)
    return result


def write_plan(path: str, manifest: dict[str, Any]) -> str:
    """Atomically write a plan manifest and return its absolute path."""
    plan_path = os.path.abspath(os.path.expanduser(path))
    parent = os.path.dirname(plan_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    temporary_path = plan_path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary_path, plan_path)
    return plan_path


def load_plan(path: str) -> dict[str, Any]:
    """Load and validate a plan manifest without contacting the API."""
    plan_path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(plan_path):
        raise ConfigError(f"Plan file not found: {plan_path}")
    try:
        with open(plan_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid plan JSON {plan_path}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Unable to read plan {plan_path}: {exc}") from exc

    if not isinstance(manifest, dict):
        raise ConfigError("Plan root must be a JSON object")
    if manifest.get("plan_schema_version") != PLAN_SCHEMA_VERSION:
        raise ConfigError(
            "Unsupported plan schema version: "
            f"{manifest.get('plan_schema_version')!r}; expected {PLAN_SCHEMA_VERSION}"
        )
    if manifest.get("valid") is not True:
        raise ConfigError("Plan is not marked valid")
    if not manifest.get("input_fingerprint"):
        raise ConfigError("Plan is missing input_fingerprint")
    if not manifest.get("config_digest"):
        raise ConfigError("Plan is missing config_digest")
    if not isinstance(manifest.get("resolved_config"), dict):
        raise ConfigError("Plan is missing resolved_config")
    return manifest


def config_from_dict(data: Mapping[str, Any]) -> ResolvedRunConfig:
    """Reconstruct a resolved configuration from a validated plan snapshot."""
    required = {
        "file_path",
        "eval_models",
        "language",
        "model_protocols",
        "model_response_modes",
        "output_base",
        "test_count",
        "workers",
        "checkpoint_dir",
        "thinking",
        "thinking_effort",
        "review_enabled",
        "doc_dirs",
        "review_models",
        "threshold",
        "merge_enabled",
        "merge_output",
        "layout",
        "max_chunks",
        "max_chunk_chars",
    }
    missing = sorted(required - set(data))
    if missing:
        raise ConfigError(f"Plan resolved_config is missing: {', '.join(missing)}")

    try:
        config = ResolvedRunConfig(
            file_path=str(data["file_path"]),
            eval_models=list(data["eval_models"]),
            language=data["language"],
            model_protocols=dict(data["model_protocols"]),
            model_response_modes=dict(data["model_response_modes"]),
            output_base=data["output_base"],
            test_count=int(data["test_count"]),
            workers=int(data["workers"]),
            checkpoint_dir=data["checkpoint_dir"],
            thinking=bool(data["thinking"]),
            thinking_effort=str(data["thinking_effort"]),
            review_enabled=bool(data["review_enabled"]),
            doc_dirs=list(data["doc_dirs"]),
            review_models=list(data["review_models"]),
            threshold=float(data["threshold"]),
            merge_enabled=bool(data["merge_enabled"]),
            merge_output=data["merge_output"],
            layout=data.get("layout"),
            max_chunks=int(data.get("max_chunks", 0)),
            max_chunk_chars=int(data.get("max_chunk_chars", 0)),
            config_path=data.get("config_path"),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid plan resolved_config: {exc}") from exc

    if not config.file_path or not config.eval_models:
        raise ConfigError("Plan resolved_config must contain file_path and eval_models")
    if config.language not in {"zh", "en", "fr", "ar", "id", "ru"}:
        raise ConfigError("Plan resolved_config has invalid language")
    if config.layout not in {None, "row_by_row", "grouped"}:
        raise ConfigError("Plan resolved_config has invalid layout")
    if config.test_count < 0 or config.workers < 1:
        raise ConfigError("Plan resolved_config has invalid test_count/workers")
    if config.max_chunks < 0 or config.max_chunk_chars < 0:
        raise ConfigError("Plan resolved_config has invalid max_chunks/max_chunk_chars")
    try:
        config.model_protocols = {
            model: validate_protocol(protocol)
            for model, protocol in config.model_protocols.items()
        }
        config.model_response_modes = {
            model: validate_response_mode(response_mode)
            for model, response_mode in config.model_response_modes.items()
        }
    except ValueError as exc:
        raise ConfigError(f"Plan resolved_config has invalid model protocol/mode: {exc}") from exc
    invalid_modes = [
        model
        for model in config.all_models
        if config.model_response_modes.get(model, RESPONSE_MODE_TEXT) != RESPONSE_MODE_TEXT
        and config.model_protocols.get(model) != "responses"
    ]
    if invalid_modes:
        raise ConfigError(
            "Plan resolved_config uses json_schema without Responses protocol for: "
            + ", ".join(invalid_modes)
        )
    return config
