"""Runtime settings loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_API_BASE_URL = "https://maragate.aibotplatform.com/v1"
API_BASE_URL_ENV = "KB_QA_API_BASE_URL"
API_KEY_ENV = "KB_QA_API_KEY"


class APISettingsError(RuntimeError):
    """Raised when required runtime API settings are missing or invalid."""


@dataclass(frozen=True)
class APISettings:
    base_url: str
    api_key: str | None


def load_api_settings(*, require_key: bool = True) -> APISettings:
    """Load gateway settings without exposing secrets in logs or manifests."""
    base_url = os.getenv(API_BASE_URL_ENV, DEFAULT_API_BASE_URL).strip()
    api_key = os.getenv(API_KEY_ENV)
    if api_key is not None:
        api_key = api_key.strip()

    if not base_url:
        raise APISettingsError(f"{API_BASE_URL_ENV} must not be empty")
    if require_key and not api_key:
        raise APISettingsError(
            f"Missing required environment variable: {API_KEY_ENV}. "
            "Set it before starting a non-plan evaluation."
        )

    return APISettings(base_url=base_url, api_key=api_key or None)
