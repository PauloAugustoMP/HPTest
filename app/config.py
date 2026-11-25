"""Central configuration, read from environment variables.

Everything the app needs to talk to the LLM lives here so the rest of the code
never reads ``os.environ`` directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

try:  # Loading a local .env is a convenience, never a requirement.
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass


def _get(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the runtime configuration."""

    # --- LLM (llama.cpp, OpenAI-compatible server) ---
    llm_base_url: str = _get("LLM_BASE_URL", "http://localhost:8080/v1")
    llm_api_key: str = _get("LLM_API_KEY", "sk-noauth")  # llama.cpp ignores the value
    llm_model: str = _get("LLM_MODEL", "qwen2.5-3b-instruct")
    llm_timeout_s: float = float(_get("LLM_TIMEOUT_S", "60"))

    # --- Domain ---
    timezone: str = _get("APP_TIMEZONE", "America/Sao_Paulo")
    reservations_path: str = _get("RESERVATIONS_PATH", "reservations.json")

    # --- Guardrails ---
    max_input_chars: int = int(_get("MAX_INPUT_CHARS", "2000"))


settings = Settings()
