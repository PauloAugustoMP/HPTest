"""Lightweight input guardrails.

Two small layers of defense:

1. ``sanitize_input`` cleans the raw user message before it ever reaches the
   LLM: it strips control characters, collapses whitespace, caps the length and
   neutralizes the most obvious prompt-injection phrases. It never blocks a
   legitimate question -- it just defangs hostile framing.
2. ``validate_gpu`` guards the tool boundary so the model can never push an
   unknown GPU into a lookup. This is the structural half of the "never invent
   data" contract (the domain layer enforces the same).
"""

from __future__ import annotations

import re
import unicodedata

from app.config import settings
from app.domain import _unknown_gpu_error, normalize_gpu_name

# Phrases commonly used to hijack an assistant. We don't reject the message
# (that would be brittle and easy to false-positive); we strip the directive so
# it isn't read as a command and flag the attempt.
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(?:all\s+)?previous\s+instructions", re.I),
    re.compile(r"disregard\s+(?:all\s+)?(?:previous|prior)\s+instructions", re.I),
    re.compile(r"ignore\s+(?:todas\s+)?(?:as\s+)?instru[çc][õo]es\s+anteriores", re.I),
    re.compile(r"(?:reveal|show|print|repeat)\s+(?:your\s+)?system\s+prompt", re.I),
    re.compile(r"you\s+are\s+now\s+(?:a|an|in)\b", re.I),
    re.compile(r"\bdeveloper\s+mode\b", re.I),
]

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize_input(raw: str) -> dict:
    """Clean a raw user message.

    Returns ``{"text": <clean>, "flagged": bool, "notes": [...]}``.
    Raises ``ValueError`` only when the message is empty after cleaning.
    """
    if raw is None:
        raise ValueError("Empty message.")

    notes: list[str] = []
    text = unicodedata.normalize("NFC", str(raw))
    text = _CONTROL_CHARS.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        raise ValueError("Empty message.")

    if len(text) > settings.max_input_chars:
        text = text[: settings.max_input_chars].rstrip()
        notes.append(f"Message truncated to {settings.max_input_chars} characters.")

    flagged = False
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            flagged = True
            text = pattern.sub("[removed]", text)
    if flagged:
        notes.append("Possible prompt-injection phrasing was neutralized.")

    return {"text": text, "flagged": flagged, "notes": notes}


def validate_gpu(gpu: str) -> dict | None:
    """Return an error dict if ``gpu`` is not a known model, else ``None``."""
    if normalize_gpu_name(gpu) is None:
        return _unknown_gpu_error(gpu)
    return None


def validate_quantity(quantity) -> dict | None:
    """Return an error dict if ``quantity`` is not a positive integer, else ``None``."""
    try:
        value = int(quantity)
    except (TypeError, ValueError):
        return {"error": f"Invalid quantity: {quantity!r}. Provide a positive integer."}
    if value < 1:
        return {"error": f"Invalid quantity: {quantity!r}. Provide a positive integer."}
    return None
