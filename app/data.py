"""Static data: GPU inventory and reservation seed.

The inventory is a plain in-memory list of dicts. Reservations live in a JSON
file (``reservations.json``) as offsets in hours relative to "now", so the demo
always has interesting state (some GPUs busy now, some later) without editing
absolute dates that would go stale. No database is involved.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from app.config import settings

# The ground truth the assistant must consult instead of guessing.
GPUS: list[dict] = [
    {"model": "NVIDIA A100", "quantity": 4, "memory_gb": 80},
    {"model": "NVIDIA L40S", "quantity": 6, "memory_gb": 48},
    {"model": "NVIDIA RTX 4090", "quantity": 3, "memory_gb": 24},
    {"model": "NVIDIA H100", "quantity": 2, "memory_gb": 80},
    {"model": "NVIDIA A40", "quantity": 4, "memory_gb": 48},
]


def load_reservations(now: datetime | None = None, path: str | None = None) -> list[dict]:
    """Load reservations, resolving each ``*_offset_h`` against ``now``.

    Returns dicts with absolute ISO-8601 ``start``/``end`` (half-open
    ``[start, end)``). ``now`` is injectable so callers (and tests) can pin the
    clock. Missing file -> no reservations.
    """
    if now is None:
        now = datetime.now(ZoneInfo(settings.timezone))
    file = Path(path or settings.reservations_path)
    if not file.exists():
        return []

    raw = json.loads(file.read_text())
    reservations: list[dict] = []
    for r in raw:
        start = now + timedelta(hours=float(r["start_offset_h"]))
        end = now + timedelta(hours=float(r["end_offset_h"]))
        reservations.append(
            {
                "gpu_model": r["gpu_model"],
                "quantity": int(r["quantity"]),
                "start": start.isoformat(),
                "end": end.isoformat(),
                "user": r.get("user", "unknown"),
            }
        )
    return reservations
