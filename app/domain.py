"""Pure business logic over the in-memory GPU inventory.

These functions are the only source of truth about the infrastructure. They
have no LLM / framework dependency, return plain JSON-serializable values, and
return an ``{"error": ..., "valid_gpus": [...]}`` dict (instead of raising) on
bad input -- which is what lets the assistant say "that GPU does not exist"
rather than invent an answer.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from app import data
from app.config import settings
from app.data import GPUS


# ---------------------------------------------------------------------------
# GPU name normalization
# ---------------------------------------------------------------------------
def _core(model: str) -> str:
    """Canonical compact key for a model, e.g. 'NVIDIA RTX 4090' -> 'rtx4090'."""
    return "".join(c for c in model.lower() if c.isalnum()).removeprefix("nvidia")


_CORES: dict[str, str] = {_core(g["model"]): g["model"] for g in GPUS}


def normalize_gpu_name(raw: str | None) -> str | None:
    """Resolve a free-form GPU name to its canonical inventory model.

    Tolerant to variations like 'a100', 'A-100', 'nvidia h100', '4090'.
    Returns ``None`` when nothing matches or the match is ambiguous.
    """
    if not raw or not str(raw).strip():
        return None
    key = "".join(c for c in str(raw).lower() if c.isalnum()).removeprefix("nvidia")
    if not key:
        return None
    matches = {
        model
        for core, model in _CORES.items()
        if key == core or key in core or core in key
    }
    return next(iter(matches)) if len(matches) == 1 else None


def valid_gpu_names() -> list[str]:
    return [g["model"] for g in GPUS]


def _unknown_gpu_error(raw: str | None) -> dict:
    return {
        "error": f"Unknown GPU: {raw!r}. It is not part of the inventory.",
        "valid_gpus": valid_gpu_names(),
    }


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------
def list_gpus() -> list[dict]:
    """Return the full GPU inventory."""
    return [dict(g) for g in GPUS]


def get_gpu(name: str) -> dict:
    """Return a single GPU's entry, or an error dict if the model is unknown."""
    model = normalize_gpu_name(name)
    if model is None:
        return _unknown_gpu_error(name)
    return next(dict(g) for g in GPUS if g["model"] == model)


# Supported comparisons for the memory filter. Several spellings map to each
# operator so the LLM can use whichever feels natural.
_COMPARISONS = {
    "gte": lambda m, t: m >= t,
    ">=": lambda m, t: m >= t,
    "gt": lambda m, t: m > t,
    ">": lambda m, t: m > t,
    "lte": lambda m, t: m <= t,
    "<=": lambda m, t: m <= t,
    "lt": lambda m, t: m < t,
    "<": lambda m, t: m < t,
    "eq": lambda m, t: m == t,
    "==": lambda m, t: m == t,
}


def filter_gpus_by_memory(memory_gb: int, comparison: str = "gte") -> list[dict] | dict:
    """Return GPUs whose memory compares to ``memory_gb`` via ``comparison``.

    ``comparison`` is one of 'gte' (>=), 'gt' (>), 'lte' (<=), 'lt' (<),
    'eq' (==).
    """
    try:
        threshold = int(memory_gb)
    except (TypeError, ValueError):
        return {"error": f"Invalid memory value: {memory_gb!r}. Provide a number in GB."}
    op = _COMPARISONS.get(str(comparison).lower())
    if op is None:
        return {
            "error": f"Invalid comparison: {comparison!r}. Use 'gte', 'lte' or 'eq'.",
        }
    return [dict(g) for g in GPUS if op(g["memory_gb"], threshold)]


# ---------------------------------------------------------------------------
# Time-based availability
# ---------------------------------------------------------------------------
def _now() -> datetime:
    return datetime.now(ZoneInfo(settings.timezone))


def _parse_dt(value: str | datetime) -> datetime:
    """Parse an ISO-8601 string (or pass a datetime through), timezone-aware."""
    dt = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:  # assume the app timezone for naive inputs
        dt = dt.replace(tzinfo=ZoneInfo(settings.timezone))
    return dt


def _gpu_total(model: str) -> int:
    return next(g["quantity"] for g in GPUS if g["model"] == model)


def _for_gpu(reservations: list[dict], model: str) -> list[dict]:
    return [r for r in reservations if r["gpu_model"] == model]


def _reserved_at(reservations: list[dict], at: datetime) -> int:
    """Units reserved at instant ``at`` (half-open [start, end))."""
    return sum(
        r["quantity"]
        for r in reservations
        if _parse_dt(r["start"]) <= at < _parse_dt(r["end"])
    )


def _peak_reserved(reservations: list[dict], win_start: datetime, win_end: datetime) -> int:
    """Max simultaneously-reserved units within [win_start, win_end).

    Sweep over reservation boundaries clipped to the window. At a shared
    timestamp, releases (-) are applied before acquisitions (+) so a reservation
    ending exactly when another starts does not double-count (half-open).
    """
    events: list[tuple[datetime, int]] = []
    for r in reservations:
        start = max(_parse_dt(r["start"]), win_start)
        end = min(_parse_dt(r["end"]), win_end)
        if start < end:
            events.append((start, r["quantity"]))
            events.append((end, -r["quantity"]))
    events.sort(key=lambda e: (e[0], e[1]))  # negatives (releases) first on ties
    current = peak = 0
    for _t, delta in events:
        current += delta
        peak = max(peak, current)
    return peak


def check_availability(
    gpu: str,
    quantity: int = 1,
    start: str | None = None,
    end: str | None = None,
    now: datetime | None = None,
    reservations: list[dict] | None = None,
) -> dict:
    """How many units of ``gpu`` are free.

    No ``start``/``end`` -> right now. ``start`` only -> at that instant.
    Both -> across the whole window [start, end) (uses peak concurrent usage, so
    "available" means the request holds for the entire window).
    Returns ``{gpu, requested, free, total, available, window}`` or an error.
    """
    model = normalize_gpu_name(gpu)
    if model is None:
        return _unknown_gpu_error(gpu)
    try:
        requested = int(quantity)
    except (TypeError, ValueError):
        return {"error": f"Invalid quantity: {quantity!r}. Provide a positive integer."}
    if requested < 1:
        return {"error": f"Invalid quantity: {quantity!r}. Provide a positive integer."}

    now = now or _now()
    if reservations is None:
        reservations = data.load_reservations(now=now)
    res = _for_gpu(reservations, model)
    total = _gpu_total(model)

    try:
        win_start = _parse_dt(start) if start else now
        win_end = _parse_dt(end) if end else None
    except (ValueError, TypeError):
        return {"error": f"Invalid date/time. Use ISO 8601 (e.g. 2026-01-31T14:00)."}
    if win_end is not None and win_end <= win_start:
        return {"error": "Invalid window: 'end' must be after 'start'."}

    if win_end is None:
        free = total - _reserved_at(res, win_start)
        window = {"at": win_start.isoformat()}
    else:
        free = total - _peak_reserved(res, win_start, win_end)
        window = {"start": win_start.isoformat(), "end": win_end.isoformat()}

    return {
        "gpu": model,
        "requested": requested,
        "free": free,
        "total": total,
        "available": free >= requested,
        "window": window,
    }


def list_reservations(
    gpu: str | None = None,
    active_only: bool = False,
    now: datetime | None = None,
    reservations: list[dict] | None = None,
) -> list[dict] | dict:
    """List reservations, optionally filtered by GPU and/or active right now."""
    now = now or _now()
    if reservations is None:
        reservations = data.load_reservations(now=now)
    rows = list(reservations)
    if gpu is not None:
        model = normalize_gpu_name(gpu)
        if model is None:
            return _unknown_gpu_error(gpu)
        rows = [r for r in rows if r["gpu_model"] == model]
    if active_only:
        rows = [r for r in rows if _parse_dt(r["start"]) <= now < _parse_dt(r["end"])]
    return sorted(rows, key=lambda r: r["start"])


def find_next_slot(
    gpu: str,
    quantity: int = 1,
    hours: float = 1,
    now: datetime | None = None,
    reservations: list[dict] | None = None,
) -> dict:
    """Earliest window of ``hours`` where at least ``quantity`` units are free.

    Returns ``{gpu, requested, hours, start, end, available_now}`` or an error
    if more units are requested than exist.
    """
    model = normalize_gpu_name(gpu)
    if model is None:
        return _unknown_gpu_error(gpu)
    try:
        requested = int(quantity)
        duration_h = float(hours)
    except (TypeError, ValueError):
        return {"error": "Invalid quantity or hours. Provide positive numbers."}
    if requested < 1 or duration_h <= 0:
        return {"error": "Invalid quantity or hours. Provide positive numbers."}

    total = _gpu_total(model)
    if requested > total:
        return {
            "error": f"Only {total} {model} exist, so {requested} can never be free.",
            "total": total,
        }

    now = now or _now()
    if reservations is None:
        reservations = data.load_reservations(now=now)
    res = _for_gpu(reservations, model)
    duration = timedelta(hours=duration_h)

    # Capacity only changes at reservation boundaries, so the earliest feasible
    # start is either now or the end of some reservation.
    candidates = sorted({now} | {_parse_dt(r["end"]) for r in res if _parse_dt(r["end"]) > now})
    for start in candidates:
        if _peak_reserved(res, start, start + duration) <= total - requested:
            return {
                "gpu": model,
                "requested": requested,
                "hours": duration_h,
                "start": start.isoformat(),
                "end": (start + duration).isoformat(),
                "available_now": start == now,
            }
    # Unreachable in practice (after the last reservation ends everything frees),
    # but return a safe fallback just in case.
    return {"error": "No free slot found in the scheduled horizon."}
