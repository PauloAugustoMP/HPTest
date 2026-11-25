"""LangChain tool wrappers over the domain layer.

Three thin, read-only tools. The docstrings matter: they are the schema the LLM
sees and uses to decide which tool to call and with what arguments. Guardrail
validation runs at this boundary so the model can never push an unknown GPU
further in.
"""

from __future__ import annotations

from langchain_core.tools import tool

from app import domain, guardrails


@tool
def list_gpus() -> list[dict]:
    """List every GPU in the infrastructure with its model, total quantity and
    memory in GB. Use this to answer "which GPUs exist?"."""
    return domain.list_gpus()


@tool
def get_gpu(gpu: str) -> dict:
    """Get the entry (quantity and memory in GB) for a single GPU model by name.
    Accepts loose names like 'a100' or 'nvidia h100'. Returns an error with the
    list of valid GPUs if the model is unknown."""
    err = guardrails.validate_gpu(gpu)
    if err:
        return err
    return domain.get_gpu(gpu)


@tool
def filter_gpus_by_memory(memory_gb: int, comparison: str = "gte") -> list[dict] | dict:
    """List GPUs whose memory compares to ``memory_gb`` (in GB). ``comparison``
    is 'gte' (>=), 'gt' (>), 'lte' (<=), 'lt' (<), or 'eq' (==). Examples:
    "which GPUs have 80GB?" -> eq 80; "at least 48GB?" -> gte 48;
    "less than 48GB?" -> lt 48."""
    result = domain.filter_gpus_by_memory(memory_gb, comparison=comparison)
    if isinstance(result, list):
        # Wrap with an explicit count so the model lists every match (small
        # models otherwise tend to drop items when summarizing a list).
        return {"count": len(result), "matches": result}
    return result


@tool
def check_availability(
    gpu: str, quantity: int = 1, start: str | None = None, end: str | None = None
) -> dict:
    """Check how many units of a GPU are free. ``quantity`` is how many units
    are requested. Omit ``start``/``end`` to check right now; pass an ISO-8601
    ``start`` to check at a moment, or both ``start`` and ``end`` to check across
    a whole time window. Returns how many are free, the total, and whether the
    request can be satisfied. Convert relative times to ISO-8601 first."""
    err = guardrails.validate_gpu(gpu) or guardrails.validate_quantity(quantity)
    if err:
        return err
    return domain.check_availability(gpu, quantity=quantity, start=start, end=end)


@tool
def list_reservations(gpu: str | None = None, active_only: bool = False) -> dict | list:
    """List current reservations (who booked what, and the start/end times).
    Optionally filter by GPU model and/or to only those active right now. Use
    this to explain why a GPU is busy or when it frees up."""
    result = domain.list_reservations(gpu=gpu, active_only=active_only)
    if isinstance(result, list):
        return {"count": len(result), "reservations": result}
    return result


@tool
def find_next_slot(gpu: str, quantity: int = 1, hours: float = 1) -> dict:
    """Find the earliest time window of ``hours`` hours when at least
    ``quantity`` units of a GPU are free. Use this for questions like "when will
    I have 3 L40S free for 4 hours?"."""
    err = guardrails.validate_gpu(gpu) or guardrails.validate_quantity(quantity)
    if err:
        return err
    return domain.find_next_slot(gpu, quantity=quantity, hours=hours)


ALL_TOOLS = [
    list_gpus,
    get_gpu,
    filter_gpus_by_memory,
    check_availability,
    list_reservations,
    find_next_slot,
]
