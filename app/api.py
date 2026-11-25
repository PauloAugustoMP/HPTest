"""FastAPI backend.

Endpoints:
- ``POST /chat``    sanitize the message, run the agent, stream the trace
                    (tool calls, tool results, final answer) as NDJSON.
- ``GET  /gpus``    raw inventory (for the UI panel).
- ``GET  /health``  liveness + llama.cpp connectivity.

The NDJSON stream is what makes the assistant auditable: every infrastructure
fact in the final answer is preceded by the exact tool call and tool result
that produced it.
"""

from __future__ import annotations

import json
from typing import Iterator

import requests
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel

from app import domain
from app.config import settings
from app.graph import api_graph

app = FastAPI(title="GPU Assistant API", version="0.1.0")


class ChatRequest(BaseModel):
    thread_id: str
    message: str


def _events(thread_id: str, message: str) -> Iterator[str]:
    """Yield NDJSON lines describing the agent run."""
    from app.guardrails import sanitize_input

    try:
        clean = sanitize_input(message)
    except ValueError as exc:
        yield json.dumps({"type": "error", "message": str(exc)}) + "\n"
        return

    if clean["notes"]:
        yield json.dumps({"type": "sanitized", "notes": clean["notes"]}) + "\n"

    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": [HumanMessage(content=clean["text"])]}

    try:
        for update in api_graph.stream(inputs, config, stream_mode="updates"):
            for node, payload in update.items():
                for msg in payload.get("messages", []):
                    if isinstance(msg, AIMessage):
                        for call in msg.tool_calls or []:
                            yield json.dumps(
                                {"type": "tool_call", "name": call["name"], "args": call["args"]}
                            ) + "\n"
                        if msg.content:
                            yield json.dumps({"type": "final", "content": msg.content}) + "\n"
                    elif isinstance(msg, ToolMessage):
                        yield json.dumps(
                            {"type": "tool_result", "name": msg.name, "result": msg.content}
                        ) + "\n"
    except Exception as exc:  # LLM server unreachable, timeout, etc.
        yield json.dumps(
            {
                "type": "error",
                "message": f"LLM request failed: {exc}. Is the llama.cpp server running "
                f"at {settings.llm_base_url}?",
            }
        ) + "\n"


@app.post("/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        _events(req.thread_id, req.message), media_type="application/x-ndjson"
    )


@app.get("/gpus")
def gpus() -> list[dict]:
    return domain.list_gpus()


@app.get("/health")
def health() -> JSONResponse:
    llm_ok, detail = _check_llm()
    status = "ok" if llm_ok else "degraded"
    return JSONResponse(
        {
            "status": status,
            "llm": {
                "reachable": llm_ok,
                "base_url": settings.llm_base_url,
                "model": settings.llm_model,
                "detail": detail,
            },
        }
    )


def _check_llm() -> tuple[bool, str]:
    """Probe the llama.cpp OpenAI-compatible ``/models`` endpoint."""
    url = settings.llm_base_url.rstrip("/") + "/models"
    try:
        resp = requests.get(url, timeout=3)
        return (resp.ok, f"HTTP {resp.status_code}")
    except requests.RequestException as exc:
        return (False, str(exc))
