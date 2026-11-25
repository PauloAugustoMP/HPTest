"""Streamlit frontend for the GPU assistant.

A minimal chat UI that talks to the FastAPI backend. A header shows the LLM
status; each assistant turn carries an expandable "tool trace" so you can see
exactly which tools were called and what they returned -- the visible proof
that no fact was invented. Icons use Streamlit's native Material symbols
(``:material/...:``) rather than emoji.
"""

from __future__ import annotations

import json
import os
import uuid

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="GPU Assistant",
    page_icon=":material/memory:",
    layout="centered",
)

AVATARS = {"user": ":material/person:", "assistant": ":material/smart_toy:"}


def _get_json(path: str):
    try:
        return requests.get(f"{API_URL}{path}", timeout=5).json()
    except requests.RequestException as exc:
        return {"error": str(exc)}


def _stream_chat(thread_id: str, message: str):
    """Yield parsed NDJSON events from the backend."""
    with requests.post(
        f"{API_URL}/chat",
        json={"thread_id": thread_id, "message": message},
        stream=True,
        timeout=120,
    ) as resp:
        for line in resp.iter_lines(decode_unicode=True):
            if line:
                yield json.loads(line)


def _render_trace(trace: list[dict] | None) -> None:
    """Render an expandable tool trace (calls, results, guardrail notes)."""
    if not trace:
        return
    with st.expander(f"Tool trace ({len(trace)})", icon=":material/build:"):
        for ev in trace:
            if ev["type"] == "tool_call":
                st.markdown(f"**{ev['name']}**")
                st.code(json.dumps(ev["args"], ensure_ascii=False), language="json")
            elif ev["type"] == "tool_result":
                st.caption(f"result · {ev['name']}")
                st.code(ev["result"], language="json")
            elif ev["type"] == "sanitized":
                st.info(" · ".join(ev["notes"]), icon=":material/shield:")


# --- session state -----------------------------------------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history = []  # list of {role, content, trace}


# --- header ------------------------------------------------------------------
title_col, button_col = st.columns([4, 1], vertical_alignment="center")
with title_col:
    st.title("GPU Assistant")
with button_col:
    if st.button("New chat", icon=":material/add:", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.history = []
        st.rerun()

st.caption("Ask which GPUs exist, how much memory they have, and when they are free — in PT or EN.")

health = _get_json("/health")
if health.get("status") == "ok":
    st.badge(f"LLM · {health['llm']['model']}", icon=":material/check_circle:", color="green")
else:
    st.badge("LLM offline", icon=":material/error:", color="red")
    st.caption("Start the llama.cpp server with `docker compose up`.")

st.divider()

# --- conversation history ----------------------------------------------------
for turn in st.session_state.history:
    with st.chat_message(turn["role"], avatar=AVATARS[turn["role"]]):
        st.markdown(turn["content"])
        _render_trace(turn.get("trace"))

# The chat input lives at the top level so Streamlit docks it at the bottom.
prompt = st.chat_input("Ask about the GPUs…")
if prompt:
    st.session_state.history.append({"role": "user", "content": prompt, "trace": []})
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(prompt)

    trace: list[dict] = []
    answer_parts: list[str] = []
    error: str | None = None
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        status = st.status("Thinking…", expanded=False)
        try:
            for ev in _stream_chat(st.session_state.thread_id, prompt):
                if ev["type"] in ("tool_call", "tool_result", "sanitized"):
                    trace.append(ev)
                    if ev["type"] == "tool_call":
                        status.update(label=f"Running {ev['name']}…")
                elif ev["type"] == "final":
                    answer_parts.append(ev["content"])
                elif ev["type"] == "error":
                    error = ev["message"]
        except requests.RequestException as exc:
            error = f"Could not reach the API at {API_URL}: {exc}"
        status.update(label="Done", state="complete")

        content = error or ("\n\n".join(answer_parts) or "_(no answer)_")
        st.markdown(content)
        _render_trace(trace)

    st.session_state.history.append(
        {"role": "assistant", "content": content, "trace": trace}
    )
