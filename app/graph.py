"""LangGraph definition: a single tool-calling agent.

Flow:  START -> agent -> (tools -> agent)* -> END

``agent`` injects a system prompt and calls the LLM bound to the read tools. If
the LLM asked for tool calls we run them in ``tools`` and loop back; otherwise we
stop. A single agent keeps the flow easy to read and explain -- no supervisor or
multiple agents are needed for a read-only Q&A assistant.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from app.config import settings
from app.tools import ALL_TOOLS

SYSTEM_PROMPT = """\
You are an assistant for a team that shares GPU infrastructure. You help users \
find out which GPUs exist, their specs (quantity and memory in GB), and when \
they are available.

Rules:
- Answer ONLY from tool results. Never assume or invent models, quantities, \
memory sizes or availability.
- ALWAYS call a tool before stating any fact about the GPUs.
- If a tool returns an error or finds nothing, say so clearly; for an unknown \
GPU, tell the user which GPUs do exist.
- When a tool returns a list, include EVERY item from it in your answer. Never \
drop, skip or summarize away any item the tool returned.
- Do NOT add facts the tools did not return. No invented times, limits, counts \
or "next availability" unless a tool explicitly provided them.
- Convert relative dates/times ("tomorrow 2pm", "amanhã às 14h", "in 3 hours") \
to ISO 8601 using the current date and time below before calling a tool.
- Answer briefly, in the same language the user wrote in.

Current date and time: {now} ({tz})."""


def _system_message() -> SystemMessage:
    now = datetime.now(ZoneInfo(settings.timezone))
    return SystemMessage(
        content=SYSTEM_PROMPT.format(now=now.isoformat(timespec="minutes"), tz=settings.timezone)
    )


def _route(state: MessagesState) -> str:
    """After the agent: go run tools if it asked for any, else stop."""
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


def build_graph(llm: BaseChatModel | None = None, checkpointer=None):
    """Compile the agent graph. ``llm`` is injectable for tests."""
    if llm is None:
        from app.llm import build_llm

        llm = build_llm()
    model = llm.bind_tools(ALL_TOOLS)

    def agent(state: MessagesState) -> dict:
        messages = [_system_message(), *state["messages"]]
        return {"messages": [model.invoke(messages)]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(ALL_TOOLS))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", _route, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=checkpointer)


# Graph used by the API: in-memory checkpointer gives per-thread_id memory.
api_graph = build_graph(checkpointer=InMemorySaver())
