"""Graph tests using a scripted fake chat model -- no LLM server required.

The fake returns a pre-defined sequence of AIMessages (some with tool_calls),
so we can assert the graph runs the right tools and loops back to a final
answer, all deterministically in CI.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from app.graph import build_graph


class ScriptedChatModel(BaseChatModel):
    """Yields the next scripted AIMessage on each call; ignores bound tools."""

    responses: list[AIMessage]
    _i: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):  # the agent binds tools; we ignore them
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        msg = self.responses[min(self._i, len(self.responses) - 1)]
        self._i += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _run(responses, text="hi"):
    graph = build_graph(llm=ScriptedChatModel(responses=responses))
    return graph.invoke({"messages": [HumanMessage(content=text)]})


def test_read_tool_then_final_answer():
    """Model asks for list_gpus, tool runs, model gives a final answer."""
    scripted = [
        AIMessage(content="", tool_calls=[{"name": "list_gpus", "args": {}, "id": "c1"}]),
        AIMessage(content="We have 5 GPU models."),
    ]
    out = _run(scripted, "Which GPUs exist?")
    tool_msgs = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert "NVIDIA A100" in tool_msgs[0].content
    assert out["messages"][-1].content == "We have 5 GPU models."


def test_filter_by_memory_tool_result_reaches_graph():
    """filter_gpus_by_memory runs with real args and returns the inventory truth."""
    scripted = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "filter_gpus_by_memory",
                    "args": {"memory_gb": 80, "comparison": "eq"},
                    "id": "c1",
                }
            ],
        ),
        AIMessage(content="NVIDIA A100 and NVIDIA H100 have 80GB."),
    ]
    out = _run(scripted, "Which GPUs have 80GB?")
    tool_msgs = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert "NVIDIA A100" in tool_msgs[0].content
    assert "NVIDIA H100" in tool_msgs[0].content
    assert "NVIDIA RTX 4090" not in tool_msgs[0].content


def test_no_tool_calls_ends_immediately():
    """If the model answers without tools, the graph stops at once."""
    out = _run([AIMessage(content="Hello!")], "hi")
    assert out["messages"][-1].content == "Hello!"
    assert not any(isinstance(m, ToolMessage) for m in out["messages"])


def test_unknown_gpu_tool_returns_error_payload():
    """A nonexistent GPU yields a structured error the model can relay."""
    scripted = [
        AIMessage(content="", tool_calls=[{"name": "get_gpu", "args": {"gpu": "B200"}, "id": "c1"}]),
        AIMessage(content="That GPU does not exist."),
    ]
    out = _run(scripted, "Tell me about the B200")
    tool_msgs = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert "error" in tool_msgs[0].content
    assert "NVIDIA A100" in tool_msgs[0].content  # valid_gpus list included
