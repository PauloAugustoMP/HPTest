"""LLM factory.

The assistant talks to a llama.cpp ``llama-server`` running in Docker. That
server speaks the OpenAI-compatible API and (with ``--jinja`` + a tool-aware
Qwen2.5 GGUF) supports function/tool calling, so we drive it through
``langchain-openai``'s ``ChatOpenAI``. Isolating this here means the provider
can be swapped (Ollama, a hosted API, ...) by editing a single function.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from app.config import settings


def build_llm() -> ChatOpenAI:
    """Return a ChatOpenAI client pointed at the llama.cpp server."""
    return ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        temperature=0,
        timeout=settings.llm_timeout_s,
        max_retries=1,
    )
