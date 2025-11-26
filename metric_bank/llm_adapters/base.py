"""LLM client protocol."""

from __future__ import annotations

from typing import Dict, List, Protocol


class LLMClient(Protocol):
    """Minimal chat interface used by LLM-based metrics."""

    def chat(self, messages: List[Dict[str, str]]) -> str:
        ...


