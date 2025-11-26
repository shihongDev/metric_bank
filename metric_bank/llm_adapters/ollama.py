"""LLM client backed by a local Ollama server."""

from __future__ import annotations

import os
from typing import Dict, List

import requests

from .base import LLMClient


class OllamaLLMClient:
    """Implements the Metric Bank LLMClient interface using Ollama's HTTP API."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("MB_OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("MB_OLLAMA_MODEL", "llama3")
        self.timeout = timeout or float(os.getenv("MB_OLLAMA_TIMEOUT", "120"))

    def chat(self, messages: List[Dict[str, str]]) -> str:
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={"model": self.model, "messages": messages, "stream": False},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"]


