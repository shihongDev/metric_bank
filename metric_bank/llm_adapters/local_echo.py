"""A deterministic LLM client for local testing."""

from __future__ import annotations

import json
import re
from typing import Dict, List

from .base import LLMClient


class LocalEchoLLMClient:
    """Heuristic client that imitates an LLM judge locally."""

    def chat(self, messages: List[Dict[str, str]]) -> str:
        payload = messages[-1]["content"]
        match = re.search(r"METRIC_BANK_JSON:(.*)", payload, re.DOTALL)
        score = 0.0
        if match:
            try:
                data = json.loads(match.group(1))
                score = self._score_candidate(
                    data.get("candidate", ""),
                    data.get("reference", ""),
                    data.get("question", ""),
                    data.get("context", ""),
                )
            except json.JSONDecodeError:
                score = 0.0
        return f"score:{score:.3f}\nreason:local_echo_overlap"

    @staticmethod
    def _score_candidate(
        candidate: str, reference: str, question, context
    ) -> float:
        question_text = _to_text(question)
        context_text = _to_text(context)
        tokens_candidate = set(candidate.lower().split())
        tokens_reference = set(reference.lower().split())
        if not tokens_reference:
            return 0.0
        overlap = len(tokens_candidate & tokens_reference) / len(tokens_reference)
        question_tokens = question_text.split()
        bonus = 0.1 if question_tokens and question_tokens[0] in candidate.lower() else 0.0
        context_bonus = (
            0.1
            if context_text and all(word in context_text for word in ["rate"])
            else 0.0
        )
        score = min(1.0, max(0.0, overlap + bonus + context_bonus))
        return round(score, 3)


def _to_text(value) -> str:
    if isinstance(value, str):
        return value.lower()
    if isinstance(value, dict):
        return " ".join(str(v) for v in value.values()).lower()
    if isinstance(value, list):
        return " ".join(str(v) for v in value).lower()
    return str(value).lower() if value else ""

