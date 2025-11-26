"""LLM-judge helpfulness metric."""

from __future__ import annotations

import json
import re
from statistics import mean
from typing import List, Optional

from ...core.types import Example, MetricMetadata, MetricResult


class HelpfulnessLLMJudge:
    metadata = MetricMetadata(
        id="helpfulness_llm_judge",
        name="Helpfulness (LLM Judge)",
        task_types=["chat", "rag_qa"],
        metric_type="llm_judge",
        requires={"reference": True, "context": False},
        cost_estimate="medium",
        description="LLM-based judge that scores helpfulness vs reference.",
    )

    def compute(self, examples: List[Example], llm_client=None) -> MetricResult:
        if llm_client is None:
            raise ValueError("LLM client is required for LLM-judge metrics.")
        per_example = []
        scores: List[float] = []
        for example in examples:
            score = self._score_example(example, llm_client)
            scores.append(score)
            per_example.append(
                {"example_id": example.id, "score": score, "output": example.output}
            )
        return MetricResult(
            metric_id=self.metadata.id,
            score=mean(scores) if scores else 0.0,
            details={"per_example": per_example},
        )

    @staticmethod
    def _score_example(example: Example, llm_client) -> float:
        payload = json.dumps(
            {
                "question": example.inputs,
                "candidate": example.output or "",
                "reference": example.reference or "",
            }
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise evaluator. Output `score:<0-1>` followed by "
                    "a short reason."
                ),
            },
            {
                "role": "user",
                "content": f"METRIC_BANK_JSON:{payload}",
            },
        ]
        response = llm_client.chat(messages)
        return _parse_score(response)


SCORE_PATTERN = re.compile(r"score\s*:\s*([0-1](?:\.\d+)?)", re.IGNORECASE)


def _parse_score(response: str) -> float:
    match = SCORE_PATTERN.search(response)
    if not match:
        return 0.0
    value = float(match.group(1))
    if value > 1:
        value = 1.0
    if value < 0:
        value = 0.0
    return value


