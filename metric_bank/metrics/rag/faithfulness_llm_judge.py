"""RAG faithfulness metric."""

from __future__ import annotations

import json
import re
from statistics import mean
from typing import List

from ...core.types import Example, MetricMetadata, MetricResult

SCORE_PATTERN = re.compile(r"score\s*:\s*([0-1](?:\.\d+)?)", re.IGNORECASE)


class RagFaithfulnessLLMJudge:
    metadata = MetricMetadata(
        id="rag_faithfulness_llm_judge",
        name="RAG Faithfulness (LLM Judge)",
        task_types=["rag_qa"],
        metric_type="llm_judge",
        requires={"reference": False, "context": True},
        cost_estimate="high",
        description="Checks if answers are grounded in retrieved context.",
    )

    def compute(self, examples: List[Example], llm_client=None) -> MetricResult:
        if llm_client is None:
            raise ValueError("LLM client is required for LLM-judge metrics.")
        per_example = []
        scores = []
        for example in examples:
            payload = json.dumps(
                {
                    "candidate": example.output or "",
                    "context": example.context or "",
                }
            )
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Determine if the answer is fully supported by the context. "
                        "Return score:<0 or 1>."
                    ),
                },
                {"role": "user", "content": f"METRIC_BANK_JSON:{payload}"},
            ]
            score = _parse_score(llm_client.chat(messages))
            scores.append(score)
            per_example.append({"example_id": example.id, "score": score})
        return MetricResult(
            metric_id=self.metadata.id,
            score=mean(scores) if scores else 0.0,
            details={"per_example": per_example},
        )


def _parse_score(response: str) -> float:
    match = SCORE_PATTERN.search(response)
    return float(match.group(1)) if match else 0.0


