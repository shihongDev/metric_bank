"""Answer quality judge metric."""

from __future__ import annotations

import json
import re
from statistics import mean
from typing import List

from ...core.types import Example, MetricMetadata, MetricResult

SCORE_PATTERN = re.compile(r"score\s*:\s*([0-1](?:\.\d+)?)", re.IGNORECASE)


class AnswerQualityLLMJudge:
    metadata = MetricMetadata(
        id="answer_quality_llm_judge",
        name="Answer Quality (LLM Judge)",
        task_types=["rag_qa"],
        metric_type="llm_judge",
        requires={"reference": True, "context": False},
        cost_estimate="medium",
        description="LLM judge scoring how well the answer matches the reference.",
    )

    def compute(self, examples: List[Example], llm_client=None) -> MetricResult:
        if llm_client is None:
            raise ValueError("LLM client is required for LLM-judge metrics.")
        scores = []
        per_example = []
        for example in examples:
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
                        "Score factual correctness on [0,1]. Respond with score:<x>."
                    ),
                },
                {"role": "user", "content": f"METRIC_BANK_JSON:{payload}"},
            ]
            response = llm_client.chat(messages)
            score = _parse_score(response)
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


