"""Classification accuracy metric."""

from __future__ import annotations

from typing import List

from ...core.types import Example, MetricMetadata, MetricResult


class AccuracyMetric:
    metadata = MetricMetadata(
        id="accuracy",
        name="Accuracy",
        task_types=["classification"],
        metric_type="objective",
        requires={"reference": True, "context": False},
        cost_estimate="low",
        description="Exact match accuracy.",
    )

    def compute(self, examples: List[Example], llm_client=None) -> MetricResult:
        total = 0
        correct = 0
        per_example = []
        for example in examples:
            if example.reference is None:
                continue
            total += 1
            is_correct = example.output == example.reference
            correct += int(is_correct)
            per_example.append(
                {
                    "example_id": example.id,
                    "output": example.output,
                    "reference": example.reference,
                    "correct": is_correct,
                }
            )
        score = correct / total if total else 0.0
        return MetricResult(
            metric_id=self.metadata.id,
            score=score,
            details={"per_example": per_example, "total": total},
        )


