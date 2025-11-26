"""Invalid tool call rate metric."""

from __future__ import annotations

from typing import List

from ...core.types import Example, MetricMetadata, MetricResult


class InvalidToolCallRateMetric:
    metadata = MetricMetadata(
        id="invalid_tool_call_rate",
        name="Invalid Tool Call Rate",
        task_types=["tool_calling"],
        metric_type="objective",
        requires={"reference": False, "context": False},
        cost_estimate="low",
        description="Percentage of tool calls flagged as invalid.",
    )

    def compute(self, examples: List[Example], llm_client=None) -> MetricResult:
        invalid = 0
        total = 0
        for example in examples:
            if "tool_invalid" in example.metadata:
                total += 1
                invalid += int(bool(example.metadata["tool_invalid"]))
        score = 1 - (invalid / total if total else 0.0)
        return MetricResult(
            metric_id=self.metadata.id,
            score=score,
            details={"invalid": invalid, "total": total},
        )


