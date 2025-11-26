"""Tool success rate metric."""

from __future__ import annotations

from typing import List

from ...core.types import Example, MetricMetadata, MetricResult


class ToolSuccessRateMetric:
    metadata = MetricMetadata(
        id="tool_success_rate",
        name="Tool Success Rate",
        task_types=["tool_calling"],
        metric_type="objective",
        requires={"reference": False, "context": False},
        cost_estimate="low",
        description="Fraction of tool calls flagged as successful in metadata.",
    )

    def compute(self, examples: List[Example], llm_client=None) -> MetricResult:
        successes = 0
        total = 0
        for example in examples:
            if "tool_success" in example.metadata:
                total += 1
                successes += int(bool(example.metadata["tool_success"]))
        score = successes / total if total else 0.0
        return MetricResult(
            metric_id=self.metadata.id,
            score=score,
            details={"successes": successes, "total": total},
        )


