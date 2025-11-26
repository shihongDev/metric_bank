"""Latency metric."""

from __future__ import annotations

from statistics import mean
from typing import List

from ...core.types import Example, MetricMetadata, MetricResult


class LatencyMetric:
    metadata = MetricMetadata(
        id="latency_ms",
        name="Latency (ms)",
        task_types=["chat", "rag_qa", "classification", "tool_calling"],
        metric_type="system",
        requires={"reference": False, "context": False},
        cost_estimate="low",
        description="Average latency captured by the decorator.",
    )

    def compute(self, examples: List[Example], llm_client=None) -> MetricResult:
        latencies = [
            example.latency_ms for example in examples if example.latency_ms is not None
        ]
        avg_latency = mean(latencies) if latencies else 0.0
        return MetricResult(
            metric_id=self.metadata.id,
            score=max(0.0, 1 - avg_latency / 1000.0) if avg_latency else 1.0,
            details={"avg_latency_ms": avg_latency, "per_example_ms": latencies},
        )

