"""Simple rule-based metric router."""

from __future__ import annotations

from typing import List, Optional, Sequence

from ..core.registry import MetricRegistry
from ..core.types import Metric, UseCase


class RuleBasedMetricRouter:
    """Deterministic mapping from task type to metrics."""

    DEFAULTS = {
        "classification": ["accuracy", "f1_macro"],
        "chat": ["helpfulness_llm_judge", "latency_ms"],
        "rag_qa": [
            "answer_quality_llm_judge",
            "rag_faithfulness_llm_judge",
            "latency_ms",
            "bleu_precision",
        ],
        "tool_calling": [
            "tool_success_rate",
            "invalid_tool_call_rate",
            "latency_ms",
        ],
    }

    def __init__(self, registry: MetricRegistry) -> None:
        self.registry = registry

    def select_metrics(
        self,
        use_case: UseCase,
        explicit_metrics: Optional[Sequence[str]] = None,
        preset: Optional[str] = None,
    ) -> List[Metric]:
        if explicit_metrics:
            metric_ids = list(explicit_metrics)
        else:
            metric_ids = self.DEFAULTS.get(use_case.task_type, [])
            if not metric_ids:
                raise ValueError(f"No default metrics for task {use_case.task_type}")
            if preset == "lightweight":
                metric_ids = metric_ids[:1]
        return [self.registry.get_metric(metric_id) for metric_id in metric_ids]


