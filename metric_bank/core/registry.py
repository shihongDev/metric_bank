"""Metric registry for discovery and routing."""

from __future__ import annotations

from typing import Dict, Iterable, List

from .types import Metric, MetricMetadata


class MetricRegistry:
    """Holds metric implementations keyed by metadata id."""

    def __init__(self) -> None:
        self._metrics: Dict[str, Metric] = {}

    def register_metric(self, metric: Metric) -> None:
        if metric.metadata.id in self._metrics:
            raise ValueError(f"Duplicate metric id {metric.metadata.id}")
        self._metrics[metric.metadata.id] = metric

    def get_metric(self, metric_id: str) -> Metric:
        try:
            return self._metrics[metric_id]
        except KeyError as exc:
            raise KeyError(f"Metric '{metric_id}' not found") from exc

    def get_metrics_for_task(self, task_type: str) -> List[Metric]:
        return [
            metric
            for metric in self._metrics.values()
            if task_type in metric.metadata.task_types
        ]

    def all_metadata(self) -> Iterable[MetricMetadata]:
        return [metric.metadata for metric in self._metrics.values()]


