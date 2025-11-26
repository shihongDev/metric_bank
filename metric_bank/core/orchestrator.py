"""Evaluation orchestration."""

from __future__ import annotations

import time
import uuid
from typing import Callable, Iterable, List, Optional, Sequence

from .registry import MetricRegistry
from .types import Example, Metric, MetricResult, Run, UseCase


class EvalOrchestrator:
    """Coordinates dataset execution and metric computation."""

    def __init__(
        self,
        registry: MetricRegistry,
        router,
        llm_client_resolver: Callable[[], object],
    ) -> None:
        self.registry = registry
        self.router = router
        self.llm_client_resolver = llm_client_resolver

    def _prepare_metrics(
        self,
        use_case: UseCase,
        explicit_metrics: Optional[Sequence[str]],
        preset: Optional[str],
    ) -> List[Metric]:
        if explicit_metrics:
            return [self.registry.get_metric(metric_id) for metric_id in explicit_metrics]
        return self.router.select_metrics(use_case, preset=preset)

    def run(
        self,
        use_case: UseCase,
        examples: Iterable[Example],
        explicit_metrics: Optional[Sequence[str]] = None,
        preset: Optional[str] = None,
        baseline_name: Optional[str] = None,
        candidate_name: Optional[str] = None,
    ) -> Run:
        examples_list = list(examples)
        if not examples_list:
            raise ValueError("No examples available for evaluation.")
        llm_client = self.llm_client_resolver()
        metrics = self._prepare_metrics(use_case, explicit_metrics, preset)
        metric_results: List[MetricResult] = []
        for metric in metrics:
            start = time.perf_counter()
            result = metric.compute(examples_list, llm_client=llm_client)
            elapsed_ms = (time.perf_counter() - start) * 1000
            result.details.setdefault("compute_latency_ms", elapsed_ms)
            metric_results.append(result)

        return Run(
            id=str(uuid.uuid4()),
            use_case=use_case,
            metrics=[metric.metadata for metric in metrics],
            metric_results=metric_results,
            examples=examples_list,
            baseline_name=baseline_name,
            candidate_name=candidate_name,
        )

