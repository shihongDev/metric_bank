"""Core dataclasses and protocols."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol

if TYPE_CHECKING:  # pragma: no cover - avoids runtime import cycle
    from ..llm_adapters.base import LLMClient


@dataclass
class Example:
    """Single evaluation example."""

    id: str
    inputs: Dict[str, Any]
    output: Optional[Any] = None
    reference: Optional[Any] = None
    context: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    latency_ms: Optional[float] = None


@dataclass
class UseCase:
    """Description of a user's scenario."""

    task_type: str
    io_schema: Dict[str, Any] = field(default_factory=dict)
    has_reference: bool = False
    latency_sensitive: bool = False
    safety_critical: bool = False
    cost_sensitive: bool = False
    domain: str = "general"
    notes: Optional[str] = None


@dataclass
class MetricMetadata:
    """Metric description stored in registry."""

    id: str
    name: str
    task_types: List[str]
    metric_type: str
    requires: Dict[str, bool]
    cost_estimate: str
    description: str


@dataclass
class MetricResult:
    """Metric output aggregated across examples."""

    metric_id: str
    score: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Run:
    """Full evaluation run."""

    id: str
    use_case: UseCase
    metrics: List[MetricMetadata]
    metric_results: List[MetricResult]
    examples: List[Example]
    baseline_name: Optional[str] = None
    candidate_name: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "use_case": self.use_case.__dict__,
            "metrics": [m.__dict__ for m in self.metrics],
            "metric_results": [r.__dict__ for r in self.metric_results],
            "examples": [e.__dict__ for e in self.examples],
            "baseline_name": self.baseline_name,
            "candidate_name": self.candidate_name,
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Metric Bank Run `{self.id}`",
            f"Task: **{self.use_case.task_type}**",
        ]
        for result in self.metric_results:
            lines.append(f"- {result.metric_id}: {result.score:.3f}")
        return "\n".join(lines)


class Metric(Protocol):
    """Metric interface."""

    metadata: MetricMetadata

    def compute(
        self,
        examples: List[Example],
        llm_client: Optional["LLMClient"] = None,
    ) -> MetricResult:
        ...

