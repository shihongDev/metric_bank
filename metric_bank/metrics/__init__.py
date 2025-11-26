"""Metric entrypoints."""

from __future__ import annotations

from typing import List

from ..core.registry import MetricRegistry
from .classification.accuracy import AccuracyMetric
from .classification.f1_macro import F1MacroMetric
from .chat.helpfulness_llm_judge import HelpfulnessLLMJudge
from .rag.answer_quality_llm_judge import AnswerQualityLLMJudge
from .rag.bleu_precision import BleuPrecisionMetric
from .rag.faithfulness_llm_judge import RagFaithfulnessLLMJudge
from .system.latency_ms import LatencyMetric
from .tool_calling.invalid_tool_call_rate import InvalidToolCallRateMetric
from .tool_calling.tool_success_rate import ToolSuccessRateMetric


def load_builtin_metrics(registry: MetricRegistry) -> None:
    metrics = [
        AccuracyMetric(),
        F1MacroMetric(),
        HelpfulnessLLMJudge(),
        AnswerQualityLLMJudge(),
        RagFaithfulnessLLMJudge(),
        LatencyMetric(),
        ToolSuccessRateMetric(),
        InvalidToolCallRateMetric(),
        BleuPrecisionMetric(),
    ]
    for metric in metrics:
        registry.register_metric(metric)


