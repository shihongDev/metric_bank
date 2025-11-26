"""Lightweight BLEU-style precision metric."""

from __future__ import annotations

from collections import Counter
from typing import List

from ...core.types import Example, MetricMetadata, MetricResult


class BleuPrecisionMetric:
    metadata = MetricMetadata(
        id="bleu_precision",
        name="BLEU Precision",
        task_types=["rag_qa", "chat"],
        metric_type="objective",
        requires={"reference": True, "context": False},
        cost_estimate="low",
        description="1-gram BLEU-style precision proxy.",
    )

    def compute(self, examples: List[Example], llm_client=None) -> MetricResult:
        per_example = []
        scores = []
        for example in examples:
            if not example.reference:
                continue
            score = self._score(example.output or "", example.reference)
            scores.append(score)
            per_example.append({"example_id": example.id, "score": score})
        avg_score = sum(scores) / len(scores) if scores else 0.0
        return MetricResult(
            metric_id=self.metadata.id,
            score=avg_score,
            details={"per_example": per_example},
        )

    @staticmethod
    def _score(candidate: str, reference: str) -> float:
        candidate_tokens = candidate.lower().split()
        reference_tokens = reference.lower().split()
        if not candidate_tokens or not reference_tokens:
            return 0.0
        candidate_counts = Counter(candidate_tokens)
        reference_counts = Counter(reference_tokens)
        overlap = sum(
            min(candidate_counts[token], reference_counts[token])
            for token in candidate_counts
        )
        return overlap / len(candidate_tokens)


