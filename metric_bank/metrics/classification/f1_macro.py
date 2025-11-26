"""Macro F1 metric."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import List

from ...core.types import Example, MetricMetadata, MetricResult


class F1MacroMetric:
    metadata = MetricMetadata(
        id="f1_macro",
        name="F1 Macro",
        task_types=["classification"],
        metric_type="objective",
        requires={"reference": True, "context": False},
        cost_estimate="low",
        description="Unweighted macro-average F1 score.",
    )

    def compute(self, examples: List[Example], llm_client=None) -> MetricResult:
        labels = set()
        counts = defaultdict(lambda: Counter(tp=0, fp=0, fn=0))
        for example in examples:
            if example.reference is None:
                continue
            reference_label = example.reference
            predicted_label = example.output if example.output is not None else "__missing__"
            labels.add(reference_label)
            labels.add(predicted_label)
            if predicted_label == reference_label:
                counts[reference_label]["tp"] += 1
            else:
                counts[predicted_label]["fp"] += 1
                counts[reference_label]["fn"] += 1
        per_label = {}
        f1_sum = 0.0
        valid_labels = 0
        for label in labels:
            stats = counts[label]
            precision_denom = stats["tp"] + stats["fp"]
            recall_denom = stats["tp"] + stats["fn"]
            precision = stats["tp"] / precision_denom if precision_denom else 0.0
            recall = stats["tp"] / recall_denom if recall_denom else 0.0
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            valid_labels += 1
            f1_sum += f1
            per_label[label] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": stats["tp"],
                "fp": stats["fp"],
                "fn": stats["fn"],
            }
        score = f1_sum / valid_labels if valid_labels else 0.0
        return MetricResult(
            metric_id=self.metadata.id,
            score=score,
            details={"per_label": per_label},
        )

