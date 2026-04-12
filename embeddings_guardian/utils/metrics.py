"""
Detection performance metrics.

Track true/false positives and negatives to evaluate detector accuracy
on labeled test data.

Usage::

    from embeddings_guardian.utils.metrics import DetectionMetrics

    metrics = DetectionMetrics()

    for emb, label in test_set:
        poisoned, conf, reason = detector.is_poisoned(emb)
        metrics.record(predicted=poisoned, actual=label)

    print(f"F1: {metrics.f1_score():.3f}")
    print(f"FPR: {metrics.false_positive_rate():.3f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class DetectionMetrics:
    """Accumulate and compute binary classification metrics."""

    tp: int = 0  # True positives  (correctly flagged poisoned)
    fp: int = 0  # False positives (clean flagged as poisoned)
    tn: int = 0  # True negatives  (correctly accepted clean)
    fn: int = 0  # False negatives (missed poisoned)
    _confidences: List[float] = field(default_factory=list, repr=False)

    def record(self, predicted: bool, actual: bool, confidence: float = 0.0) -> None:
        """Record a single prediction against ground truth."""
        if predicted and actual:
            self.tp += 1
        elif predicted and not actual:
            self.fp += 1
        elif not predicted and not actual:
            self.tn += 1
        else:
            self.fn += 1
        self._confidences.append(confidence)

    def precision(self) -> float:
        """TP / (TP + FP). Returns 0.0 if no positive predictions."""
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    def recall(self) -> float:
        """TP / (TP + FN). Returns 0.0 if no actual positives."""
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    def f1_score(self) -> float:
        """Harmonic mean of precision and recall."""
        p, r = self.precision(), self.recall()
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def false_positive_rate(self) -> float:
        """FP / (FP + TN). Returns 0.0 if no actual negatives."""
        denom = self.fp + self.tn
        return self.fp / denom if denom > 0 else 0.0

    def accuracy(self) -> float:
        """(TP + TN) / total."""
        total = self.tp + self.fp + self.tn + self.fn
        return (self.tp + self.tn) / total if total > 0 else 0.0

    def total(self) -> int:
        """Total number of recorded predictions."""
        return self.tp + self.fp + self.tn + self.fn

    def mean_confidence(self) -> float:
        """Mean detection confidence across all predictions."""
        if not self._confidences:
            return 0.0
        return sum(self._confidences) / len(self._confidences)

    def to_dict(self) -> Dict:
        """Return all metrics as a dictionary."""
        return {
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "precision": self.precision(),
            "recall": self.recall(),
            "f1_score": self.f1_score(),
            "false_positive_rate": self.false_positive_rate(),
            "accuracy": self.accuracy(),
            "total": self.total(),
            "mean_confidence": self.mean_confidence(),
        }

    def reset(self) -> None:
        """Clear all accumulated counts."""
        self.tp = self.fp = self.tn = self.fn = 0
        self._confidences.clear()
