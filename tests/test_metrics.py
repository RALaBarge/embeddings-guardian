"""Tests for detection metrics."""

import pytest
from embeddings_guardian.utils.metrics import DetectionMetrics


class TestDetectionMetrics:
    def test_empty(self):
        m = DetectionMetrics()
        assert m.total() == 0
        assert m.precision() == 0.0
        assert m.recall() == 0.0
        assert m.f1_score() == 0.0

    def test_perfect_precision(self):
        m = DetectionMetrics()
        m.record(predicted=True, actual=True)
        m.record(predicted=True, actual=True)
        assert m.precision() == 1.0

    def test_perfect_recall(self):
        m = DetectionMetrics()
        m.record(predicted=True, actual=True)
        m.record(predicted=False, actual=False)
        assert m.recall() == 1.0

    def test_f1(self):
        m = DetectionMetrics()
        m.record(predicted=True, actual=True)   # TP
        m.record(predicted=True, actual=False)  # FP
        m.record(predicted=False, actual=True)  # FN
        m.record(predicted=False, actual=False) # TN
        assert m.precision() == 0.5
        assert m.recall() == 0.5
        assert m.f1_score() == 0.5

    def test_false_positive_rate(self):
        m = DetectionMetrics()
        m.record(predicted=True, actual=False)  # FP
        m.record(predicted=False, actual=False) # TN
        assert m.false_positive_rate() == 0.5

    def test_accuracy(self):
        m = DetectionMetrics()
        m.record(predicted=True, actual=True)
        m.record(predicted=False, actual=False)
        m.record(predicted=False, actual=False)
        assert m.accuracy() == pytest.approx(1.0)

    def test_mean_confidence(self):
        m = DetectionMetrics()
        m.record(predicted=True, actual=True, confidence=0.9)
        m.record(predicted=True, actual=True, confidence=0.7)
        assert m.mean_confidence() == pytest.approx(0.8)

    def test_to_dict(self):
        m = DetectionMetrics()
        m.record(predicted=True, actual=True)
        d = m.to_dict()
        assert "precision" in d
        assert "recall" in d
        assert d["tp"] == 1

    def test_reset(self):
        m = DetectionMetrics()
        m.record(predicted=True, actual=True)
        m.reset()
        assert m.total() == 0
        assert m.mean_confidence() == 0.0
