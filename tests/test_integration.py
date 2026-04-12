"""Integration tests: full workflow from baseline to detection to reporting."""

import numpy as np
import pytest

from embeddings_guardian import PoisoningDetector
from embeddings_guardian.utils.metrics import DetectionMetrics
from embeddings_guardian.utils.reporting import Reporter


class TestFullWorkflow:
    """End-to-end: build baseline, scan, evaluate, report."""

    def test_magnitude_pipeline(self):
        rng = np.random.default_rng(42)
        detector = PoisoningDetector(algorithm="magnitude", sensitivity=0.95)

        # Build baseline from 200 clean embeddings
        clean = rng.standard_normal((200, 128)).astype(np.float32)
        detector.update_baseline_batch(clean)

        # Create test set: 50 clean + 10 poisoned
        test_clean = rng.standard_normal((50, 128)).astype(np.float32)
        test_poisoned = rng.standard_normal((10, 128)).astype(np.float32) * 500

        metrics = DetectionMetrics()
        reporter = Reporter()

        for i, emb in enumerate(test_clean):
            poisoned, conf, reason = detector.is_poisoned(emb)
            metrics.record(predicted=poisoned, actual=False, confidence=conf)
            if poisoned:
                reporter.add_finding(f"clean_{i}", conf, "magnitude", reason)

        for i, emb in enumerate(test_poisoned):
            poisoned, conf, reason = detector.is_poisoned(emb)
            metrics.record(predicted=poisoned, actual=True, confidence=conf)
            if poisoned:
                reporter.add_finding(f"poisoned_{i}", conf, "magnitude", reason)

        # Poisoned should be caught
        assert metrics.recall() > 0.8
        # False positive rate should be low
        assert metrics.false_positive_rate() < 0.1
        # Report should contain findings
        report = reporter.json_report()
        assert report["summary"]["total_findings"] > 0

    def test_ensemble_pipeline(self):
        rng = np.random.default_rng(99)
        detector = PoisoningDetector(
            algorithm=["magnitude", "dimension"],
            sensitivity=0.95,
        )

        clean = rng.standard_normal((100, 64)).astype(np.float32)
        detector.update_baseline_batch(clean)

        # Poisoned: one dimension spiked
        poisoned = rng.standard_normal(64).astype(np.float32)
        poisoned[0] = 200.0  # extreme spike

        is_p, conf, reason = detector.is_poisoned(poisoned)
        assert is_p
        assert conf > 0.0

    def test_scan_batch_integration(self):
        rng = np.random.default_rng(7)
        detector = PoisoningDetector(algorithm="magnitude")

        clean = rng.standard_normal((100, 32)).astype(np.float32)
        detector.update_baseline_batch(clean)

        # Mix clean and poisoned
        test = rng.standard_normal((20, 32)).astype(np.float32)
        test[15] = test[15] * 500  # poison one

        ids = [f"doc_{i}" for i in range(20)]
        findings = detector.scan_batch(test, ids=ids)

        poisoned_ids = {f["id"] for f in findings}
        assert "doc_15" in poisoned_ids

    def test_detector_stats_after_workflow(self):
        rng = np.random.default_rng(0)
        detector = PoisoningDetector(algorithm="magnitude")

        for _ in range(50):
            detector.update_baseline(rng.standard_normal(64).astype(np.float32))

        stats = detector.get_stats()
        assert stats["scorer"]["count"] == 50
        assert stats["scorer"]["mean_norm"] > 0

    def test_reset_and_retrain(self):
        rng = np.random.default_rng(0)
        detector = PoisoningDetector(algorithm="magnitude")

        # First training
        for _ in range(30):
            detector.update_baseline(rng.standard_normal(64).astype(np.float32))
        assert detector.get_stats()["scorer"]["count"] == 30

        # Reset
        detector.reset_baseline()
        assert detector.get_stats()["scorer"]["count"] == 0

        # Retrain
        for _ in range(20):
            detector.update_baseline(rng.standard_normal(64).astype(np.float32))
        assert detector.get_stats()["scorer"]["count"] == 20
