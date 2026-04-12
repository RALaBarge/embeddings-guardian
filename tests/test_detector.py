"""Tests for PoisoningDetector (core detection interface)."""

import numpy as np
import pytest

from embeddings_guardian import PoisoningDetector


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestDetectorInit:
    def test_default_init(self):
        det = PoisoningDetector()
        stats = det.get_stats()
        assert stats["sensitivity"] == 0.95
        assert stats["mode"] == "warn"

    def test_custom_sensitivity(self):
        det = PoisoningDetector(sensitivity=0.99)
        assert det.get_stats()["z_threshold"] > 3.5

    def test_invalid_algorithm_raises(self):
        with pytest.raises(ValueError, match="Unknown algorithm"):
            PoisoningDetector(algorithm="nonexistent")

    def test_ensemble_from_list(self):
        det = PoisoningDetector(algorithm=["magnitude", "centroid"])
        assert det.get_stats()["scorer"]["algorithm"] == "ensemble"

    def test_ensemble_keyword(self):
        det = PoisoningDetector(algorithm="ensemble")
        assert det.get_stats()["scorer"]["algorithm"] == "ensemble"

    def test_custom_scorer(self):
        from embeddings_guardian.core.scoring import DimensionScorer
        scorer = DimensionScorer()
        det = PoisoningDetector(algorithm=scorer)
        assert det.get_stats()["scorer"]["algorithm"] == "dimension"

    def test_modes(self):
        for mode in ("warn", "quarantine", "strict"):
            det = PoisoningDetector(mode=mode)
            assert det.get_stats()["mode"] == mode


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

class TestBaseline:
    def test_update_single(self, rng):
        det = PoisoningDetector()
        emb = rng.standard_normal(64).astype(np.float32)
        det.update_baseline(emb)
        assert det.get_stats()["scorer"]["count"] == 1

    def test_update_batch(self, normal_embeddings):
        det = PoisoningDetector()
        det.update_baseline_batch(normal_embeddings)
        assert det.get_stats()["scorer"]["count"] == len(normal_embeddings)

    def test_update_from_list(self):
        det = PoisoningDetector()
        det.update_baseline([1.0, 2.0, 3.0])
        assert det.get_stats()["scorer"]["count"] == 1

    def test_reset(self, normal_embeddings):
        det = PoisoningDetector()
        det.update_baseline_batch(normal_embeddings)
        det.reset_baseline()
        assert det.get_stats()["scorer"]["count"] == 0


# ---------------------------------------------------------------------------
# Detection: clean embeddings
# ---------------------------------------------------------------------------

class TestCleanDetection:
    def test_normal_embedding_passes(self, trained_detector, rng):
        emb = rng.standard_normal(128).astype(np.float32)
        poisoned, confidence, reason = trained_detector.is_poisoned(emb)
        assert not poisoned
        assert confidence == 0.0
        assert reason == ""

    def test_batch_clean(self, trained_detector, rng):
        batch = rng.standard_normal((20, 128)).astype(np.float32)
        findings = trained_detector.scan_batch(batch)
        # Most should pass (allow a few false positives)
        assert len(findings) < 5


# ---------------------------------------------------------------------------
# Detection: poisoned embeddings
# ---------------------------------------------------------------------------

class TestPoisonedDetection:
    def test_empty_embedding(self, trained_detector):
        poisoned, confidence, reason = trained_detector.is_poisoned(np.array([]))
        assert poisoned
        assert confidence == 1.0
        assert "Empty" in reason

    def test_high_magnitude(self, trained_detector, poisoned_embedding_high_norm):
        poisoned, confidence, reason = trained_detector.is_poisoned(poisoned_embedding_high_norm)
        assert poisoned
        assert confidence > 0.5

    def test_near_zero_magnitude(self, trained_detector, poisoned_embedding_tiny):
        poisoned, confidence, reason = trained_detector.is_poisoned(poisoned_embedding_tiny)
        assert poisoned

    def test_all_zeros(self, trained_detector, poisoned_embedding_zero):
        poisoned, confidence, reason = trained_detector.is_poisoned(poisoned_embedding_zero)
        assert poisoned

    def test_scan_batch_with_ids(self, trained_detector, poisoned_embedding_high_norm, rng):
        clean = rng.standard_normal((5, 128)).astype(np.float32)
        batch = np.vstack([clean, poisoned_embedding_high_norm.reshape(1, -1)])
        ids = [f"doc_{i}" for i in range(6)]
        findings = trained_detector.scan_batch(batch, ids=ids)
        poisoned_ids = [f["id"] for f in findings]
        assert "doc_5" in poisoned_ids

    def test_list_input(self, trained_detector):
        poisoned, _, _ = trained_detector.is_poisoned([0.0] * 128)
        assert poisoned


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_no_baseline_accepts(self):
        det = PoisoningDetector()
        emb = np.ones(64, dtype=np.float32)
        poisoned, confidence, reason = det.is_poisoned(emb)
        # Without baseline, magnitude bounds are the only check
        # norm(ones(64)) = 8.0, within [0.1, 100]
        assert not poisoned

    def test_single_dimension(self, rng):
        det = PoisoningDetector()
        for _ in range(50):
            det.update_baseline(rng.standard_normal(1).astype(np.float32))
        poisoned, _, _ = det.is_poisoned(np.array([100.0], dtype=np.float32))
        assert poisoned

    def test_large_batch_scan(self, trained_detector, rng):
        batch = rng.standard_normal((1000, 128)).astype(np.float32)
        findings = trained_detector.scan_batch(batch)
        # False positive rate should be low
        assert len(findings) / 1000 < 0.05
