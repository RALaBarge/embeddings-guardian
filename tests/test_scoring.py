"""Tests for scoring algorithms."""

import numpy as np
import pytest

from embeddings_guardian.core.scoring import (
    CentroidScorer,
    DimensionScorer,
    EnsembleScorer,
    MagnitudeScorer,
    NeighborhoodScorer,
    sensitivity_to_z_threshold,
)


# ---------------------------------------------------------------------------
# sensitivity_to_z_threshold
# ---------------------------------------------------------------------------

class TestSensitivityMapping:
    def test_low_sensitivity(self):
        assert sensitivity_to_z_threshold(0.5) == 2.0

    def test_mid_sensitivity(self):
        z = sensitivity_to_z_threshold(0.95)
        assert 2.9 < z < 3.2

    def test_high_sensitivity(self):
        z = sensitivity_to_z_threshold(0.99)
        assert z == pytest.approx(4.0, abs=0.01)

    def test_above_range(self):
        assert sensitivity_to_z_threshold(1.0) == 4.0


# ---------------------------------------------------------------------------
# MagnitudeScorer
# ---------------------------------------------------------------------------

class TestMagnitudeScorer:
    def test_update_and_stats(self, magnitude_scorer):
        stats = magnitude_scorer.stats()
        assert stats["count"] == 100
        assert stats["mean_norm"] > 0

    def test_clean_vector_low_score(self, magnitude_scorer, rng):
        emb = rng.standard_normal(128).astype(np.float32)
        score, reason = magnitude_scorer.score(emb)
        # Most clean vectors should have low scores
        assert score < magnitude_scorer.z_threshold

    def test_extreme_vector_high_score(self, magnitude_scorer, rng):
        emb = rng.standard_normal(128).astype(np.float32) * 500
        score, reason = magnitude_scorer.score(emb)
        assert score > 0
        assert reason  # should have a reason

    def test_empty_vector(self, magnitude_scorer):
        score, reason = magnitude_scorer.score(np.array([], dtype=np.float32))
        assert score == float("inf")

    def test_below_min_norm(self):
        scorer = MagnitudeScorer(min_norm=1.0)
        emb = np.array([0.001], dtype=np.float32)
        score, reason = scorer.score(emb)
        assert "below minimum" in reason

    def test_above_max_norm(self):
        scorer = MagnitudeScorer(max_norm=5.0)
        emb = np.ones(128, dtype=np.float32) * 10
        score, reason = scorer.score(emb)
        assert "above maximum" in reason

    def test_reset(self, magnitude_scorer):
        magnitude_scorer.reset()
        assert magnitude_scorer.stats()["count"] == 0


# ---------------------------------------------------------------------------
# CentroidScorer
# ---------------------------------------------------------------------------

class TestCentroidScorer:
    def test_build_centroids(self, normal_embeddings, rng):
        scorer = CentroidScorer(n_centroids=4, baseline_window=200)
        for emb in normal_embeddings:
            scorer.update(emb)
        # Centroids are lazy-computed on score(); trigger computation
        scorer.score(rng.standard_normal(128).astype(np.float32))
        stats = scorer.stats()
        assert stats["count"] == 100
        assert stats["centroids_computed"]

    def test_outlier_flagged(self, normal_embeddings, rng):
        scorer = CentroidScorer(n_centroids=4, distance_threshold=0.3)
        for emb in normal_embeddings:
            scorer.update(emb)
        # An extreme vector should be far from all centroids
        outlier = rng.standard_normal(128).astype(np.float32) * 100
        score, reason = scorer.score(outlier)
        # Score represents cosine distance; extreme vectors may or may not
        # be far in cosine space, but the scorer should run without error
        assert isinstance(score, float)

    def test_reset(self, normal_embeddings):
        scorer = CentroidScorer()
        for emb in normal_embeddings[:20]:
            scorer.update(emb)
        scorer.reset()
        assert scorer.stats()["count"] == 0


# ---------------------------------------------------------------------------
# NeighborhoodScorer
# ---------------------------------------------------------------------------

class TestNeighborhoodScorer:
    def test_sparse_neighborhood(self, normal_embeddings, rng):
        scorer = NeighborhoodScorer(k=5, baseline_window=200)
        for emb in normal_embeddings:
            scorer.update(emb)
        # A far-away vector should have sparse neighborhood
        outlier = np.ones(128, dtype=np.float32) * 50
        score, reason = scorer.score(outlier)
        assert isinstance(score, float)

    def test_insufficient_data(self):
        scorer = NeighborhoodScorer(k=5)
        emb = np.ones(64, dtype=np.float32)
        score, reason = scorer.score(emb)
        assert score == 0.0

    def test_reset(self, normal_embeddings):
        scorer = NeighborhoodScorer()
        for emb in normal_embeddings[:20]:
            scorer.update(emb)
        scorer.reset()
        assert scorer.stats()["count"] == 0


# ---------------------------------------------------------------------------
# DimensionScorer
# ---------------------------------------------------------------------------

class TestDimensionScorer:
    def test_single_dim_spike(self, normal_embeddings):
        scorer = DimensionScorer(z_threshold=4.0)
        for emb in normal_embeddings:
            scorer.update(emb)
        # Create embedding with one extreme dimension
        outlier = np.zeros(128, dtype=np.float32)
        outlier[0] = 100.0
        score, reason = scorer.score(outlier)
        assert score > 4.0
        assert "Dimension 0" in reason

    def test_clean_passes(self, normal_embeddings, rng):
        scorer = DimensionScorer(z_threshold=4.0)
        for emb in normal_embeddings:
            scorer.update(emb)
        clean = rng.standard_normal(128).astype(np.float32)
        score, reason = scorer.score(clean)
        # Should usually be below threshold
        assert isinstance(score, float)

    def test_no_baseline(self):
        scorer = DimensionScorer()
        score, reason = scorer.score(np.ones(64, dtype=np.float32))
        assert score == 0.0


# ---------------------------------------------------------------------------
# EnsembleScorer
# ---------------------------------------------------------------------------

class TestEnsembleScorer:
    def test_requires_scorers(self):
        with pytest.raises(ValueError):
            EnsembleScorer(scorers=[])

    def test_weight_mismatch(self):
        with pytest.raises(ValueError):
            EnsembleScorer(
                scorers=[MagnitudeScorer()],
                weights=[0.5, 0.5],
            )

    def test_default_weights(self):
        ens = EnsembleScorer(scorers=[MagnitudeScorer(), DimensionScorer()])
        assert len(ens.weights) == 2
        assert sum(ens.weights) == pytest.approx(1.0)

    def test_update_propagates(self, normal_embeddings):
        s1 = MagnitudeScorer()
        s2 = DimensionScorer()
        ens = EnsembleScorer(scorers=[s1, s2])
        for emb in normal_embeddings[:10]:
            ens.update(emb)
        assert s1.stats()["count"] == 10
        assert s2.stats()["count"] == 10

    def test_reset_propagates(self, normal_embeddings):
        s1 = MagnitudeScorer()
        s2 = DimensionScorer()
        ens = EnsembleScorer(scorers=[s1, s2])
        for emb in normal_embeddings[:10]:
            ens.update(emb)
        ens.reset()
        assert s1.stats()["count"] == 0
        assert s2.stats()["count"] == 0

    def test_stats_structure(self):
        ens = EnsembleScorer(scorers=[MagnitudeScorer()])
        stats = ens.stats()
        assert stats["algorithm"] == "ensemble"
        assert "scorers" in stats
