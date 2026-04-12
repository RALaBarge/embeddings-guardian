"""Shared test fixtures for embeddings-guardian."""

import numpy as np
import pytest

from embeddings_guardian import PoisoningDetector
from embeddings_guardian.core.scoring import MagnitudeScorer


# ---------------------------------------------------------------------------
# Embedding generators
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    """Seeded random generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def normal_embeddings(rng):
    """100 clean embeddings drawn from a normal distribution (dim=128)."""
    return rng.standard_normal((100, 128)).astype(np.float32)


@pytest.fixture
def poisoned_embedding_high_norm(rng):
    """Single poisoned embedding with abnormally high magnitude."""
    emb = rng.standard_normal(128).astype(np.float32)
    return emb * 500.0  # way outside normal range


@pytest.fixture
def poisoned_embedding_zero():
    """All-zeros embedding."""
    return np.zeros(128, dtype=np.float32)


@pytest.fixture
def poisoned_embedding_tiny(rng):
    """Embedding with near-zero magnitude."""
    emb = rng.standard_normal(128).astype(np.float32)
    return emb * 1e-6


# ---------------------------------------------------------------------------
# Pre-configured detectors
# ---------------------------------------------------------------------------

@pytest.fixture
def trained_detector(normal_embeddings):
    """Detector with baseline built from normal_embeddings."""
    det = PoisoningDetector(algorithm="magnitude", sensitivity=0.95)
    det.update_baseline_batch(normal_embeddings)
    return det


@pytest.fixture
def magnitude_scorer(normal_embeddings):
    """MagnitudeScorer with baseline built from normal_embeddings."""
    scorer = MagnitudeScorer(sensitivity=0.95)
    for emb in normal_embeddings:
        scorer.update(emb)
    return scorer
