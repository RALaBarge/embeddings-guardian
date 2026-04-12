"""
Ensemble detection example.

Demonstrates combining multiple scoring algorithms with custom weights
for maximum detection accuracy.
"""

import numpy as np

from embeddings_guardian import PoisoningDetector
from embeddings_guardian.core.scoring import (
    CentroidScorer,
    DimensionScorer,
    EnsembleScorer,
    MagnitudeScorer,
)

# Option 1: Use algorithm list (equal weights)
detector_simple = PoisoningDetector(
    algorithm=["magnitude", "centroid", "dimension"],
    sensitivity=0.95,
)

# Option 2: Custom weights via EnsembleScorer
ensemble = EnsembleScorer(
    scorers=[
        MagnitudeScorer(sensitivity=0.95),
        CentroidScorer(n_centroids=16, distance_threshold=0.4),
        DimensionScorer(z_threshold=4.0),
    ],
    weights=[0.5, 0.3, 0.2],  # magnitude weighted highest
    threshold=3.0,
)

detector_custom = PoisoningDetector(algorithm=ensemble)

# Build baseline
rng = np.random.default_rng(42)
baseline = rng.standard_normal((500, 128)).astype(np.float32)

detector_custom.update_baseline_batch(baseline)

# Test detection
clean = rng.standard_normal(128).astype(np.float32)
poisoned = rng.standard_normal(128).astype(np.float32) * 100

print("Clean embedding:")
is_p, conf, reason = detector_custom.is_poisoned(clean)
print(f"  Poisoned: {is_p}, Confidence: {conf:.2f}")

print("Poisoned embedding:")
is_p, conf, reason = detector_custom.is_poisoned(poisoned)
print(f"  Poisoned: {is_p}, Confidence: {conf:.2f}, Reason: {reason}")

# Stats show all sub-scorers
stats = detector_custom.get_stats()
for scorer_stats in stats["scorer"]["scorers"]:
    print(f"  {scorer_stats['algorithm']}: count={scorer_stats['count']}")
