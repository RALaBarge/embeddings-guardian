# Detection Algorithms

## Overview

embeddings-guardian provides five detection algorithms, each targeting
a different statistical property of poisoned embeddings. They can be used
individually or combined in an ensemble for maximum accuracy.

All algorithms share the same interface:
1. **`update(embedding)`** -- feed a known-good vector into the baseline.
2. **`score(embedding)`** -- return an anomaly score and explanation.
3. **`reset()`** -- clear baseline state.

## 1. Magnitude (L2 Norm Z-Score)

**Class:** `MagnitudeScorer`

**How it works:**

Poisoned embeddings often have abnormal magnitudes. The magnitude scorer
tracks the L2 norm (magnitude) of baseline embeddings in a rolling window
and computes the z-score of each incoming vector:

```
z = |norm(embedding) - mean_norm| / std_norm
```

If the z-score exceeds the threshold (derived from sensitivity), the
embedding is flagged. Additionally, hard bounds check for norms below
`min_norm` or above `max_norm`.

**Parameters:**
- `sensitivity` (float, default 0.95): Maps to z-threshold. 0.95 = 3-sigma.
- `baseline_window` (int, default 1000): Rolling window size.
- `min_norm` (float, default 0.1): Lower magnitude bound.
- `max_norm` (float, default 100.0): Upper magnitude bound.

**Performance:** O(1) per vector, <0.5ms.

**When to use:** Default choice. Fast, effective against naive poisoning
attacks that produce embeddings with unusual magnitudes.

**Limitations:** Sophisticated attacks that carefully match the magnitude
distribution of the corpus will bypass this check.

## 2. Centroid Distance

**Class:** `CentroidScorer`

**How it works:**

Clusters the baseline embeddings using mini-batch k-means, then measures
the cosine distance from each incoming embedding to the nearest centroid.
Embeddings far from all centroids are flagged as semantic outliers.

```
distance = 1 - max(cosine_similarity(embedding, centroid_i))
```

**Parameters:**
- `n_centroids` (int, default 8): Number of cluster centroids.
- `baseline_window` (int, default 2000): Buffer size for clustering.
- `distance_threshold` (float, default 0.5): Cosine distance above which
  to flag.

**Performance:** O(k) per vector where k = centroids, ~1ms.

**When to use:** When you expect poisoned documents to be semantically
unrelated to the rest of the corpus. Good for detecting topic drift.

**Limitations:** Requires enough baseline data to form meaningful clusters.
Recomputes centroids on each score call if the buffer has changed (lazy).

## 3. Neighborhood Density (RevPRAG-inspired)

**Class:** `NeighborhoodScorer`

**How it works:**

Computes the mean distance to k nearest neighbors for each embedding.
Poisoned embeddings tend to sit in sparse neighborhoods (few nearby vectors)
because they are optimized to be close to specific queries rather than to
the rest of the corpus.

```
knn_mean = mean(distances to k nearest neighbors)
z = (knn_mean - corpus_mean_density) / corpus_std_density
```

**Parameters:**
- `k` (int, default 5): Number of neighbors to consider.
- `baseline_window` (int, default 2000): Corpus size cap.
- `z_threshold` (float, default 3.0): Z-score threshold.

**Performance:** O(n) brute-force, 5--10ms. Can be accelerated with an
approximate nearest-neighbor index.

**When to use:** For sophisticated attacks where the attacker matches
magnitude and topic but the embedding is isolated in vector space.
Inspired by the RevPRAG paper (Zeng et al., 2024).

**Limitations:** Slower than magnitude/dimension checks. Brute-force
k-NN becomes expensive above ~10k baseline vectors.

## 4. Dimension Outlier

**Class:** `DimensionScorer`

**How it works:**

Tracks per-dimension mean and standard deviation across the baseline.
For each incoming embedding, computes the z-score of every dimension
and flags if any single dimension exceeds the threshold.

```
z_scores = |embedding - dim_means| / dim_stds
max_z = max(z_scores)
```

**Parameters:**
- `z_threshold` (float, default 4.0): Per-dimension z-score threshold.
- `baseline_window` (int, default 1000): Baseline buffer size.

**Performance:** O(d) per vector where d = dimensions, <0.5ms.

**When to use:** As a secondary check alongside magnitude. Catches
attacks that spike individual dimensions while keeping overall magnitude
normal.

**Limitations:** High-dimensional embeddings (d > 1000) naturally have
some dimensions with high z-scores (curse of dimensionality). May need
a higher threshold for very high dimensions.

## 5. Ensemble

**Class:** `EnsembleScorer`

**How it works:**

Combines multiple scorers with configurable weights. The final score is
a weighted average of all component scores.

```python
detector = PoisoningDetector(
    algorithm=["magnitude", "centroid", "dimension"],
    sensitivity=0.95,
)
```

Or with explicit weights:

```python
from embeddings_guardian.core.scoring import (
    EnsembleScorer, MagnitudeScorer, CentroidScorer,
)

ensemble = EnsembleScorer(
    scorers=[MagnitudeScorer(), CentroidScorer()],
    weights=[0.7, 0.3],
    threshold=3.0,
)
detector = PoisoningDetector(algorithm=ensemble)
```

**When to use:** Maximum detection accuracy. Recommended for production
systems where false negatives are costly.

## Choosing an Algorithm

| Scenario | Recommended |
|----------|-------------|
| Real-time ingestion, high throughput | `magnitude` |
| Offline corpus audit | `ensemble` |
| Known sophisticated attacker | `neighborhood` |
| Quick secondary validation | `dimension` |
| Topic drift detection | `centroid` |
| Maximum coverage | `["magnitude", "centroid", "dimension"]` |

## Sensitivity Tuning

The `sensitivity` parameter on `PoisoningDetector` maps to a z-score
threshold via linear interpolation:

| Sensitivity | Z-Threshold | Approx. False Positive Rate |
|-------------|-------------|---------------------------|
| 0.90 | 2.0 | ~5% |
| 0.95 | 3.0 | ~1% |
| 0.97 | 3.5 | ~0.3% |
| 0.99 | 4.0 | ~0.1% |

Start with 0.95 and adjust based on your false positive tolerance.
Run `DetectionMetrics` on a labeled test set to calibrate.
