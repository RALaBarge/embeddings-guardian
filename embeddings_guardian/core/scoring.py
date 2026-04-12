"""
Anomaly detection scoring algorithms for embedding poisoning detection.

Each scorer implements a different statistical approach to identifying
anomalous embeddings. All scorers share a common interface: ``update()``
to feed baseline data and ``score()`` to evaluate a candidate vector.

Available algorithms:

- **MagnitudeScorer** -- L2 norm z-score (fastest, default)
- **CentroidScorer** -- cosine distance to semantic centroids
- **NeighborhoodScorer** -- k-NN density analysis (RevPRAG-inspired)
- **DimensionScorer** -- per-dimension z-score outlier detection
- **EnsembleScorer** -- weighted combination of multiple scorers
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import deque
from threading import Lock
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_float32(embedding: ArrayLike) -> np.ndarray:
    """Coerce input to a float32 numpy array."""
    if isinstance(embedding, list):
        return np.array(embedding, dtype=np.float32)
    return np.asarray(embedding, dtype=np.float32)


def sensitivity_to_z_threshold(sensitivity: float) -> float:
    """Map a sensitivity value in [0, 1] to a z-score threshold.

    sensitivity=0.95 -> z=3.0 (3-sigma)
    sensitivity=0.99 -> z=4.0 (4-sigma)

    Below 0.90 clamps to z=2.0; above 0.99 clamps to z=4.0.
    """
    if sensitivity < 0.90:
        return 2.0
    if sensitivity > 0.99:
        return 4.0
    return 2.0 + (sensitivity - 0.90) / 0.09 * 2.0


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseScorer(ABC):
    """Abstract interface that all scoring algorithms implement."""

    @abstractmethod
    def update(self, embedding: np.ndarray) -> None:
        """Incorporate a known-good embedding into the baseline."""

    @abstractmethod
    def score(self, embedding: np.ndarray) -> Tuple[float, str]:
        """Return ``(anomaly_score, reason)`` for *embedding*.

        The anomaly score is non-negative.  Higher values indicate greater
        likelihood of poisoning.  The *reason* string is human-readable.
        """

    @abstractmethod
    def reset(self) -> None:
        """Clear all accumulated baseline state."""

    @abstractmethod
    def stats(self) -> Dict:
        """Return diagnostic statistics about the current baseline."""


# ---------------------------------------------------------------------------
# Magnitude scorer  (L2 norm z-score)
# ---------------------------------------------------------------------------

class MagnitudeScorer(BaseScorer):
    """Detect anomalies via L2-norm z-score.

    Maintains a rolling window of embedding magnitudes and flags vectors
    whose norm deviates significantly from the baseline distribution.

    Complexity: O(1) per vector, <0.5 ms typical.
    """

    def __init__(
        self,
        sensitivity: float = 0.95,
        baseline_window: int = 1000,
        min_norm: float = 0.1,
        max_norm: float = 100.0,
    ) -> None:
        self.sensitivity = sensitivity
        self.baseline_window = baseline_window
        self.min_norm = min_norm
        self.max_norm = max_norm
        self.z_threshold = sensitivity_to_z_threshold(sensitivity)

        self._lock = Lock()
        self._norms: deque = deque(maxlen=baseline_window)
        self._mean: float = 0.0
        self._std: float = 1.0
        self._count: int = 0

    # -- BaseScorer interface -----------------------------------------------

    def update(self, embedding: np.ndarray) -> None:
        norm = float(np.linalg.norm(embedding))
        with self._lock:
            self._norms.append(norm)
            self._count += 1
            if len(self._norms) > 0:
                self._mean = float(np.mean(self._norms))
                if len(self._norms) > 1:
                    self._std = float(np.std(self._norms))
                    if self._std < 1e-6:
                        self._std = 1.0

    def score(self, embedding: np.ndarray) -> Tuple[float, str]:
        if embedding.size == 0:
            return (float("inf"), "Empty embedding")

        norm = float(np.linalg.norm(embedding))

        # Hard bounds check
        if norm < self.min_norm:
            return (
                abs(self.min_norm - norm) / max(self.min_norm, 1e-9),
                f"Magnitude below minimum (norm={norm:.4f}, min={self.min_norm})",
            )
        if norm > self.max_norm:
            return (
                abs(norm - self.max_norm) / max(self.max_norm, 1e-9),
                f"Magnitude above maximum (norm={norm:.4f}, max={self.max_norm})",
            )

        # Z-score against rolling baseline
        with self._lock:
            mean, std = self._mean, self._std

        if self._count > 0 and std > 0:
            z = abs((norm - mean) / std)
            if z > self.z_threshold:
                return (
                    z,
                    f"Magnitude anomaly (z-score={z:.2f}, threshold={self.z_threshold:.2f})",
                )
            return (z, "")

        return (0.0, "")

    def reset(self) -> None:
        with self._lock:
            self._norms.clear()
            self._mean = 0.0
            self._std = 1.0
            self._count = 0

    def stats(self) -> Dict:
        with self._lock:
            return {
                "algorithm": "magnitude",
                "count": self._count,
                "mean_norm": self._mean,
                "std_norm": self._std,
                "z_threshold": self.z_threshold,
                "window_size": len(self._norms),
                "min_norm": self.min_norm,
                "max_norm": self.max_norm,
            }


# ---------------------------------------------------------------------------
# Centroid scorer  (cosine distance to cluster centroids)
# ---------------------------------------------------------------------------

class CentroidScorer(BaseScorer):
    """Detect anomalies via cosine distance to semantic centroids.

    Embeddings far from all centroids are flagged.  Centroids are computed
    via mini-batch k-means over the baseline window.

    Complexity: O(k) per vector where k = number of centroids, ~1 ms.
    """

    def __init__(
        self,
        n_centroids: int = 8,
        baseline_window: int = 2000,
        distance_threshold: float = 0.5,
    ) -> None:
        self.n_centroids = n_centroids
        self.baseline_window = baseline_window
        self.distance_threshold = distance_threshold

        self._lock = Lock()
        self._buffer: List[np.ndarray] = []
        self._centroids: Optional[np.ndarray] = None
        self._count: int = 0
        self._dirty: bool = False

    def _recompute_centroids(self) -> None:
        """Fit k-means on the buffered embeddings."""
        if len(self._buffer) < self.n_centroids:
            return
        from sklearn.cluster import MiniBatchKMeans

        X = np.vstack(self._buffer)
        km = MiniBatchKMeans(n_clusters=self.n_centroids, n_init=1, random_state=42)
        km.fit(X)
        self._centroids = km.cluster_centers_
        self._dirty = False

    def update(self, embedding: np.ndarray) -> None:
        with self._lock:
            self._buffer.append(embedding.copy())
            self._count += 1
            self._dirty = True
            # Cap buffer
            if len(self._buffer) > self.baseline_window:
                self._buffer = self._buffer[-self.baseline_window :]

    def score(self, embedding: np.ndarray) -> Tuple[float, str]:
        with self._lock:
            if self._dirty:
                self._recompute_centroids()
            centroids = self._centroids

        if centroids is None:
            return (0.0, "")

        # Cosine distances to each centroid
        emb_norm = embedding / (np.linalg.norm(embedding) + 1e-9)
        cent_norms = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9)
        similarities = cent_norms @ emb_norm
        min_distance = 1.0 - float(np.max(similarities))

        if min_distance > self.distance_threshold:
            return (
                min_distance,
                f"Far from all centroids (min_cos_dist={min_distance:.4f}, "
                f"threshold={self.distance_threshold})",
            )
        return (min_distance, "")

    def reset(self) -> None:
        with self._lock:
            self._buffer.clear()
            self._centroids = None
            self._count = 0
            self._dirty = False

    def stats(self) -> Dict:
        with self._lock:
            return {
                "algorithm": "centroid",
                "count": self._count,
                "n_centroids": self.n_centroids,
                "centroids_computed": self._centroids is not None,
                "buffer_size": len(self._buffer),
                "distance_threshold": self.distance_threshold,
            }


# ---------------------------------------------------------------------------
# Neighborhood scorer  (k-NN density, RevPRAG-inspired)
# ---------------------------------------------------------------------------

class NeighborhoodScorer(BaseScorer):
    """Detect anomalies via k-NN neighborhood density.

    Poisoned embeddings tend to sit in sparse neighborhoods.  This scorer
    computes mean distance to k nearest neighbors and flags embeddings
    whose neighborhood is unusually sparse compared to the baseline.

    Complexity: O(n) brute-force or O(log n) with an index, 5-10 ms.
    """

    def __init__(
        self,
        k: int = 5,
        baseline_window: int = 2000,
        z_threshold: float = 3.0,
    ) -> None:
        self.k = k
        self.baseline_window = baseline_window
        self.z_threshold = z_threshold

        self._lock = Lock()
        self._embeddings: List[np.ndarray] = []
        self._density_scores: deque = deque(maxlen=baseline_window)
        self._count: int = 0

    def _knn_mean_distance(self, embedding: np.ndarray) -> float:
        """Brute-force k-NN mean distance."""
        if len(self._embeddings) < self.k + 1:
            return 0.0
        X = np.vstack(self._embeddings)
        dists = np.linalg.norm(X - embedding, axis=1)
        # k+1 because the embedding itself may be in the set
        topk = np.sort(dists)[: self.k + 1]
        return float(np.mean(topk[topk > 1e-9][: self.k]))

    def update(self, embedding: np.ndarray) -> None:
        with self._lock:
            self._embeddings.append(embedding.copy())
            self._count += 1
            if len(self._embeddings) > self.baseline_window:
                self._embeddings = self._embeddings[-self.baseline_window :]
            # Track density of baseline embeddings
            d = self._knn_mean_distance(embedding)
            if d > 0:
                self._density_scores.append(d)

    def score(self, embedding: np.ndarray) -> Tuple[float, str]:
        with self._lock:
            d = self._knn_mean_distance(embedding)

        if len(self._density_scores) < 2 or d == 0.0:
            return (0.0, "")

        with self._lock:
            mean_d = float(np.mean(self._density_scores))
            std_d = float(np.std(self._density_scores))

        if std_d < 1e-9:
            return (0.0, "")

        z = (d - mean_d) / std_d
        if z > self.z_threshold:
            return (
                z,
                f"Sparse neighborhood (z={z:.2f}, threshold={self.z_threshold})",
            )
        return (abs(z), "")

    def reset(self) -> None:
        with self._lock:
            self._embeddings.clear()
            self._density_scores.clear()
            self._count = 0

    def stats(self) -> Dict:
        with self._lock:
            return {
                "algorithm": "neighborhood",
                "count": self._count,
                "k": self.k,
                "corpus_size": len(self._embeddings),
                "z_threshold": self.z_threshold,
            }


# ---------------------------------------------------------------------------
# Dimension scorer  (per-dimension z-score)
# ---------------------------------------------------------------------------

class DimensionScorer(BaseScorer):
    """Detect anomalies via per-dimension z-score.

    Tracks mean and std for each dimension across the baseline.  Flags
    embeddings where any single dimension exceeds the z-score threshold.

    Complexity: O(d) per vector, <0.5 ms.
    """

    def __init__(
        self,
        z_threshold: float = 4.0,
        baseline_window: int = 1000,
    ) -> None:
        self.z_threshold = z_threshold
        self.baseline_window = baseline_window

        self._lock = Lock()
        self._buffer: List[np.ndarray] = []
        self._dim_means: Optional[np.ndarray] = None
        self._dim_stds: Optional[np.ndarray] = None
        self._count: int = 0

    def _recompute(self) -> None:
        if len(self._buffer) < 2:
            return
        X = np.vstack(self._buffer)
        self._dim_means = X.mean(axis=0)
        self._dim_stds = X.std(axis=0)
        # Avoid div-by-zero
        self._dim_stds[self._dim_stds < 1e-9] = 1.0

    def update(self, embedding: np.ndarray) -> None:
        with self._lock:
            self._buffer.append(embedding.copy())
            self._count += 1
            if len(self._buffer) > self.baseline_window:
                self._buffer = self._buffer[-self.baseline_window :]
            self._recompute()

    def score(self, embedding: np.ndarray) -> Tuple[float, str]:
        with self._lock:
            means = self._dim_means
            stds = self._dim_stds

        if means is None or stds is None:
            return (0.0, "")

        z_scores = np.abs((embedding - means) / stds)
        max_z = float(np.max(z_scores))
        max_dim = int(np.argmax(z_scores))

        if max_z > self.z_threshold:
            return (
                max_z,
                f"Dimension {max_dim} outlier (z={max_z:.2f}, threshold={self.z_threshold})",
            )
        return (max_z, "")

    def reset(self) -> None:
        with self._lock:
            self._buffer.clear()
            self._dim_means = None
            self._dim_stds = None
            self._count = 0

    def stats(self) -> Dict:
        with self._lock:
            return {
                "algorithm": "dimension",
                "count": self._count,
                "z_threshold": self.z_threshold,
                "dimensions": self._dim_means.shape[0] if self._dim_means is not None else 0,
            }


# ---------------------------------------------------------------------------
# Ensemble scorer
# ---------------------------------------------------------------------------

class EnsembleScorer(BaseScorer):
    """Combine multiple scorers with configurable weights.

    Example::

        ensemble = EnsembleScorer(
            scorers=[MagnitudeScorer(), CentroidScorer()],
            weights=[0.6, 0.4],
            threshold=3.0,
        )
    """

    def __init__(
        self,
        scorers: Sequence[BaseScorer],
        weights: Optional[Sequence[float]] = None,
        threshold: float = 3.0,
    ) -> None:
        if not scorers:
            raise ValueError("At least one scorer is required")
        self.scorers = list(scorers)
        if weights is None:
            weights = [1.0 / len(scorers)] * len(scorers)
        if len(weights) != len(scorers):
            raise ValueError("weights must match number of scorers")
        total = sum(weights)
        self.weights = [w / total for w in weights]
        self.threshold = threshold

    def update(self, embedding: np.ndarray) -> None:
        for s in self.scorers:
            s.update(embedding)

    def score(self, embedding: np.ndarray) -> Tuple[float, str]:
        total = 0.0
        reasons = []
        for scorer, w in zip(self.scorers, self.weights):
            s, reason = scorer.score(embedding)
            total += s * w
            if reason:
                reasons.append(reason)

        if total > self.threshold:
            combined = "; ".join(reasons) if reasons else f"Ensemble score {total:.2f}"
            return (total, f"Ensemble anomaly ({combined})")
        return (total, "")

    def reset(self) -> None:
        for s in self.scorers:
            s.reset()

    def stats(self) -> Dict:
        return {
            "algorithm": "ensemble",
            "scorers": [s.stats() for s in self.scorers],
            "weights": self.weights,
            "threshold": self.threshold,
        }
