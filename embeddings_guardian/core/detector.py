"""
PoisoningDetector -- main detection interface.

This is the primary entry point for users of the library.  It wraps one
or more scoring algorithms and provides a clean ``is_poisoned()`` API
that returns a verdict, confidence, and human-readable reason.

Ported from BeigeBox ``beigebox/security/rag_poisoning_detector.py``
and extended with multi-algorithm support.

Basic usage::

    from embeddings_guardian import PoisoningDetector

    detector = PoisoningDetector(algorithm="magnitude", sensitivity=0.95)

    # Feed known-good embeddings
    for emb in training_embeddings:
        detector.update_baseline(emb)

    # Check a suspect vector
    poisoned, confidence, reason = detector.is_poisoned(query_vector)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from embeddings_guardian.core.scoring import (
    BaseScorer,
    CentroidScorer,
    DimensionScorer,
    EnsembleScorer,
    MagnitudeScorer,
    NeighborhoodScorer,
    sensitivity_to_z_threshold,
)

logger = logging.getLogger(__name__)

# Registry of built-in algorithm names -> scorer factories
_ALGORITHM_FACTORIES = {
    "magnitude": lambda **kw: MagnitudeScorer(**kw),
    "centroid": lambda **kw: CentroidScorer(**kw),
    "neighborhood": lambda **kw: NeighborhoodScorer(**kw),
    "dimension": lambda **kw: DimensionScorer(**kw),
}


def _to_float32(embedding: ArrayLike) -> np.ndarray:
    if isinstance(embedding, list):
        return np.array(embedding, dtype=np.float32)
    return np.asarray(embedding, dtype=np.float32)


class PoisoningDetector:
    """Main poisoning detection interface.

    Parameters
    ----------
    algorithm : str or BaseScorer or list
        Detection algorithm to use.  Built-in names: ``"magnitude"``,
        ``"centroid"``, ``"neighborhood"``, ``"dimension"``.
        Pass ``"ensemble"`` or a list of names to combine algorithms.
        You can also pass a pre-configured :class:`BaseScorer` instance.
    sensitivity : float
        Detection sensitivity in ``[0, 1]``.  Higher values mean fewer
        false positives but more missed detections.  Default 0.95
        corresponds to a 3-sigma threshold.
    baseline_window : int
        Number of vectors kept in the rolling baseline window.
    mode : str
        Response mode: ``"warn"`` (log only), ``"quarantine"``
        (flag in store), or ``"strict"`` (reject outright).
    **scorer_kwargs
        Extra keyword arguments forwarded to the scorer constructor.
    """

    def __init__(
        self,
        algorithm: Union[str, BaseScorer, Sequence[str]] = "magnitude",
        sensitivity: float = 0.95,
        baseline_window: int = 1000,
        mode: str = "warn",
        **scorer_kwargs,
    ) -> None:
        self.sensitivity = sensitivity
        self.baseline_window = baseline_window
        self.mode = mode
        self._z_threshold = sensitivity_to_z_threshold(sensitivity)

        # Build scorer(s)
        if isinstance(algorithm, BaseScorer):
            self._scorer = algorithm
        elif isinstance(algorithm, str) and algorithm != "ensemble":
            factory = _ALGORITHM_FACTORIES.get(algorithm)
            if factory is None:
                raise ValueError(
                    f"Unknown algorithm {algorithm!r}. "
                    f"Choose from: {list(_ALGORITHM_FACTORIES)}"
                )
            kw = {"baseline_window": baseline_window}
            if algorithm in ("magnitude",):
                kw["sensitivity"] = sensitivity
            kw.update(scorer_kwargs)
            self._scorer = factory(**kw)
        elif isinstance(algorithm, (list, tuple)):
            scorers = []
            for name in algorithm:
                factory = _ALGORITHM_FACTORIES.get(name)
                if factory is None:
                    raise ValueError(f"Unknown algorithm {name!r}")
                kw = {"baseline_window": baseline_window}
                if name == "magnitude":
                    kw["sensitivity"] = sensitivity
                scorers.append(factory(**kw))
            self._scorer = EnsembleScorer(scorers=scorers, threshold=self._z_threshold)
        else:
            # "ensemble" keyword without explicit list -> magnitude + centroid
            self._scorer = EnsembleScorer(
                scorers=[
                    MagnitudeScorer(sensitivity=sensitivity, baseline_window=baseline_window),
                    CentroidScorer(baseline_window=baseline_window),
                ],
                threshold=self._z_threshold,
            )

        logger.info(
            "PoisoningDetector initialized (algorithm=%s, sensitivity=%.2f, "
            "z_threshold=%.2f, mode=%s)",
            algorithm,
            sensitivity,
            self._z_threshold,
            mode,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_baseline(self, embedding: ArrayLike) -> None:
        """Feed a known-good embedding into the baseline statistics.

        Call this during corpus loading or when you trust the source.
        """
        emb = _to_float32(embedding)
        self._scorer.update(emb)

    def update_baseline_batch(self, embeddings: ArrayLike) -> None:
        """Feed a batch of known-good embeddings into the baseline."""
        arr = np.asarray(embeddings, dtype=np.float32)
        if arr.ndim == 1:
            self._scorer.update(arr)
            return
        for row in arr:
            self._scorer.update(row)

    def is_poisoned(
        self, embedding: ArrayLike
    ) -> Tuple[bool, float, str]:
        """Check whether *embedding* appears poisoned.

        Returns
        -------
        is_poisoned : bool
            ``True`` if the embedding is flagged as suspicious.
        confidence : float
            Detection confidence in ``[0.0, 1.0]``.
        reason : str
            Human-readable explanation (empty string if clean).
        """
        emb = _to_float32(embedding)

        if emb.size == 0:
            return (True, 1.0, "Empty embedding")

        anomaly_score, reason = self._scorer.score(emb)

        if reason:
            confidence = min(1.0, anomaly_score / (2 * self._z_threshold))
            return (True, confidence, reason)

        return (False, 0.0, "")

    def scan_batch(
        self, embeddings: ArrayLike, ids: Optional[Sequence[str]] = None
    ) -> List[Dict]:
        """Scan a batch of embeddings and return findings.

        Returns a list of dicts, one per flagged embedding::

            [{"id": "doc_42", "confidence": 0.97, "reason": "..."}]
        """
        arr = np.asarray(embeddings, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        findings = []
        for i, row in enumerate(arr):
            poisoned, confidence, reason = self.is_poisoned(row)
            if poisoned:
                finding = {
                    "index": i,
                    "id": ids[i] if ids else None,
                    "confidence": confidence,
                    "reason": reason,
                }
                findings.append(finding)
        return findings

    def get_stats(self) -> Dict:
        """Return current baseline and scorer statistics."""
        return {
            "sensitivity": self.sensitivity,
            "z_threshold": self._z_threshold,
            "mode": self.mode,
            "scorer": self._scorer.stats(),
        }

    def reset_baseline(self) -> None:
        """Clear all accumulated baseline data."""
        self._scorer.reset()
        logger.info("PoisoningDetector baseline reset")
