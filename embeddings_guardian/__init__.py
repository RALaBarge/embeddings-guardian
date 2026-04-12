"""
embeddings-guardian -- Open-source RAG poisoning detection.

Detect poisoned embeddings in vector databases before they corrupt
your retrieval-augmented generation pipeline. OWASP LLM08:2025
reference implementation.

Basic usage::

    from embeddings_guardian import PoisoningDetector

    detector = PoisoningDetector(algorithm="magnitude", sensitivity=0.95)

    # Build baseline from known-good embeddings
    for emb in corpus_embeddings:
        detector.update_baseline(emb)

    # Check new embeddings
    is_poisoned, confidence, reason = detector.is_poisoned(new_embedding)
"""

from embeddings_guardian.core.detector import PoisoningDetector
from embeddings_guardian.core.adapters import VectorStoreAdapter
from embeddings_guardian.core.scoring import (
    BaseScorer,
    MagnitudeScorer,
    CentroidScorer,
    NeighborhoodScorer,
    DimensionScorer,
    EnsembleScorer,
)

__version__ = "0.1.0"
__all__ = [
    "PoisoningDetector",
    "VectorStoreAdapter",
    "BaseScorer",
    "MagnitudeScorer",
    "CentroidScorer",
    "NeighborhoodScorer",
    "DimensionScorer",
    "EnsembleScorer",
]
