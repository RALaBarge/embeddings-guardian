"""
Abstract base class for vector store adapters.

Every backend adapter (ChromaDB, Pinecone, Weaviate, ...) inherits from
:class:`VectorStoreAdapter` and implements the five required methods.
This lets :class:`~embeddings_guardian.core.detector.PoisoningDetector`
work identically regardless of the underlying vector database.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


class VectorStoreAdapter(ABC):
    """Protocol that all vector-store backends must implement.

    Parameters
    ----------
    name : str
        Human-readable adapter name (e.g. ``"chromadb"``).
    """

    name: str = "base"

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    @abstractmethod
    def upsert_embeddings(
        self,
        ids: Sequence[str],
        embeddings: np.ndarray,
        metadata: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        """Store embeddings (and optional metadata) keyed by *ids*.

        Implementations should call the detector's pre-check hook before
        persisting, and quarantine flagged vectors when configured to do so.
        """

    @abstractmethod
    def query_embeddings(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Return the *k* nearest neighbors as ``(id, distance)`` pairs."""

    # ------------------------------------------------------------------
    # Baseline & quarantine
    # ------------------------------------------------------------------

    @abstractmethod
    def get_baseline_embeddings(self, sample_size: int = 1000) -> np.ndarray:
        """Sample existing embeddings for baseline calibration.

        Returns an ``(n, d)`` float32 array where ``n <= sample_size``.
        """

    @abstractmethod
    def quarantine_embedding(self, embedding_id: str, reason: str) -> None:
        """Flag or move a suspicious embedding so it is excluded from queries."""

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Return store-specific stats (doc count, dimensions, etc.)."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Open / verify the connection.  Override if the backend needs it."""

    def close(self) -> None:
        """Release resources.  Override if needed."""

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
