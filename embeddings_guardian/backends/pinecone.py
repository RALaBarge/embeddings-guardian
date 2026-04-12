"""
Pinecone backend adapter.

Requires: ``pip install embeddings-guardian[pinecone]``

Usage::

    from pinecone import Pinecone
    from embeddings_guardian import PoisoningDetector
    from embeddings_guardian.backends import PineconeAdapter

    pc = Pinecone(api_key="...")
    index = pc.Index("my-index")

    detector = PoisoningDetector()
    adapter = PineconeAdapter(index=index, detector=detector)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from embeddings_guardian.core.adapters import VectorStoreAdapter
from embeddings_guardian.core.detector import PoisoningDetector

logger = logging.getLogger(__name__)


class PineconeAdapter(VectorStoreAdapter):
    """Adapter for Pinecone serverless / pod indexes.

    Uses metadata-based quarantine (no separate table).
    """

    name = "pinecone"

    def __init__(
        self,
        index: Any,
        detector: PoisoningDetector,
        namespace: str = "",
    ) -> None:
        self.index = index
        self.detector = detector
        self.namespace = namespace

    def upsert_embeddings(
        self,
        ids: Sequence[str],
        embeddings: np.ndarray,
        metadata: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if metadata is None:
            metadata = [{}] * len(ids)

        vectors = []
        for i, (eid, emb) in enumerate(zip(ids, embeddings)):
            poisoned, confidence, reason = self.detector.is_poisoned(emb)
            meta = dict(metadata[i])
            if poisoned:
                logger.warning("Poisoned embedding [%s]: %s", eid, reason)
                if self.detector.mode == "strict":
                    continue
                if self.detector.mode == "quarantine":
                    meta["__eg_poisoned"] = True
                    meta["__eg_reason"] = reason
            vectors.append({"id": eid, "values": emb.tolist(), "metadata": meta})

        if vectors:
            self.index.upsert(vectors=vectors, namespace=self.namespace)

    def query_embeddings(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
    ) -> List[Tuple[str, float]]:
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=k,
            namespace=self.namespace,
        )
        return [(m["id"], m.get("score", 0.0)) for m in results.get("matches", [])]

    def get_baseline_embeddings(self, sample_size: int = 1000) -> np.ndarray:
        # Pinecone does not support listing all vectors easily;
        # callers should provide baseline embeddings directly.
        logger.warning(
            "Pinecone does not support efficient bulk export. "
            "Supply baseline embeddings via detector.update_baseline_batch()."
        )
        return np.empty((0, 0), dtype=np.float32)

    def quarantine_embedding(self, embedding_id: str, reason: str) -> None:
        self.index.update(
            id=embedding_id,
            set_metadata={"__eg_poisoned": True, "__eg_reason": reason},
            namespace=self.namespace,
        )

    def get_statistics(self) -> Dict[str, Any]:
        stats = self.index.describe_index_stats()
        return {
            "backend": self.name,
            "namespace": self.namespace,
            "total_vector_count": stats.get("total_vector_count", 0),
            "dimension": stats.get("dimension", 0),
        }
