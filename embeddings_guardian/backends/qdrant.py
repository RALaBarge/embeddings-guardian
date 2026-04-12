"""
Qdrant backend adapter.

Requires: ``pip install embeddings-guardian[qdrant]``

Usage::

    from qdrant_client import QdrantClient
    from embeddings_guardian import PoisoningDetector
    from embeddings_guardian.backends import QdrantAdapter

    client = QdrantClient(url="http://localhost:6333")
    detector = PoisoningDetector()
    adapter = QdrantAdapter(client=client, collection_name="docs", detector=detector)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from embeddings_guardian.core.adapters import VectorStoreAdapter
from embeddings_guardian.core.detector import PoisoningDetector

logger = logging.getLogger(__name__)


class QdrantAdapter(VectorStoreAdapter):
    """Adapter for Qdrant (REST or gRPC).

    Uses payload-based quarantine.
    """

    name = "qdrant"

    def __init__(
        self,
        client: Any,
        collection_name: str,
        detector: PoisoningDetector,
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.detector = detector

    def upsert_embeddings(
        self,
        ids: Sequence[str],
        embeddings: np.ndarray,
        metadata: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        from qdrant_client.models import PointStruct

        embeddings = np.asarray(embeddings, dtype=np.float32)
        if metadata is None:
            metadata = [{}] * len(ids)

        points = []
        for i, (eid, emb) in enumerate(zip(ids, embeddings)):
            poisoned, confidence, reason = self.detector.is_poisoned(emb)
            payload = dict(metadata[i])

            if poisoned:
                logger.warning("Poisoned embedding [%s]: %s", eid, reason)
                if self.detector.mode == "strict":
                    continue
                if self.detector.mode == "quarantine":
                    payload["_eg_quarantined"] = True
                    payload["_eg_reason"] = reason

            points.append(PointStruct(id=eid, vector=emb.tolist(), payload=payload))

        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)

    def query_embeddings(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
    ) -> List[Tuple[str, float]]:
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=k,
        )
        return [(str(hit.id), hit.score) for hit in results]

    def get_baseline_embeddings(self, sample_size: int = 1000) -> np.ndarray:
        results, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=sample_size,
            with_vectors=True,
        )
        vectors = [p.vector for p in results if p.vector is not None]
        if vectors:
            return np.array(vectors, dtype=np.float32)
        return np.empty((0, 0), dtype=np.float32)

    def quarantine_embedding(self, embedding_id: str, reason: str) -> None:
        self.client.set_payload(
            collection_name=self.collection_name,
            payload={"_eg_quarantined": True, "_eg_reason": reason},
            points=[embedding_id],
        )

    def get_statistics(self) -> Dict[str, Any]:
        info = self.client.get_collection(self.collection_name)
        return {
            "backend": self.name,
            "collection_name": self.collection_name,
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
        }
