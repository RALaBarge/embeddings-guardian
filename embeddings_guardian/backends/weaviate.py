"""
Weaviate backend adapter.

Requires: ``pip install embeddings-guardian[weaviate]``

Usage::

    import weaviate
    from embeddings_guardian import PoisoningDetector
    from embeddings_guardian.backends import WeaviateAdapter

    client = weaviate.connect_to_local()
    detector = PoisoningDetector()
    adapter = WeaviateAdapter(client=client, class_name="Document", detector=detector)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from embeddings_guardian.core.adapters import VectorStoreAdapter
from embeddings_guardian.core.detector import PoisoningDetector

logger = logging.getLogger(__name__)


class WeaviateAdapter(VectorStoreAdapter):
    """Adapter for Weaviate (v4 client).

    Uses property-based quarantine within the same class.
    """

    name = "weaviate"

    def __init__(
        self,
        client: Any,
        class_name: str,
        detector: PoisoningDetector,
    ) -> None:
        self.client = client
        self.class_name = class_name
        self.detector = detector

    def upsert_embeddings(
        self,
        ids: Sequence[str],
        embeddings: np.ndarray,
        metadata: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if metadata is None:
            metadata = [{}] * len(ids)

        collection = self.client.collections.get(self.class_name)

        for i, (eid, emb) in enumerate(zip(ids, embeddings)):
            poisoned, confidence, reason = self.detector.is_poisoned(emb)
            props = dict(metadata[i])

            if poisoned:
                logger.warning("Poisoned embedding [%s]: %s", eid, reason)
                if self.detector.mode == "strict":
                    continue
                if self.detector.mode == "quarantine":
                    props["_eg_quarantined"] = True
                    props["_eg_reason"] = reason

            collection.data.insert(
                properties=props,
                uuid=eid,
                vector=emb.tolist(),
            )

    def query_embeddings(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
    ) -> List[Tuple[str, float]]:
        collection = self.client.collections.get(self.class_name)
        response = collection.query.near_vector(
            near_vector=query_embedding.tolist(),
            limit=k,
            return_metadata=["distance"],
        )
        return [
            (str(obj.uuid), obj.metadata.distance if obj.metadata else 0.0)
            for obj in response.objects
        ]

    def get_baseline_embeddings(self, sample_size: int = 1000) -> np.ndarray:
        collection = self.client.collections.get(self.class_name)
        response = collection.query.fetch_objects(
            limit=sample_size,
            include_vector=True,
        )
        vectors = [obj.vector for obj in response.objects if obj.vector is not None]
        if vectors:
            return np.array(vectors, dtype=np.float32)
        return np.empty((0, 0), dtype=np.float32)

    def quarantine_embedding(self, embedding_id: str, reason: str) -> None:
        collection = self.client.collections.get(self.class_name)
        collection.data.update(
            uuid=embedding_id,
            properties={"_eg_quarantined": True, "_eg_reason": reason},
        )

    def get_statistics(self) -> Dict[str, Any]:
        collection = self.client.collections.get(self.class_name)
        agg = collection.aggregate.over_all(total_count=True)
        return {
            "backend": self.name,
            "class_name": self.class_name,
            "total_count": agg.total_count if agg else 0,
        }
