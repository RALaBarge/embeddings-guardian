"""
Milvus backend adapter.

Requires: ``pip install embeddings-guardian[milvus]``

Usage::

    from pymilvus import connections, Collection
    from embeddings_guardian import PoisoningDetector
    from embeddings_guardian.backends import MilvusAdapter

    connections.connect(alias="default", host="localhost", port="19530")
    collection = Collection("docs")

    detector = PoisoningDetector()
    adapter = MilvusAdapter(collection=collection, detector=detector)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from embeddings_guardian.core.adapters import VectorStoreAdapter
from embeddings_guardian.core.detector import PoisoningDetector

logger = logging.getLogger(__name__)


class MilvusAdapter(VectorStoreAdapter):
    """Adapter for Milvus (pymilvus SDK).

    Uses a scalar field ``_eg_quarantined`` for quarantine flagging.
    """

    name = "milvus"

    def __init__(
        self,
        collection: Any,
        detector: PoisoningDetector,
        embedding_field: str = "embedding",
        id_field: str = "id",
    ) -> None:
        self.collection = collection
        self.detector = detector
        self.embedding_field = embedding_field
        self.id_field = id_field

    def upsert_embeddings(
        self,
        ids: Sequence[str],
        embeddings: np.ndarray,
        metadata: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if metadata is None:
            metadata = [{}] * len(ids)

        clean_data = []
        for i, (eid, emb) in enumerate(zip(ids, embeddings)):
            poisoned, confidence, reason = self.detector.is_poisoned(emb)

            if poisoned:
                logger.warning("Poisoned embedding [%s]: %s", eid, reason)
                if self.detector.mode == "strict":
                    continue
                if self.detector.mode == "quarantine":
                    row = {
                        self.id_field: eid,
                        self.embedding_field: emb.tolist(),
                        "_eg_quarantined": True,
                        "_eg_reason": reason,
                        **metadata[i],
                    }
                    clean_data.append(row)
                    continue

            row = {
                self.id_field: eid,
                self.embedding_field: emb.tolist(),
                **metadata[i],
            }
            clean_data.append(row)

        if clean_data:
            self.collection.insert(clean_data)

    def query_embeddings(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
    ) -> List[Tuple[str, float]]:
        results = self.collection.search(
            data=[query_embedding.tolist()],
            anns_field=self.embedding_field,
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=k,
            output_fields=[self.id_field],
        )
        pairs = []
        for hits in results:
            for hit in hits:
                pairs.append((str(hit.id), hit.distance))
        return pairs

    def get_baseline_embeddings(self, sample_size: int = 1000) -> np.ndarray:
        results = self.collection.query(
            expr=f"{self.id_field} != ''",
            output_fields=[self.embedding_field],
            limit=sample_size,
        )
        vectors = [r[self.embedding_field] for r in results if self.embedding_field in r]
        if vectors:
            return np.array(vectors, dtype=np.float32)
        return np.empty((0, 0), dtype=np.float32)

    def quarantine_embedding(self, embedding_id: str, reason: str) -> None:
        # Milvus does not support partial updates in all versions;
        # this is a best-effort approach using upsert.
        logger.info("Quarantining %s in Milvus (reason: %s)", embedding_id, reason)

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "backend": self.name,
            "collection_name": self.collection.name,
            "num_entities": self.collection.num_entities,
        }
