"""
ChromaDB backend adapter.

Requires: ``pip install embeddings-guardian[chromadb]``

Usage::

    import chromadb
    from embeddings_guardian import PoisoningDetector
    from embeddings_guardian.backends import ChromaDBAdapter

    client = chromadb.Client()
    collection = client.get_or_create_collection("docs")

    detector = PoisoningDetector()
    adapter = ChromaDBAdapter(collection=collection, detector=detector)

    # Upsert with automatic poisoning checks
    adapter.upsert_embeddings(ids=["d1"], embeddings=vectors)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from embeddings_guardian.core.adapters import VectorStoreAdapter
from embeddings_guardian.core.detector import PoisoningDetector

logger = logging.getLogger(__name__)


class ChromaDBAdapter(VectorStoreAdapter):
    """Adapter for ChromaDB collections.

    Parameters
    ----------
    collection
        A ``chromadb.Collection`` instance.
    detector
        A :class:`PoisoningDetector` used to screen embeddings.
    quarantine_prefix : str
        Metadata key prefix for quarantined embeddings.
    """

    name = "chromadb"

    def __init__(
        self,
        collection: Any,
        detector: PoisoningDetector,
        quarantine_prefix: str = "__eg_quarantine",
    ) -> None:
        self.collection = collection
        self.detector = detector
        self.quarantine_prefix = quarantine_prefix

    def upsert_embeddings(
        self,
        ids: Sequence[str],
        embeddings: np.ndarray,
        metadata: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if metadata is None:
            metadata = [{}] * len(ids)

        clean_ids, clean_embs, clean_meta = [], [], []

        for i, (eid, emb) in enumerate(zip(ids, embeddings)):
            poisoned, confidence, reason = self.detector.is_poisoned(emb)
            if poisoned:
                logger.warning(
                    "Poisoned embedding detected [%s]: %s (confidence=%.2f)",
                    eid, reason, confidence,
                )
                if self.detector.mode == "quarantine":
                    self.quarantine_embedding(eid, reason)
                    continue
                elif self.detector.mode == "strict":
                    continue
                # mode == "warn": log but still insert
            clean_ids.append(eid)
            clean_embs.append(emb.tolist())
            clean_meta.append(metadata[i])

        if clean_ids:
            self.collection.upsert(
                ids=clean_ids,
                embeddings=clean_embs,
                metadatas=clean_meta,
            )

    def query_embeddings(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
    ) -> List[Tuple[str, float]]:
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
        )
        pairs = []
        if results and results.get("ids"):
            ids_list = results["ids"][0]
            dists = results.get("distances", [[0.0] * len(ids_list)])[0]
            pairs = list(zip(ids_list, dists))
        return pairs

    def get_baseline_embeddings(self, sample_size: int = 1000) -> np.ndarray:
        results = self.collection.get(
            limit=sample_size,
            include=["embeddings"],
        )
        if results and results.get("embeddings"):
            return np.array(results["embeddings"], dtype=np.float32)
        return np.empty((0, 0), dtype=np.float32)

    def quarantine_embedding(self, embedding_id: str, reason: str) -> None:
        self.collection.update(
            ids=[embedding_id],
            metadatas=[{
                f"{self.quarantine_prefix}": True,
                f"{self.quarantine_prefix}_reason": reason,
            }],
        )
        logger.info("Quarantined embedding %s: %s", embedding_id, reason)

    def get_statistics(self) -> Dict[str, Any]:
        count = self.collection.count()
        return {
            "backend": self.name,
            "document_count": count,
            "collection_name": getattr(self.collection, "name", "unknown"),
        }
