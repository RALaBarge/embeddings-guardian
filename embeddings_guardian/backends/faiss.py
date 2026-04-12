"""
FAISS backend adapter (in-memory, file-based index).

Requires: ``pip install embeddings-guardian[faiss]``

Primarily intended for testing and research.  Data is lost on restart
unless the index is saved to disk.

Usage::

    import faiss
    from embeddings_guardian import PoisoningDetector
    from embeddings_guardian.backends import FAISSAdapter

    index = faiss.IndexFlatL2(768)
    detector = PoisoningDetector()
    adapter = FAISSAdapter(index=index, dimension=768, detector=detector)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from embeddings_guardian.core.adapters import VectorStoreAdapter
from embeddings_guardian.core.detector import PoisoningDetector

logger = logging.getLogger(__name__)


class FAISSAdapter(VectorStoreAdapter):
    """In-memory FAISS index adapter.

    FAISS does not natively support string IDs or metadata, so this
    adapter maintains a parallel mapping.
    """

    name = "faiss"

    def __init__(
        self,
        index: Any,
        dimension: int,
        detector: PoisoningDetector,
    ) -> None:
        self.index = index
        self.dimension = dimension
        self.detector = detector

        # Parallel bookkeeping (FAISS uses integer IDs internally)
        self._id_map: Dict[int, str] = {}
        self._reverse_map: Dict[str, int] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._embeddings: Dict[str, np.ndarray] = {}
        self._quarantined: Dict[str, str] = {}
        self._next_idx: int = 0

    def upsert_embeddings(
        self,
        ids: Sequence[str],
        embeddings: np.ndarray,
        metadata: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if metadata is None:
            metadata = [{}] * len(ids)

        clean_embs = []
        clean_ids = []

        for i, (eid, emb) in enumerate(zip(ids, embeddings)):
            poisoned, confidence, reason = self.detector.is_poisoned(emb)

            if poisoned:
                logger.warning("Poisoned embedding [%s]: %s", eid, reason)
                if self.detector.mode == "strict":
                    continue
                if self.detector.mode == "quarantine":
                    self._quarantined[eid] = reason
                    continue

            clean_embs.append(emb)
            clean_ids.append(eid)
            self._metadata[eid] = metadata[i]
            self._embeddings[eid] = emb

        if clean_embs:
            batch = np.vstack(clean_embs)
            self.index.add(batch)
            for eid in clean_ids:
                idx = self._next_idx
                self._id_map[idx] = eid
                self._reverse_map[eid] = idx
                self._next_idx += 1

    def query_embeddings(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
    ) -> List[Tuple[str, float]]:
        query = query_embedding.reshape(1, -1).astype(np.float32)
        distances, indices = self.index.search(query, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx in self._id_map:
                results.append((self._id_map[idx], float(dist)))
        return results

    def get_baseline_embeddings(self, sample_size: int = 1000) -> np.ndarray:
        all_embs = list(self._embeddings.values())
        if not all_embs:
            return np.empty((0, self.dimension), dtype=np.float32)
        selected = all_embs[:sample_size]
        return np.vstack(selected).astype(np.float32)

    def quarantine_embedding(self, embedding_id: str, reason: str) -> None:
        self._quarantined[embedding_id] = reason
        logger.info("Quarantined %s: %s", embedding_id, reason)

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "backend": self.name,
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "quarantined_count": len(self._quarantined),
            "metadata_count": len(self._metadata),
        }
