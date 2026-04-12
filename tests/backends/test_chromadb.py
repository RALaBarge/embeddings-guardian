"""Tests for ChromaDB adapter (mocked -- no real ChromaDB needed)."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from embeddings_guardian import PoisoningDetector
from embeddings_guardian.backends.chromadb import ChromaDBAdapter


@pytest.fixture
def mock_collection():
    """Mock ChromaDB collection."""
    coll = MagicMock()
    coll.name = "test_collection"
    coll.count.return_value = 100
    coll.get.return_value = {
        "ids": [f"doc_{i}" for i in range(10)],
        "embeddings": np.random.default_rng(0).standard_normal((10, 64)).tolist(),
    }
    coll.query.return_value = {
        "ids": [["doc_1", "doc_2"]],
        "distances": [[0.1, 0.2]],
    }
    return coll


@pytest.fixture
def adapter(mock_collection):
    detector = PoisoningDetector(algorithm="magnitude", sensitivity=0.95)
    # Build baseline
    rng = np.random.default_rng(42)
    for _ in range(50):
        detector.update_baseline(rng.standard_normal(64).astype(np.float32))
    return ChromaDBAdapter(collection=mock_collection, detector=detector)


class TestChromaDBAdapter:
    def test_upsert_clean(self, adapter, mock_collection):
        rng = np.random.default_rng(1)
        embs = rng.standard_normal((3, 64)).astype(np.float32)
        adapter.upsert_embeddings(
            ids=["a", "b", "c"],
            embeddings=embs,
        )
        mock_collection.upsert.assert_called_once()
        call_args = mock_collection.upsert.call_args
        assert len(call_args.kwargs["ids"]) == 3

    def test_upsert_strict_rejects_poisoned(self, mock_collection):
        detector = PoisoningDetector(algorithm="magnitude", sensitivity=0.95, mode="strict")
        rng = np.random.default_rng(42)
        for _ in range(50):
            detector.update_baseline(rng.standard_normal(64).astype(np.float32))
        adapter = ChromaDBAdapter(collection=mock_collection, detector=detector)

        # One poisoned (extreme magnitude), two clean
        poisoned = rng.standard_normal(64).astype(np.float32) * 500
        clean = rng.standard_normal((2, 64)).astype(np.float32)
        batch = np.vstack([poisoned.reshape(1, -1), clean])

        adapter.upsert_embeddings(ids=["p", "c1", "c2"], embeddings=batch)
        call_args = mock_collection.upsert.call_args
        upserted_ids = call_args.kwargs["ids"]
        assert "p" not in upserted_ids

    def test_upsert_quarantine(self, mock_collection):
        detector = PoisoningDetector(algorithm="magnitude", sensitivity=0.95, mode="quarantine")
        rng = np.random.default_rng(42)
        for _ in range(50):
            detector.update_baseline(rng.standard_normal(64).astype(np.float32))
        adapter = ChromaDBAdapter(collection=mock_collection, detector=detector)

        poisoned = rng.standard_normal(64).astype(np.float32) * 500
        adapter.upsert_embeddings(ids=["p"], embeddings=poisoned.reshape(1, -1))
        # Should have called update for quarantine metadata
        mock_collection.update.assert_called()

    def test_query(self, adapter, mock_collection):
        q = np.ones(64, dtype=np.float32)
        results = adapter.query_embeddings(q, k=2)
        assert len(results) == 2
        assert results[0][0] == "doc_1"

    def test_get_baseline_embeddings(self, adapter, mock_collection):
        baseline = adapter.get_baseline_embeddings(sample_size=10)
        assert baseline.shape[0] == 10

    def test_get_statistics(self, adapter, mock_collection):
        stats = adapter.get_statistics()
        assert stats["backend"] == "chromadb"
        assert stats["document_count"] == 100

    def test_quarantine_embedding(self, adapter, mock_collection):
        adapter.quarantine_embedding("doc_99", "test reason")
        mock_collection.update.assert_called_once()

    def test_repr(self, adapter):
        assert "ChromaDBAdapter" in repr(adapter)

    def test_upsert_with_metadata(self, adapter, mock_collection):
        rng = np.random.default_rng(1)
        embs = rng.standard_normal((2, 64)).astype(np.float32)
        meta = [{"source": "a"}, {"source": "b"}]
        adapter.upsert_embeddings(ids=["x", "y"], embeddings=embs, metadata=meta)
        call_args = mock_collection.upsert.call_args
        assert call_args.kwargs["metadatas"] == meta

    def test_empty_collection_baseline(self, mock_collection):
        mock_collection.get.return_value = {"ids": [], "embeddings": []}
        detector = PoisoningDetector()
        adapter = ChromaDBAdapter(collection=mock_collection, detector=detector)
        baseline = adapter.get_baseline_embeddings()
        assert baseline.size == 0
