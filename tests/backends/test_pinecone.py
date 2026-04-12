"""Tests for Pinecone adapter (mocked -- no real Pinecone needed)."""

import numpy as np
import pytest
from unittest.mock import MagicMock

from embeddings_guardian import PoisoningDetector
from embeddings_guardian.backends.pinecone import PineconeAdapter


@pytest.fixture
def mock_index():
    idx = MagicMock()
    idx.query.return_value = {
        "matches": [
            {"id": "v1", "score": 0.95},
            {"id": "v2", "score": 0.88},
        ],
    }
    idx.describe_index_stats.return_value = {
        "total_vector_count": 5000,
        "dimension": 64,
    }
    return idx


@pytest.fixture
def adapter(mock_index):
    detector = PoisoningDetector(algorithm="magnitude")
    rng = np.random.default_rng(42)
    for _ in range(50):
        detector.update_baseline(rng.standard_normal(64).astype(np.float32))
    return PineconeAdapter(index=mock_index, detector=detector)


class TestPineconeAdapter:
    def test_upsert_clean(self, adapter, mock_index):
        rng = np.random.default_rng(1)
        embs = rng.standard_normal((3, 64)).astype(np.float32)
        adapter.upsert_embeddings(ids=["a", "b", "c"], embeddings=embs)
        mock_index.upsert.assert_called_once()

    def test_upsert_strict_filters(self, mock_index):
        detector = PoisoningDetector(algorithm="magnitude", mode="strict")
        rng = np.random.default_rng(42)
        for _ in range(50):
            detector.update_baseline(rng.standard_normal(64).astype(np.float32))
        adapter = PineconeAdapter(index=mock_index, detector=detector)

        poisoned = rng.standard_normal(64).astype(np.float32) * 500
        clean = rng.standard_normal(64).astype(np.float32)
        batch = np.vstack([poisoned.reshape(1, -1), clean.reshape(1, -1)])
        adapter.upsert_embeddings(ids=["p", "c"], embeddings=batch)

        call_args = mock_index.upsert.call_args
        vectors = call_args.kwargs["vectors"]
        upserted_ids = [v["id"] for v in vectors]
        assert "p" not in upserted_ids

    def test_upsert_quarantine_adds_metadata(self, mock_index):
        detector = PoisoningDetector(algorithm="magnitude", mode="quarantine")
        rng = np.random.default_rng(42)
        for _ in range(50):
            detector.update_baseline(rng.standard_normal(64).astype(np.float32))
        adapter = PineconeAdapter(index=mock_index, detector=detector)

        poisoned = rng.standard_normal(64).astype(np.float32) * 500
        adapter.upsert_embeddings(ids=["p"], embeddings=poisoned.reshape(1, -1))

        call_args = mock_index.upsert.call_args
        vectors = call_args.kwargs["vectors"]
        assert vectors[0]["metadata"].get("__eg_poisoned") is True

    def test_query(self, adapter, mock_index):
        q = np.ones(64, dtype=np.float32)
        results = adapter.query_embeddings(q, k=2)
        assert len(results) == 2
        assert results[0] == ("v1", 0.95)

    def test_get_baseline_warns(self, adapter):
        baseline = adapter.get_baseline_embeddings()
        assert baseline.size == 0

    def test_get_statistics(self, adapter, mock_index):
        stats = adapter.get_statistics()
        assert stats["backend"] == "pinecone"
        assert stats["total_vector_count"] == 5000

    def test_quarantine(self, adapter, mock_index):
        adapter.quarantine_embedding("v1", "test")
        mock_index.update.assert_called_once()

    def test_namespace(self, mock_index):
        detector = PoisoningDetector()
        adapter = PineconeAdapter(index=mock_index, detector=detector, namespace="prod")
        assert adapter.namespace == "prod"

    def test_upsert_with_metadata(self, adapter, mock_index):
        rng = np.random.default_rng(1)
        embs = rng.standard_normal((2, 64)).astype(np.float32)
        meta = [{"src": "a"}, {"src": "b"}]
        adapter.upsert_embeddings(ids=["x", "y"], embeddings=embs, metadata=meta)
        call_args = mock_index.upsert.call_args
        vectors = call_args.kwargs["vectors"]
        assert vectors[0]["metadata"]["src"] == "a"

    def test_empty_query_result(self, mock_index):
        mock_index.query.return_value = {"matches": []}
        detector = PoisoningDetector()
        adapter = PineconeAdapter(index=mock_index, detector=detector)
        results = adapter.query_embeddings(np.ones(64, dtype=np.float32))
        assert results == []
