"""
Vector store backend adapters.

Each adapter connects to a specific vector database and implements the
:class:`~embeddings_guardian.core.adapters.VectorStoreAdapter` interface.

Install the backend you need::

    pip install embeddings-guardian[chromadb]
    pip install embeddings-guardian[pinecone]
    pip install embeddings-guardian[weaviate]
    pip install embeddings-guardian[qdrant]
    pip install embeddings-guardian[milvus]
    pip install embeddings-guardian[pgvector]
    pip install embeddings-guardian[faiss]
"""

# Lazy imports so missing optional deps don't blow up at import time.


def __getattr__(name: str):
    _map = {
        "ChromaDBAdapter": "embeddings_guardian.backends.chromadb",
        "PineconeAdapter": "embeddings_guardian.backends.pinecone",
        "WeaviateAdapter": "embeddings_guardian.backends.weaviate",
        "QdrantAdapter": "embeddings_guardian.backends.qdrant",
        "MilvusAdapter": "embeddings_guardian.backends.milvus",
        "PgvectorAdapter": "embeddings_guardian.backends.pgvector",
        "FAISSAdapter": "embeddings_guardian.backends.faiss",
    }
    if name in _map:
        import importlib

        mod = importlib.import_module(_map[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
