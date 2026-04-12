"""
Pinecone integration example.

Demonstrates setting up Pinecone with embeddings-guardian for
production-scale RAG poisoning detection.
"""

import numpy as np

from embeddings_guardian import PoisoningDetector
from embeddings_guardian.backends import PineconeAdapter

# NOTE: Requires pinecone-client>=3.0
# pip install embeddings-guardian[pinecone]
from pinecone import Pinecone

# 1. Connect to Pinecone
pc = Pinecone(api_key="your-api-key-here")
index = pc.Index("my-rag-index")

# 2. Initialize detector
detector = PoisoningDetector(
    algorithm="magnitude",
    sensitivity=0.95,
    mode="quarantine",
)

# 3. Build baseline from your embedding pipeline
# Pinecone doesn't support bulk export, so supply embeddings directly
trusted_embeddings = np.load("trusted_corpus_embeddings.npy")
detector.update_baseline_batch(trusted_embeddings)
print(f"Baseline: {detector.get_stats()['scorer']['count']} vectors")

# 4. Create adapter
adapter = PineconeAdapter(
    index=index,
    detector=detector,
    namespace="production",
)

# 5. Upsert with screening
new_ids = ["article_42", "article_43"]
new_embeddings = np.random.default_rng(0).standard_normal((2, 1536)).astype(np.float32)
new_metadata = [
    {"title": "Secure RAG Patterns", "source": "blog"},
    {"title": "Unknown Document", "source": "crawl"},
]

adapter.upsert_embeddings(
    ids=new_ids,
    embeddings=new_embeddings,
    metadata=new_metadata,
)

# 6. Statistics
stats = adapter.get_statistics()
print(f"Total vectors: {stats['total_vector_count']}")
