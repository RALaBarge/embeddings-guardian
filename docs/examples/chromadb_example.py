"""
ChromaDB integration example.

Demonstrates:
- Setting up a ChromaDB adapter with poisoning detection
- Building a baseline from existing data
- Screening new embeddings on upsert
- Scanning an existing corpus for poisoned documents
"""

import chromadb
import numpy as np

from embeddings_guardian import PoisoningDetector
from embeddings_guardian.backends import ChromaDBAdapter
from embeddings_guardian.utils.reporting import Reporter

# 1. Set up ChromaDB
client = chromadb.Client()
collection = client.get_or_create_collection("knowledge_base")

# 2. Initialize detector and adapter
detector = PoisoningDetector(
    algorithm="magnitude",
    sensitivity=0.95,
    mode="quarantine",  # flag suspicious embeddings instead of rejecting
)
adapter = ChromaDBAdapter(collection=collection, detector=detector)

# 3. Build baseline from existing trusted embeddings
baseline = adapter.get_baseline_embeddings(sample_size=1000)
if baseline.size > 0:
    detector.update_baseline_batch(baseline)
    print(f"Baseline built from {baseline.shape[0]} embeddings")
else:
    print("No existing embeddings found -- baseline will build as documents are added")

# 4. Add new documents with automatic screening
new_ids = ["doc_001", "doc_002", "doc_003"]
new_embeddings = np.random.default_rng(42).standard_normal((3, 768)).astype(np.float32)
new_metadata = [
    {"source": "internal", "topic": "security"},
    {"source": "internal", "topic": "architecture"},
    {"source": "external", "topic": "unknown"},  # could be suspicious
]

adapter.upsert_embeddings(
    ids=new_ids,
    embeddings=new_embeddings,
    metadata=new_metadata,
)
print("Upsert complete -- poisoned embeddings quarantined automatically")

# 5. Scan existing corpus
all_embeddings = adapter.get_baseline_embeddings(sample_size=5000)
if all_embeddings.size > 0:
    findings = detector.scan_batch(all_embeddings)
    reporter = Reporter()
    reporter.add_findings_from_scan(findings)
    print(reporter.json_str())

# 6. Check statistics
stats = adapter.get_statistics()
print(f"Collection: {stats['collection_name']}, Documents: {stats['document_count']}")
