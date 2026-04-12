# Getting Started

## Installation

```bash
pip install embeddings-guardian
```

For backend-specific adapters, add the extra:

```bash
pip install embeddings-guardian[chromadb]    # ChromaDB
pip install embeddings-guardian[pinecone]    # Pinecone
pip install embeddings-guardian[weaviate]    # Weaviate
pip install embeddings-guardian[qdrant]      # Qdrant
pip install embeddings-guardian[milvus]      # Milvus
pip install embeddings-guardian[pgvector]    # PostgreSQL + pgvector
pip install embeddings-guardian[faiss]       # FAISS (testing)
```

## Core Concepts

### 1. Build a Baseline

The detector needs to learn what "normal" embeddings look like. Feed it your existing trusted corpus:

```python
from embeddings_guardian import PoisoningDetector

detector = PoisoningDetector(algorithm="magnitude", sensitivity=0.95)

# Option A: one at a time
for emb in trusted_embeddings:
    detector.update_baseline(emb)

# Option B: batch
detector.update_baseline_batch(all_trusted_embeddings)
```

### 2. Screen New Embeddings

```python
is_poisoned, confidence, reason = detector.is_poisoned(new_embedding)
```

### 3. Choose a Mode

- `warn` -- log detections but insert anyway (good for initial rollout)
- `quarantine` -- flag suspicious embeddings in metadata
- `strict` -- reject poisoned embeddings outright

```python
detector = PoisoningDetector(algorithm="magnitude", mode="quarantine")
```

## Per-Backend Setup

### ChromaDB

```python
import chromadb
from embeddings_guardian import PoisoningDetector
from embeddings_guardian.backends import ChromaDBAdapter

client = chromadb.Client()
collection = client.get_or_create_collection("my_docs")

detector = PoisoningDetector(sensitivity=0.95)
adapter = ChromaDBAdapter(collection=collection, detector=detector)

# Calibrate
baseline = adapter.get_baseline_embeddings(sample_size=1000)
detector.update_baseline_batch(baseline)

# Use adapter for all writes
adapter.upsert_embeddings(ids=ids, embeddings=vectors, metadata=meta)
```

### Pinecone

```python
from pinecone import Pinecone
from embeddings_guardian import PoisoningDetector
from embeddings_guardian.backends import PineconeAdapter

pc = Pinecone(api_key="your-key")
index = pc.Index("my-index")

detector = PoisoningDetector(sensitivity=0.95)
adapter = PineconeAdapter(index=index, detector=detector)

# Pinecone doesn't support bulk export, so provide baseline directly:
detector.update_baseline_batch(your_trusted_embeddings)

adapter.upsert_embeddings(ids=ids, embeddings=vectors)
```

### Weaviate

```python
import weaviate
from embeddings_guardian import PoisoningDetector
from embeddings_guardian.backends import WeaviateAdapter

client = weaviate.connect_to_local()
detector = PoisoningDetector(sensitivity=0.95)
adapter = WeaviateAdapter(client=client, class_name="Document", detector=detector)

baseline = adapter.get_baseline_embeddings()
detector.update_baseline_batch(baseline)

adapter.upsert_embeddings(ids=ids, embeddings=vectors)
```

### Qdrant

```python
from qdrant_client import QdrantClient
from embeddings_guardian import PoisoningDetector
from embeddings_guardian.backends import QdrantAdapter

client = QdrantClient(url="http://localhost:6333")
detector = PoisoningDetector(sensitivity=0.95)
adapter = QdrantAdapter(client=client, collection_name="docs", detector=detector)

baseline = adapter.get_baseline_embeddings()
detector.update_baseline_batch(baseline)

adapter.upsert_embeddings(ids=ids, embeddings=vectors)
```

### Milvus

```python
from pymilvus import connections, Collection
from embeddings_guardian import PoisoningDetector
from embeddings_guardian.backends import MilvusAdapter

connections.connect(host="localhost", port="19530")
collection = Collection("my_collection")

detector = PoisoningDetector(sensitivity=0.95)
adapter = MilvusAdapter(collection=collection, detector=detector)

baseline = adapter.get_baseline_embeddings()
detector.update_baseline_batch(baseline)

adapter.upsert_embeddings(ids=ids, embeddings=vectors)
```

### pgvector

```python
from embeddings_guardian import PoisoningDetector
from embeddings_guardian.backends import PgvectorAdapter

detector = PoisoningDetector(sensitivity=0.95)
adapter = PgvectorAdapter(
    connection_string="postgresql://user:pass@localhost/mydb",
    table_name="embeddings",
    detector=detector,
    dimension=1536,
)
adapter.connect()  # creates tables if needed

baseline = adapter.get_baseline_embeddings()
detector.update_baseline_batch(baseline)

adapter.upsert_embeddings(ids=ids, embeddings=vectors)
adapter.close()
```

### FAISS (Testing)

```python
import faiss
from embeddings_guardian import PoisoningDetector
from embeddings_guardian.backends import FAISSAdapter

index = faiss.IndexFlatL2(768)
detector = PoisoningDetector(sensitivity=0.95)
adapter = FAISSAdapter(index=index, dimension=768, detector=detector)

adapter.upsert_embeddings(ids=ids, embeddings=vectors)
```

## Tuning Sensitivity

| Sensitivity | Z-Threshold | False Positive Rate | Use Case |
|-------------|-------------|---------------------|----------|
| 0.90 | 2.0 | ~5% | Aggressive screening |
| 0.95 | 3.0 | ~1% | Balanced (recommended) |
| 0.99 | 4.0 | ~0.1% | Conservative |

Higher sensitivity means fewer false positives but more missed attacks.

## Batch Scanning

Scan an existing corpus for already-poisoned embeddings:

```python
findings = detector.scan_batch(all_embeddings, ids=all_ids)

for f in findings:
    print(f"Suspicious: {f['id']} (confidence: {f['confidence']:.0%})")
```

## Reporting

```python
from embeddings_guardian.utils.reporting import Reporter

reporter = Reporter()
reporter.add_findings_from_scan(findings)

# JSON report
print(reporter.json_str())

# CSV export
reporter.to_csv("findings.csv")
```

## Metrics

Track detection performance on labeled test data:

```python
from embeddings_guardian.utils.metrics import DetectionMetrics

metrics = DetectionMetrics()
for emb, is_actually_poisoned in labeled_test_set:
    predicted, conf, _ = detector.is_poisoned(emb)
    metrics.record(predicted=predicted, actual=is_actually_poisoned, confidence=conf)

print(f"F1: {metrics.f1_score():.3f}")
print(f"FPR: {metrics.false_positive_rate():.3f}")
```
