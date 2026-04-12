# embeddings-guardian

**Open-source RAG poisoning detection. OWASP LLM08:2025 reference implementation.**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![PyPI](https://img.shields.io/pypi/v/embeddings-guardian.svg)](https://pypi.org/project/embeddings-guardian/)

---

## What is RAG Poisoning?

Retrieval-Augmented Generation (RAG) systems retrieve documents from a vector database and feed them into an LLM as context. **RAG poisoning** is an attack where adversarial documents are injected into the vector store with crafted embeddings designed to:

- Appear relevant to targeted queries (high cosine similarity)
- Contain malicious, misleading, or policy-violating content
- Hijack the LLM's output for specific topics

Research from Zou et al. (*PoisonedRAG*, 2024) demonstrated **97--99% attack success rates** against production RAG systems, with poisoned documents retrieved in the top-k results for targeted queries. The attack requires injecting as few as 5 documents into a corpus of thousands.

This is tracked as [OWASP LLM08:2025 -- Vector and Embedding Weaknesses](https://genai.owasp.org/).

## Why It Matters

- **97--99% attack success rate** in academic evaluations
- **5 poisoned documents** can compromise a corpus of 10,000+
- **No built-in defenses** in ChromaDB, Pinecone, Weaviate, or any major vector DB
- **Compliance risk** for healthcare, finance, and regulated industries
- **Silent failure** -- poisoned results look normal to end users

## Quick Start (2 Minutes)

```bash
pip install embeddings-guardian
```

```python
from embeddings_guardian import PoisoningDetector

# Initialize detector
detector = PoisoningDetector(algorithm="magnitude", sensitivity=0.95)

# Build baseline from your existing (trusted) embeddings
for embedding in your_corpus_embeddings:
    detector.update_baseline(embedding)

# Check new embeddings before inserting
is_poisoned, confidence, reason = detector.is_poisoned(new_embedding)

if is_poisoned:
    print(f"BLOCKED: {reason} (confidence: {confidence:.0%})")
else:
    # Safe to insert into your vector store
    collection.add(embedding=new_embedding)
```

## Installation

**Core only** (numpy + scikit-learn):
```bash
pip install embeddings-guardian
```

**With a specific backend**:
```bash
pip install embeddings-guardian[chromadb]
pip install embeddings-guardian[pinecone]
pip install embeddings-guardian[weaviate]
pip install embeddings-guardian[qdrant]
pip install embeddings-guardian[milvus]
pip install embeddings-guardian[pgvector]
pip install embeddings-guardian[faiss]
```

**All backends**:
```bash
pip install embeddings-guardian[all]
```

## Backend Integration

Use an adapter to get automatic pre-insert screening:

```python
import chromadb
from embeddings_guardian import PoisoningDetector
from embeddings_guardian.backends import ChromaDBAdapter

client = chromadb.Client()
collection = client.get_or_create_collection("documents")

detector = PoisoningDetector(algorithm="magnitude", sensitivity=0.95)
adapter = ChromaDBAdapter(collection=collection, detector=detector)

# Calibrate from existing data
baseline = adapter.get_baseline_embeddings(sample_size=1000)
detector.update_baseline_batch(baseline)

# Upsert with automatic screening
adapter.upsert_embeddings(
    ids=["doc_1", "doc_2"],
    embeddings=vectors,
    metadata=[{"source": "trusted"}, {"source": "trusted"}],
)
```

## Detection Algorithms

| Algorithm | Speed | Best For | How It Works |
|-----------|-------|----------|-------------|
| **magnitude** | <0.5ms | Real-time screening | L2 norm z-score analysis |
| **centroid** | ~1ms | Semantic outliers | Cosine distance to cluster centroids |
| **neighborhood** | 5--10ms | Sophisticated attacks | k-NN density analysis |
| **dimension** | <0.5ms | Secondary checks | Per-dimension z-score |
| **ensemble** | varies | Maximum accuracy | Weighted combination of algorithms |

## Detection Modes

| Mode | Behavior |
|------|----------|
| `warn` | Log the detection, insert anyway |
| `quarantine` | Flag in metadata, exclude from queries |
| `strict` | Reject the embedding outright |

## Supported Vector Stores

| Store | Status | Install |
|-------|--------|---------|
| ChromaDB | Full | `[chromadb]` |
| Pinecone | Full | `[pinecone]` |
| Weaviate | Full | `[weaviate]` |
| Qdrant | Full | `[qdrant]` |
| Milvus | Full | `[milvus]` |
| pgvector | Full | `[pgvector]` |
| FAISS | Testing/Research | `[faiss]` |

## Documentation

- [Getting Started](docs/getting_started.md) -- per-backend setup and configuration
- [Backends Guide](docs/backends.md) -- performance, cost, and quirks per store
- [Algorithms](docs/algorithms.md) -- detailed explanation of detection methods
- [Examples](docs/examples/) -- real-world code samples

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, and pull request guidelines.

## License

Apache License 2.0. See [LICENSE](LICENSE).

## References

- Zou et al., "PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models," 2024
- OWASP Top 10 for LLM Applications 2025 -- LLM08: Vector and Embedding Weaknesses
- Xue et al., "BadRAG: Identifying Vulnerabilities in Retrieval Augmented Generation of Large Language Models," 2024
- Zeng et al., "RevPRAG: Reverse Prompt Engineering for RAG Poisoning Detection," 2024
