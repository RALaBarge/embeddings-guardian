# Backend Adapters Guide

## Overview

Each adapter wraps a vector database client and adds poisoning detection
as a pre-insert screening step. All adapters implement the same
`VectorStoreAdapter` interface, so switching backends requires changing
only the adapter class.

## Comparison

| Backend | Deployment | Baseline Export | Quarantine Method | Latency Overhead |
|---------|-----------|-----------------|-------------------|------------------|
| **ChromaDB** | Self-hosted | `collection.get()` | Metadata flag | <1ms |
| **Pinecone** | Managed SaaS | Manual (no bulk export) | Metadata flag | <1ms |
| **Weaviate** | Self-hosted / Cloud | `fetch_objects()` | Property flag | <1ms |
| **Qdrant** | Self-hosted / Cloud | `scroll()` | Payload flag | <1ms |
| **Milvus** | Self-hosted / Cloud | `query()` | Scalar field | <1ms |
| **pgvector** | PostgreSQL | SQL `SELECT` | Quarantine table | <1ms |
| **FAISS** | In-memory | Direct access | In-memory dict | <1ms |

## ChromaDB

**Best for:** Local development, small-to-medium corpora.

- Native Python client, no authentication required for local mode.
- `get_baseline_embeddings()` pulls directly from the collection.
- Quarantine uses metadata fields (`__eg_quarantine: true`).
- No batch size limits (handles in-process).

**Quirks:**
- ChromaDB stores embeddings as Python lists internally; the adapter handles
  numpy conversion.
- Collection `count()` can be slow on very large collections.

## Pinecone

**Best for:** Managed serverless, production at scale.

- REST API with API key authentication.
- Does not support efficient bulk vector export -- you must supply baseline
  embeddings from your embedding pipeline directly.
- Quarantine uses metadata tags (`__eg_poisoned: true`).
- Namespace support for multi-tenant isolation.

**Quirks:**
- Rate limits apply to upsert and query operations.
- Metadata values must be strings, numbers, booleans, or lists of strings.

## Weaviate

**Best for:** Self-hosted with GraphQL flexibility.

- v4 Python client with `connect_to_local()` or `connect_to_wcs()`.
- Baseline export via `fetch_objects(include_vector=True)`.
- Quarantine uses custom properties (`_eg_quarantined`).
- Schema must allow extra properties or use a generic class.

**Quirks:**
- Object UUIDs must be valid UUIDs (not arbitrary strings).
- GraphQL aggregation for statistics can be slow on large classes.

## Qdrant

**Best for:** High-performance self-hosted or cloud.

- REST and gRPC client support.
- Baseline export via `scroll()` with `with_vectors=True`.
- Quarantine uses payload fields.
- Point IDs can be strings or integers.

**Quirks:**
- gRPC is faster but requires `grpcio` dependency.
- Scroll operations return points in arbitrary order.

## Milvus

**Best for:** Large-scale production with GPU acceleration.

- pymilvus SDK.
- Baseline export via `query()` with output fields.
- Quarantine via scalar field (requires schema to include it).
- Batch insert for efficiency.

**Quirks:**
- Schema must be defined before insert; the quarantine field needs to
  be included at collection creation time.
- Connection management requires explicit `connections.connect()`.

## pgvector

**Best for:** Teams already on PostgreSQL who want SQL-native vector search.

- Uses `psycopg` (v3) for async-capable PostgreSQL connections.
- Automatic table creation on `connect()`.
- Quarantine uses a separate `{table}_quarantine` table.
- Full SQL transaction support.

**Quirks:**
- Requires the `pgvector` PostgreSQL extension to be installed.
- Vector dimension must be specified at adapter creation time.
- Large bulk inserts benefit from `COPY` (not yet implemented).

## FAISS

**Best for:** Testing, research, prototyping.

- Pure in-memory, no external service required.
- Parallel ID and metadata bookkeeping (FAISS uses integer IDs).
- Data is lost on process restart unless you save/load the index.
- Not recommended for production use.

**Quirks:**
- FAISS does not support deletion; quarantine only marks in local dict.
- ID mapping adds overhead on very large indexes.
