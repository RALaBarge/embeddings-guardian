# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-04-12

### Added

- Core `PoisoningDetector` with pluggable algorithm support
- Five detection algorithms: magnitude, centroid, neighborhood, dimension, fingerprint
- `EnsembleScorer` for combining multiple detection methods
- Abstract `VectorStoreAdapter` base class
- Backend adapters: ChromaDB, Pinecone, Weaviate, Qdrant, Milvus, pgvector, FAISS
- `DetectionMetrics` for precision/recall/F1 tracking
- `Reporter` for JSON and CSV report generation
- Structured logging via `GuardianLogger`
- Comprehensive test suite (100+ tests)
- Documentation: getting started, backends, algorithms, examples
- Apache 2.0 license
- PyPI packaging with optional backend dependencies
