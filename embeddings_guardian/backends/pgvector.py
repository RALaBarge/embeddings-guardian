"""
pgvector (PostgreSQL) backend adapter.

Requires: ``pip install embeddings-guardian[pgvector]``

Usage::

    from embeddings_guardian import PoisoningDetector
    from embeddings_guardian.backends import PgvectorAdapter

    detector = PoisoningDetector()
    adapter = PgvectorAdapter(
        connection_string="postgresql://user:pass@localhost/mydb",
        table_name="embeddings",
        detector=detector,
    )
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from embeddings_guardian.core.adapters import VectorStoreAdapter
from embeddings_guardian.core.detector import PoisoningDetector

logger = logging.getLogger(__name__)


class PgvectorAdapter(VectorStoreAdapter):
    """Adapter for PostgreSQL with the pgvector extension.

    Uses a separate quarantine table (``{table}_quarantine``) by default.
    """

    name = "pgvector"

    def __init__(
        self,
        connection_string: str,
        table_name: str,
        detector: PoisoningDetector,
        dimension: int = 1536,
    ) -> None:
        self.connection_string = connection_string
        self.table_name = table_name
        self.detector = detector
        self.dimension = dimension
        self._conn: Any = None

    def connect(self) -> None:
        import psycopg

        self._conn = psycopg.connect(self.connection_string)
        # Ensure pgvector extension and tables exist
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id TEXT PRIMARY KEY,
                    embedding vector({self.dimension}),
                    metadata JSONB DEFAULT '{{}}'::jsonb
                )
            """)
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name}_quarantine (
                    id TEXT PRIMARY KEY,
                    reason TEXT,
                    quarantined_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
        self._conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def upsert_embeddings(
        self,
        ids: Sequence[str],
        embeddings: np.ndarray,
        metadata: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        import json

        embeddings = np.asarray(embeddings, dtype=np.float32)
        if metadata is None:
            metadata = [{}] * len(ids)

        with self._conn.cursor() as cur:
            for i, (eid, emb) in enumerate(zip(ids, embeddings)):
                poisoned, confidence, reason = self.detector.is_poisoned(emb)

                if poisoned:
                    logger.warning("Poisoned embedding [%s]: %s", eid, reason)
                    if self.detector.mode == "strict":
                        continue
                    if self.detector.mode == "quarantine":
                        self.quarantine_embedding(eid, reason)
                        continue

                vec_str = "[" + ",".join(str(float(v)) for v in emb) + "]"
                cur.execute(
                    f"""
                    INSERT INTO {self.table_name} (id, embedding, metadata)
                    VALUES (%s, %s::vector, %s::jsonb)
                    ON CONFLICT (id) DO UPDATE
                    SET embedding = EXCLUDED.embedding, metadata = EXCLUDED.metadata
                    """,
                    (eid, vec_str, json.dumps(metadata[i])),
                )
        self._conn.commit()

    def query_embeddings(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
    ) -> List[Tuple[str, float]]:
        vec_str = "[" + ",".join(str(float(v)) for v in query_embedding) + "]"
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, embedding <-> %s::vector AS distance
                FROM {self.table_name}
                ORDER BY distance
                LIMIT %s
                """,
                (vec_str, k),
            )
            return [(row[0], float(row[1])) for row in cur.fetchall()]

    def get_baseline_embeddings(self, sample_size: int = 1000) -> np.ndarray:
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT embedding FROM {self.table_name} LIMIT %s",
                (sample_size,),
            )
            rows = cur.fetchall()
        if rows:
            return np.array([list(r[0]) for r in rows], dtype=np.float32)
        return np.empty((0, 0), dtype=np.float32)

    def quarantine_embedding(self, embedding_id: str, reason: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {self.table_name}_quarantine (id, reason)
                VALUES (%s, %s)
                ON CONFLICT (id) DO UPDATE SET reason = EXCLUDED.reason
                """,
                (embedding_id, reason),
            )
        self._conn.commit()

    def get_statistics(self) -> Dict[str, Any]:
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
            count = cur.fetchone()[0]
            cur.execute(f"SELECT COUNT(*) FROM {self.table_name}_quarantine")
            q_count = cur.fetchone()[0]
        return {
            "backend": self.name,
            "table_name": self.table_name,
            "document_count": count,
            "quarantined_count": q_count,
            "dimension": self.dimension,
        }
