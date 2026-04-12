"""
Structured logging for detection events.

Provides a thin wrapper around Python's standard logging that emits
JSON-formatted records suitable for log aggregation (ELK, Splunk,
DataDog, etc.).

Usage::

    from embeddings_guardian.utils.logging import GuardianLogger

    glog = GuardianLogger("my-app")
    glog.poisoning_detected("doc_42", "magnitude", 0.99, "z-score=4.2")
    glog.baseline_updated(count=500, mean_norm=1.02)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class GuardianLogger:
    """Structured event logger for embeddings-guardian."""

    def __init__(
        self,
        name: str = "embeddings_guardian",
        level: int = logging.INFO,
        json_output: bool = False,
    ) -> None:
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        self.json_output = json_output

    def _emit(self, event: str, data: Dict[str, Any]) -> None:
        record = {
            "event": event,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **data,
        }
        if self.json_output:
            self._logger.info(json.dumps(record))
        else:
            self._logger.info("[%s] %s", event, " ".join(f"{k}={v}" for k, v in data.items()))

    def poisoning_detected(
        self,
        embedding_id: str,
        algorithm: str,
        confidence: float,
        reason: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a poisoning detection event."""
        data: Dict[str, Any] = {
            "embedding_id": embedding_id,
            "algorithm": algorithm,
            "confidence": confidence,
            "reason": reason,
        }
        if extra:
            data.update(extra)
        self._emit("poisoning_detected", data)

    def baseline_updated(self, count: int, mean_norm: float, std_norm: float = 0.0) -> None:
        """Log a baseline statistics update."""
        self._emit("baseline_updated", {
            "count": count,
            "mean_norm": round(mean_norm, 6),
            "std_norm": round(std_norm, 6),
        })

    def scan_complete(
        self,
        total_scanned: int,
        findings_count: int,
        duration_ms: float,
    ) -> None:
        """Log completion of a batch scan."""
        self._emit("scan_complete", {
            "total_scanned": total_scanned,
            "findings_count": findings_count,
            "duration_ms": round(duration_ms, 2),
        })

    def quarantine_applied(self, embedding_id: str, reason: str) -> None:
        """Log that an embedding was quarantined."""
        self._emit("quarantine_applied", {
            "embedding_id": embedding_id,
            "reason": reason,
        })
