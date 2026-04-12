"""
Report generation for detection results.

Supports JSON and CSV export formats.

Usage::

    from embeddings_guardian.utils.reporting import Reporter

    reporter = Reporter()
    reporter.add_finding("doc_42", 0.97, "magnitude", "z-score=4.2")

    print(reporter.json_report())
    reporter.to_csv("findings.csv")
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class Finding:
    """A single poisoning detection finding."""

    embedding_id: str
    confidence: float
    algorithm: str
    reason: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Optional[Dict[str, Any]] = None


class Reporter:
    """Accumulate findings and generate reports."""

    def __init__(self) -> None:
        self._findings: List[Finding] = []

    def add_finding(
        self,
        embedding_id: str,
        confidence: float,
        algorithm: str,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a detection finding."""
        self._findings.append(
            Finding(
                embedding_id=embedding_id,
                confidence=confidence,
                algorithm=algorithm,
                reason=reason,
                metadata=metadata,
            )
        )

    def add_findings_from_scan(self, scan_results: List[Dict]) -> None:
        """Import results from :meth:`PoisoningDetector.scan_batch`."""
        for r in scan_results:
            self.add_finding(
                embedding_id=r.get("id", str(r.get("index", "unknown"))),
                confidence=r.get("confidence", 0.0),
                algorithm=r.get("algorithm", "unknown"),
                reason=r.get("reason", ""),
            )

    @property
    def total_scanned(self) -> int:
        return len(self._findings)

    def json_report(self) -> Dict[str, Any]:
        """Generate a JSON-serializable summary report."""
        confidences = [f.confidence for f in self._findings]
        return {
            "summary": {
                "total_findings": len(self._findings),
                "confidence_avg": sum(confidences) / len(confidences) if confidences else 0.0,
                "confidence_max": max(confidences) if confidences else 0.0,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
            "findings": [
                {
                    "id": f.embedding_id,
                    "confidence": f.confidence,
                    "algorithm": f.algorithm,
                    "reason": f.reason,
                    "timestamp": f.timestamp,
                }
                for f in self._findings
            ],
        }

    def json_str(self, indent: int = 2) -> str:
        """Return report as a formatted JSON string."""
        return json.dumps(self.json_report(), indent=indent)

    def csv_report(self) -> str:
        """Generate CSV-formatted report as a string."""
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["id", "confidence", "algorithm", "reason", "timestamp"])
        for f in self._findings:
            writer.writerow([f.embedding_id, f.confidence, f.algorithm, f.reason, f.timestamp])
        return buf.getvalue()

    def to_csv(self, path: str) -> None:
        """Write findings to a CSV file."""
        with open(path, "w", newline="") as fh:
            fh.write(self.csv_report())

    def clear(self) -> None:
        """Remove all accumulated findings."""
        self._findings.clear()
