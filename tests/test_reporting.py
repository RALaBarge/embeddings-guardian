"""Tests for reporting module."""

import json
import os
import tempfile

import pytest

from embeddings_guardian.utils.reporting import Reporter


@pytest.fixture
def reporter():
    r = Reporter()
    r.add_finding("doc_1", 0.95, "magnitude", "z-score=4.1")
    r.add_finding("doc_2", 0.88, "centroid", "far from cluster")
    r.add_finding("doc_3", 0.72, "magnitude", "z-score=3.5")
    return r


class TestReporter:
    def test_add_finding(self, reporter):
        assert reporter.total_scanned == 3

    def test_json_report_structure(self, reporter):
        report = reporter.json_report()
        assert "summary" in report
        assert "findings" in report
        assert report["summary"]["total_findings"] == 3
        assert report["summary"]["confidence_avg"] > 0

    def test_json_str(self, reporter):
        s = reporter.json_str()
        parsed = json.loads(s)
        assert len(parsed["findings"]) == 3

    def test_csv_report(self, reporter):
        csv = reporter.csv_report()
        lines = csv.strip().split("\n")
        assert len(lines) == 4  # header + 3 findings
        assert "doc_1" in lines[1]

    def test_to_csv_file(self, reporter):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = f.name
        try:
            reporter.to_csv(path)
            with open(path) as fh:
                content = fh.read()
            assert "doc_2" in content
        finally:
            os.unlink(path)

    def test_clear(self, reporter):
        reporter.clear()
        assert reporter.total_scanned == 0

    def test_add_findings_from_scan(self):
        r = Reporter()
        scan_results = [
            {"id": "x", "confidence": 0.9, "reason": "test"},
            {"index": 5, "confidence": 0.8, "reason": "test2"},
        ]
        r.add_findings_from_scan(scan_results)
        assert r.total_scanned == 2

    def test_empty_report(self):
        r = Reporter()
        report = r.json_report()
        assert report["summary"]["total_findings"] == 0
        assert report["summary"]["confidence_avg"] == 0.0
