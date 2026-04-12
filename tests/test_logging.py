"""Tests for structured logging."""

import logging

from embeddings_guardian.utils.logging import GuardianLogger


class TestGuardianLogger:
    def test_poisoning_detected(self, caplog):
        glog = GuardianLogger(level=logging.INFO)
        with caplog.at_level(logging.INFO, logger="embeddings_guardian"):
            glog.poisoning_detected("doc_1", "magnitude", 0.99, "z=4.2")
        assert "poisoning_detected" in caplog.text

    def test_baseline_updated(self, caplog):
        glog = GuardianLogger(level=logging.INFO)
        with caplog.at_level(logging.INFO, logger="embeddings_guardian"):
            glog.baseline_updated(count=100, mean_norm=1.02, std_norm=0.05)
        assert "baseline_updated" in caplog.text

    def test_scan_complete(self, caplog):
        glog = GuardianLogger(level=logging.INFO)
        with caplog.at_level(logging.INFO, logger="embeddings_guardian"):
            glog.scan_complete(total_scanned=500, findings_count=3, duration_ms=42.5)
        assert "scan_complete" in caplog.text

    def test_json_output(self, caplog):
        glog = GuardianLogger(level=logging.INFO, json_output=True)
        with caplog.at_level(logging.INFO, logger="embeddings_guardian"):
            glog.quarantine_applied("doc_99", "test reason")
        assert '"event"' in caplog.text

    def test_custom_name(self, caplog):
        glog = GuardianLogger(name="myapp.guardian", level=logging.INFO)
        with caplog.at_level(logging.INFO, logger="myapp.guardian"):
            glog.poisoning_detected("x", "mag", 0.5, "test")
        assert "poisoning_detected" in caplog.text
