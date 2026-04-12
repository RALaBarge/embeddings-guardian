"""
Monitoring and structured logging example.

Demonstrates using GuardianLogger for structured event output
suitable for log aggregation systems (ELK, Splunk, DataDog).
"""

import time
import numpy as np

from embeddings_guardian import PoisoningDetector
from embeddings_guardian.utils.logging import GuardianLogger
from embeddings_guardian.utils.metrics import DetectionMetrics

# Set up structured JSON logging
glog = GuardianLogger(name="myapp.rag_security", json_output=True)

# Initialize detector
detector = PoisoningDetector(algorithm="magnitude", sensitivity=0.95)

# Build baseline
rng = np.random.default_rng(42)
baseline = rng.standard_normal((200, 128)).astype(np.float32)
detector.update_baseline_batch(baseline)

stats = detector.get_stats()
glog.baseline_updated(
    count=stats["scorer"]["count"],
    mean_norm=stats["scorer"]["mean_norm"],
    std_norm=stats["scorer"]["std_norm"],
)

# Simulate incoming embeddings
metrics = DetectionMetrics()
incoming = rng.standard_normal((100, 128)).astype(np.float32)
# Poison a few
incoming[42] *= 100
incoming[77] *= 200

start = time.perf_counter()
for i, emb in enumerate(incoming):
    is_p, conf, reason = detector.is_poisoned(emb)
    metrics.record(predicted=is_p, actual=(i in (42, 77)), confidence=conf)

    if is_p:
        glog.poisoning_detected(
            embedding_id=f"emb_{i}",
            algorithm="magnitude",
            confidence=conf,
            reason=reason,
            extra={"batch_index": i},
        )

elapsed_ms = (time.perf_counter() - start) * 1000

glog.scan_complete(
    total_scanned=len(incoming),
    findings_count=metrics.tp + metrics.fp,
    duration_ms=elapsed_ms,
)

# Print metrics summary
m = metrics.to_dict()
print(f"Precision: {m['precision']:.3f}")
print(f"Recall:    {m['recall']:.3f}")
print(f"F1:        {m['f1_score']:.3f}")
print(f"Duration:  {elapsed_ms:.1f}ms ({elapsed_ms/len(incoming):.2f}ms/vector)")
