"""
Batch detection example.

Demonstrates scanning an existing corpus for poisoned embeddings
without using a backend adapter (works with raw numpy arrays).
"""

import numpy as np

from embeddings_guardian import PoisoningDetector
from embeddings_guardian.utils.metrics import DetectionMetrics
from embeddings_guardian.utils.reporting import Reporter

# 1. Load your corpus embeddings
# In practice: corpus = np.load("embeddings.npy")
rng = np.random.default_rng(42)
corpus = rng.standard_normal((10_000, 768)).astype(np.float32)
corpus_ids = [f"doc_{i}" for i in range(len(corpus))]

# Simulate: inject 20 poisoned embeddings
poison_indices = rng.choice(len(corpus), 20, replace=False)
for idx in poison_indices:
    corpus[idx] *= 50  # inflate magnitude

# 2. Initialize detector and build baseline from clean subset
detector = PoisoningDetector(
    algorithm=["magnitude", "dimension"],  # ensemble
    sensitivity=0.95,
)

# Use first 2000 embeddings as trusted baseline (assume they are clean)
detector.update_baseline_batch(corpus[:2000])

# 3. Scan entire corpus
print("Scanning corpus...")
findings = detector.scan_batch(corpus, ids=corpus_ids)
print(f"Found {len(findings)} suspicious embeddings")

# 4. Evaluate (if you have ground truth)
metrics = DetectionMetrics()
poison_set = set(poison_indices)
for i in range(len(corpus)):
    predicted = any(f["index"] == i for f in findings)
    actual = i in poison_set
    metrics.record(predicted=predicted, actual=actual)

print(f"Precision: {metrics.precision():.3f}")
print(f"Recall:    {metrics.recall():.3f}")
print(f"F1 Score:  {metrics.f1_score():.3f}")
print(f"FPR:       {metrics.false_positive_rate():.3f}")

# 5. Generate report
reporter = Reporter()
reporter.add_findings_from_scan(findings)
reporter.to_csv("poisoning_report.csv")
print(reporter.json_str())
