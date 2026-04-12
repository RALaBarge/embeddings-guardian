# Security Policy

## Reporting a Vulnerability

**Do not** open a public issue for security vulnerabilities. Instead, email your report to security@ralabarge.dev with:

- **Title**: Brief description of the vulnerability
- **Affected version(s)**: Which embeddings-guardian versions are affected
- **Description**: Technical details and proof-of-concept (if possible)
- **Impact**: What an attacker could do if this vulnerability is exploited
- **Proposed fix** (optional): Any ideas on how to fix it

We will acknowledge receipt within 48 hours and provide an estimated timeline for a fix.

## Security Response Timeline

- **Acknowledgment**: Within 48 hours of report
- **Investigation & patch development**: 5-14 days depending on severity
- **Release**: We will issue a security release as soon as the fix is ready
- **Disclosure**: We will disclose the vulnerability details 30 days after the public fix is released

## Severity Levels

| Level | Description | Response Time |
|-------|-------------|----------------|
| **Critical** | Detection bypass, poisoned data classified as safe | 24-48 hours |
| **High** | Significant false negatives, authentication bypass | 3-5 days |
| **Medium** | Information disclosure, performance degradation | 5-14 days |
| **Low** | Minor issues, cosmetic problems | Next release |

## Security Best Practices

### For embeddings-guardian Users

1. **Keep embeddings-guardian updated**: Security patches are released regularly. Run `pip install --upgrade embeddings-guardian` to get the latest version.

2. **Calibrate baselines carefully**: The detector's effectiveness depends on a clean baseline. Use data from a trusted source:
   ```python
   # Good: Use production embeddings from before poison injection
   baseline = detector.update_baseline_batch(trusted_embeddings)
   
   # Bad: Don't use user-supplied or untrusted embeddings for baseline
   ```

3. **Monitor confidence scores**: Low confidence scores (< 0.7) indicate uncertainty. Increase sensitivity or re-baseline:
   ```python
   if confidence < 0.7:
       logger.warning(f"Low confidence detection: {confidence}")
       # Consider manual review or re-baselining
   ```

4. **Use ensemble mode**: For critical systems, use `algorithm="ensemble"` to combine multiple detectors:
   ```python
   detector = PoisoningDetector(algorithm="ensemble", sensitivity=0.98)
   ```

5. **Log all blocks**: Keep audit logs of blocked embeddings for forensics:
   ```python
   if is_poisoned:
       audit_log.write(f"BLOCKED: {embedding_id} - {reason}")
   ```

### For Contributors

1. All commits should be signed (`git commit -S`)
2. All pull requests must pass security checks:
   - `pip-audit` for dependency vulnerabilities
   - `bandit` for code-level issues
   - `semgrep` for pattern-based security flaws
3. Changes to detection algorithms need careful review and testing
4. New backends should validate input from vector stores
5. Performance optimizations must not weaken detection accuracy

## Known Limitations & Disclaimers

- **No guaranteed detection**: embeddings-guardian detects anomalies but is not 100% accurate. Use as part of a defense-in-depth strategy.
- **Algorithm sensitivity trade-off**: Sensitivity vs false positive rate. You must tune for your use case.
- **Baseline drift**: Legitimate embeddings can drift over time. Re-baseline periodically.
- **Adaptive attacks**: Sophisticated adversaries may craft embeddings that evade detection. Assume this is possible.
- **Backend-specific behavior**: Different vector databases may have different insertion patterns. Test on your backend.

## Security Considerations

### When to Use embeddings-guardian

- ✅ You have control over the embedding model (using internal/trusted providers)
- ✅ You want to detect anomalies in your vector database
- ✅ You can afford latency for per-vector screening (~1-10ms per vector)
- ✅ You can tolerate false positives (safe to be conservative)

### When NOT to Use embeddings-guardian Alone

- ❌ You have a completely untrusted embedding source (you need additional validation)
- ❌ Your adversary has white-box access to the detector (assume they will adapt)
- ❌ You need real-time streaming with microsecond latency (detection adds latency)
- ❌ You're using a proprietary black-box embedding model (you can't reason about drift)

## Comparison with Alternatives

embeddings-guardian is inspired by academic research but is not production-tested at scale. Before deployment:

1. Conduct your own evaluation on representative data
2. Test with your specific embedding model and vector database
3. Implement monitoring and alerting for detection anomalies
4. Have a response plan for high-confidence detections

## Dependency Scanning

We use `pip-audit` to continuously check for vulnerable dependencies. Run it yourself:

```bash
pip-audit --desc
```

## Security Releases

Security releases are issued as `X.Y.Z` (no `-alpha` or `-beta` tags) and announced on:

- GitHub Releases page
- Project README (pinned notice)
- Email to watchers (if subscribed)

## Testing & Validation

To validate embeddings-guardian's behavior in your environment:

```bash
# Run the test suite
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=embeddings_guardian --cov-report=html

# Benchmark on your data
python examples/benchmark_on_custom_data.py --file your_embeddings.npy
```

## Responsible Disclosure

If you discover a way to bypass the detector or craft undetectable poisoned embeddings, please report it confidentially to security@ralabarge.dev. Do not:

- Post proof-of-concept attacks on public channels
- Share bypass techniques before we can patch
- Publish research on weaknesses without giving us time to respond

We commit to working with you on responsible disclosure and will credit you in release notes.
