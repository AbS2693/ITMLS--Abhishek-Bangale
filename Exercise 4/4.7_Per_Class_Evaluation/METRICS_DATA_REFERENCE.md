# Exercise 4.7 - Raw Metrics Data & References

## Complete Metrics from Exercise 3.6 (All Test Splits)

### Pedestrian Model Performance

| Split | Accuracy | Precision | Recall | F1-Score | TP | FP | FN | TN |
|-------|----------|-----------|--------|----------|-----|------|------|------|
| **test** | 0.6528 | 0.2781 | 0.4830 | 0.3530 | 341 | 885 | 365 | 2009 |
| test-fog | 0.7947 | 0.4444 | 0.0327 | 0.0610 | 24 | 30 | 709 | 2837 |
| test-night | 0.5958 | 0.2509 | 0.4973 | 0.3335 | 364 | 1087 | 368 | 1781 |
| test-town-01 | 0.7461 | 0.1665 | 0.4523 | 0.2434 | 147 | 736 | 178 | 2539 |

### Traffic Light Model Performance

| Split | Accuracy | Precision | Recall | F1-Score | TP | FP | FN | TN |
|-------|----------|-----------|--------|----------|------|------|-----|------|
| **test** | 0.9472 | 0.9456 | 0.9830 | 0.9639 | 2540 | 146 | 44 | 870 |
| test-fog | 0.3108 | 0.9866 | 0.0560 | 0.1059 | 147 | 2 | 2479 | 972 |
| test-night | 0.2708 | 1.0000 | 0.0008 | 0.0015 | 2 | 0 | 2625 | 973 |
| test-town-01 | 0.7481 | 0.8658 | 0.7501 | 0.8038 | 1858 | 288 | 619 | 835 |

### Vehicle Model Performance

| Split | Accuracy | Precision | Recall | F1-Score | TP | FP | FN | TN |
|-------|----------|-----------|--------|----------|------|-------|------|------|
| **test** | 0.8678 | 0.9724 | 0.8478 | 0.9058 | 2289 | 65 | 411 | 835 |
| test-fog | 0.5800 | 0.9984 | 0.4578 | 0.6278 | 1275 | 2 | 1510 | 813 |
| test-night | 0.6975 | 0.9813 | 0.6208 | 0.7605 | 1729 | 33 | 1056 | 782 |
| test-town-01 | 0.8458 | 0.9854 | 0.8156 | 0.8935 | 2027 | 31 | 457 | 1085 |

---

## Key Findings Summary

### Recall Ranking on test/test/ (Primary Test Split)
1. **Traffic Light: 0.9830** ⭐ Highest recall (detects 98.3%)
2. **Vehicle: 0.8478** — Good recall (detects 84.8%)
3. **Pedestrian: 0.4830** ⚠️ Lowest recall (detects only 48.3%)

### Safety Criticality Ranking (from Hazard Analysis)
1. **Pedestrian: HIGHEST** — Missed pedestrian = collision/death
2. **Traffic Light: HIGH** — Missed red light = cross-traffic collision
3. **Vehicle: MEDIUM** — Missed vehicle = traffic collision

### Critical Mismatch
- ❌ Pedestrian is HIGHEST safety priority but has LOWEST recall
- ❌ Model performance is INVERSE to safety importance
- ❌ **Major Safety Concern:** System is weakest where it matters most

---

## Safety Constraint Validation (from Exercise 2)

### SC-1: Pedestrian Recall ≥ 95%
- **Current:** 0.4830 (48.30%)
- **Required:** 0.9500 (95.00%)
- **Status:** ❌ **FAILS** (46.7% below threshold)
- **Gap:** -0.467 (needs improvement of 97%)

### SC-2: Calibration (ECE ≤ 0.10)
- Current model calibration not evaluated in this analysis
- Future work: Compute ECE on test set

### SC-3: Latency (p95 ≤ 150ms)
- Latency metrics not evaluated in this analysis
- Future work: Measure inference time distribution

### SC-5: GradCAM Explainability (≥ 0.70)
- Attention maps not evaluated in this analysis
- Future work: Generate GradCAM visualizations

### SC-6: Stop on Presence Logic
- System-level constraint (planner logic)
- Assumes model provides presence signal correctly

---

## Deployment Readiness Assessment

| Model | Recall | SC-1 Status | Deployment Ready? | Priority |
|-------|--------|-------------|-------------------|----------|
| Pedestrian | 0.4830 | ❌ FAILS | **NO** | 🔴 CRITICAL |
| Vehicle | 0.8478 | ⚠️ Unknown* | **MAYBE** | 🟡 MEDIUM |
| Traffic Light | 0.9830 | ✅ PASSES* | **YES** | 🟢 LOW |

*Vehicle and Traffic Light constraints not explicitly defined in SC-1 (pedestrian-specific)

---

## Recommended Actions

### Immediate (Critical Path)
1. **Pedestrian Model Improvement**
   - Data augmentation (synthetic pedestrian variations)
   - Retraining with class weighting
   - Target: 70% recall minimum within 1 week

2. **Retraining Strategy**
   - Increase pedestrian training data by 50%
   - Apply hard negative mining (focus on failure cases)
   - Use focal loss for class imbalance
   - Expected improvement: +15-20%

3. **Validation Protocol**
   - Test on edge cases (occluded, small, far pedestrians)
   - Evaluate on all test variants (fog, night, town-01)
   - Verify no performance regression on other models

### Medium-term (2-4 weeks)
1. Architecture improvements
   - Two-stage detection (RPN + classifier)
   - Anchor optimization for pedestrian scale
   - Expected: +15-20% improvement

2. Ensemble methods
   - Combine multiple pedestrian detectors
   - Voting mechanism for high-confidence detections
   - Expected: +5-10% improvement

### Long-term (1 month+)
1. Continuous monitoring post-deployment
2. Collect real-world failure cases
3. Iterative model refinement

---

## References

**Exercise 2 (Hazard Analysis):**
- H1: Vehicle fails to detect pedestrian in blind spot (low-speed) → Severity: Catastrophic
- H2: Vehicle fails to detect pedestrian crossing intersection (high-speed) → Severity: Catastrophic
- H3: Vehicle fails to detect traffic light → Severity: Severe
- H4: Vehicle fails to detect other vehicle → Severity: Severe

**Safety Constraints:**
- SC-1: Pedestrian Recall ≥ 95%
- SC-2: Calibration ECE ≤ 0.10
- SC-3: Latency p95 ≤ 150ms
- SC-5: GradCAM Overlap ≥ 0.70
- SC-6: Stop on presence logic (planner design)

**Industry Standards Referenced:**
- SOTIF (Safety of the Intended Functionality) - ISO 26262
- NHTSA autonomous vehicle testing protocols
- EU regulations on autonomous vehicle safety
