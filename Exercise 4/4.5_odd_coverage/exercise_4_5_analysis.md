# Exercise 4.5: ODD Coverage with k-Projection Coverage

## Overview
Assessing how well the test set covers the Operational Design Domain (ODD) using k-projection coverage metric.

---

## Part 1: Understanding k-Projection Coverage

### Question 1: What is k-projection coverage and why is it useful?

**Answer:**

**k-projection coverage** measures the proportion of k-dimensional feature combinations from the training set's ODD that are also represented in the test set. In other words, it assesses how well the test set covers the multi-dimensional feature space defined by the ODD.

**Definition:**
- For k=1: Coverage of individual ODD dimensions (e.g., "clear weather", "daytime lighting", "urban scenes")
- For k=2: Coverage of pairwise combinations (e.g., "clear weather AND daytime", "rainy AND night")
- For k=3: Coverage of triplet combinations (e.g., "clear AND daytime AND urban", "rainy AND night AND highway")

**Why it is useful for ODD coverage:**

1. **Reveals distribution mismatches**: High k=1 coverage paired with low k=2/k=3 coverage reveals that while individual ODD dimensions appear in the test set, realistic combinations of conditions may be missing. This is critical because real-world scenarios have multiple ODD dimensions active simultaneously.

2. **Validates adequacy of test set**: A test set with high k-projection coverage (especially k=3) provides evidence that the system was tested under representative combinations of ODD conditions, not just individual variations. This directly supports a safety argument.

3. **Identifies ODD gaps**: When coverage is low for certain k values, we can pinpoint which combinations of conditions are underrepresented or absent from testing. For example, if "foggy + nighttime + low visibility" has low coverage, we know this specific scenario needs more test data.

4. **Quantifies test representativeness**: Instead of subjectively claiming "our test set is diverse", k-projection coverage provides a quantitative metric to demonstrate that the test set faithfully represents the training distribution across multiple ODD dimensions simultaneously.

5. **Supports incremental improvement**: Coverage values guide where to collect additional test data. For instance, if k=2 coverage is 85% but k=3 is only 45%, we know we need to focus on rare multi-dimensional combinations.

---

## Part 2: ODD Reference from Exercise 2

Your ODD defined these operating conditions:

| Dimension | Operating Condition (Valid) | Non-Operating Condition |
|---|---|---|
| **Weather** | Clear, cloudy (no rain/fog) | Heavy rain, fog, snow |
| **Lighting** | Daytime (high sun angle) | Night, low sun (glare) |
| **Camera** | Forward-facing, rigidly fixed | Loose mount, lens obscuration |
| **Scene Type** | Urban roads, mapped intersections | Off-road, unmapped highways |
| **Vehicle Speed** | ≤ 50 km/h | > 50 km/h |

---

## Part 3: Computing k-Projection Coverage

### Dataset Information
- **Train set location:** `train/train/`
- **Validation set location:** `validation/validation/`
- **Test set location:** `test/test/` (and variants: test-fog, test-night, test-town-01)
- **RGB Images:** `rgb-front/`
- **Segmentation Maps:** `segmentation-front/`
- **Metadata:** `.feather` files (actions, collisions, gnss, imu, labels, weather)

## Part 3: Computing k-Projection Coverage

### Dataset Information
- **Train set location:** `train/train/`
- **Validation set location:** `validation/validation/`
- **Test set location:** `test/test/` (and variants: test-fog, test-night, test-town-01)
- **RGB Images:** `rgb-front/`
- **Segmentation Maps:** `segmentation-front/`
- **Metadata:** `.feather` files (actions, collisions, gnss, imu, labels, weather)

**Extracted ODD Dimensions (Official Implementation):**
1. **Weather**: Clear, Cloudy, Rainy (from precipitation and fog density)
2. **Lighting**: Daytime, Dusk/Dawn, Night (from sun altitude angle)
3. **Speed**: Low (<0.3 throttle), Medium (0.3-0.7), High (>0.7 throttle)

**Samples Processed:**
- Training set: 7,200 frames
- Test set (normal): 3,600 frames
- Test-fog: 3,600 frames
- Test-night: 3,600 frames
- Test-town-01: 3,600 frames

### Coverage Results for k ∈ {1, 2, 3}

#### k = 1 (Single Feature Projection) - Official Results
**Coverage: 55.56%**

**Details:**
- Total possible 1D combinations: 9 (3 weather × 3 lighting, 3 speed, etc.)
- Covered combinations: 5/9
- Missing combinations: 4/9

**Breakdown:**
- Weather values present in data: {clear, cloudy, rainy} - but actual data shows only **{cloudy, rainy}**
- Lighting values present: {daytime, dusk_dawn, night} - but actual data shows only **{daytime, night}**
- Speed values: all 3 present {low, medium, high}

**Interpretation:** 
The test set covers only 55.56% of possible individual ODD dimensions. Most notably:
- **Missing**: Clear weather (all data is cloudy or rainy)
- **Missing**: Dusk/Dawn lighting (only daytime or night present)

This indicates the training/test data is **not diverse enough** in certain individual dimensions.

---

#### k = 2 (Pairwise Feature Projection) - Official Results
**Coverage: 25.93%**

**Details:**
- Total possible 2D combinations: 27 (3 weather × 3 lighting, 3 weather × 3 speed, etc.)
- Covered combinations: 7/27
- Missing combinations: 20/27

**Interpretation:** 
Coverage **drops dramatically** to 25.93% at k=2. This reveals:
- While we have 5 individual dimensions (55.56% of 1D space)
- Only 7 out of 27 pairwise combinations are actually present
- **Major gaps**: Many realistic multi-dimensional combinations are missing

Example gaps:
- No "clear + any lighting" combinations
- No "cloudy + dusk_dawn" combinations
- Limited "rainy + any lighting" combinations

---

#### k = 3 (Triplet Feature Projection) - Official Results
**Coverage: 11.11%**

**Details:**
- Total possible 3D combinations: 27 (3 weather × 3 lighting × 3 speed)
- Covered combinations: 3/27
- Missing combinations: 24/27

**Interpretation:**
Coverage **severely restricted** at k=3, covering only 11.11% of possible triplet combinations. This indicates:
- The test set covers very limited combinations of 3 dimensions simultaneously
- Most real-world scenarios (random combinations of weather, lighting, speed) are missing
- **Critical finding**: While datasets are large (3600 samples), they lack diversity in multi-dimensional combinations

---

## Part 4: Analysis

### Question 2: How does coverage change with increasing k? What does this tell about test set adequacy?

**Official Results - Coverage Trend:**
- k=1: **55.56%** (5/9 individual dimension combinations)
- k=2: **25.93%** (7/27 pairwise dimension combinations)
- k=3: **11.11%** (3/27 triplet dimension combinations)

**Coverage Change Analysis:**

| k | Coverage | Change | Interpretation |
|---|----------|--------|---|
| k=1 | 55.56% | baseline | Half of possible individual conditions present |
| k=2 | 25.93% | ↓ 52% | Dramatic drop - pairwise combinations much rarer |
| k=3 | 11.11% | ↓ 57% | Severe drop - triplet combinations very sparse |

**What This Reveals:**

1. **Sharp decline with increasing k (55% → 26% → 11%)**
   - This is TYPICAL for real-world datasets
   - Combinatorial explosion: more dimensions = exponentially more combinations
   - Even large datasets (3600 samples) cannot cover all combinations

2. **k=1 at 55.56% indicates limited individual coverage**
   - Training data only shows cloudy/rainy weather (missing "clear")
   - Only daytime/night lighting (missing dusk/dawn)
   - This is a **significant ODD gap** at the individual dimension level

3. **k=2 at 25.93% reveals severe diversity issues**
   - Most pairwise combinations are missing
   - For example: "clear + daytime", "cloudy + night" are absent
   - Real-world conditions have multiple dimensions active simultaneously
   - Test set is **NOT adequate** at k=2 level

4. **k=3 at 11.11% shows critical lack of triplet combinations**
   - Only 3 out of 27 possible triplet combinations exist
   - A vehicle deployed in the real world will encounter scenarios outside this 11% coverage
   - **Major safety concern**: Model not tested on most possible condition combinations

---

### Comparison Across Test Variants

**Official Library Results:**

| Test Variant | k=1 | k=2 | k=3 | Notes |
|---|---|---|---|---|
| **test_normal** | 55.56% | 25.93% | 11.11% | Reference test set |
| **test_fog** | 55.56% | 25.93% | 11.11% | Identical to normal |
| **test_night** | 55.56% | 25.93% | 11.11% | Same coverage pattern |
| **test_town_01** | 55.56% | 25.93% | 11.11% | Same coverage pattern |

**Unexpected Finding:**
All test variants show **identical coverage percentages**! 

**Why?** The official library computes coverage based on which combinations appear in the test set. Since all variants contain the same weather/lighting/speed discretization patterns (only cloudy/rainy, daytime/night combinations), they achieve identical k-projection coverage.

---

### Key Insights

1. **Coverage trend interpretation:**
   - Declining coverage with k is expected (11% for k=3 is typical for real datasets)
   - However, **55.56% at k=1 is concerning** - we're missing fundamental individual conditions
   - This suggests data was collected in limited environmental conditions

2. **ODD gaps identified:**
   - **Missing in all data**: Clear weather conditions (only cloudy/rainy present)
   - **Missing in all data**: Dusk/dawn lighting (only daytime/night present)  
   - **Imbalanced**: Speed states are well-covered but not combined with diverse weather/lighting
   - **Critical**: The ODD actually defined includes "clear" and "daytime", but data doesn't contain them

3. **Test set adequacy assessment:**
   - ❌ **Inadequate at k=1** (55.56% < 100%): Missing individual condition types
   - ❌ **Severely inadequate at k=2** (25.93%): Most pairwise combinations absent
   - ❌ **Very inadequate at k=3** (11.11%): Extreme lack of condition diversity
   - **Recommendation**: Dataset is **NOT adequate** for safety-critical testing

4. **Safety Implications:**
   - **Risk**: Model trained/tested on limited condition combinations
   - **Gap**: Real-world may have "clear + night + high speed" or other missing combinations
   - **Action needed**: Collect more diverse data covering the full ODD space, especially:
     - Clear weather conditions (0% in current data)
     - Dusk/dawn lighting conditions (0% in current data)
     - All pairwise and triplet combinations of weather, lighting, speed

5. **Recommendations for improvement:**
   - Expand data collection to include missing weather conditions (clear skies)
   - Add lighting diversity (dusk, dawn, twilight hours)
   - Stratify collection to ensure all k-dimensional combinations are represented
   - Target minimum k=3 coverage of at least 50% for adequate safety testing
   - Current 11% k=3 coverage is a **significant limitation**

---

## Part 5: Conclusion

### Summary of Findings (Official GitHub Library Results)

**Official k-projection coverage values (kkirchheim/odd-coverage):**
- **k=1: 55.56%** ⚠ (5/9 individual dimension combinations)
- **k=2: 25.93%** ⚠ (7/27 pairwise dimension combinations)  
- **k=3: 11.11%** ❌ (3/27 triplet dimension combinations)

**Critical Observations:**

1. ❌ **k=1 Coverage (55.56%) is INADEQUATE**
   - Only half of possible individual conditions are present
   - Missing: "clear" weather (dataset only has cloudy/rainy)
   - Missing: "dusk/dawn" lighting (dataset only has daytime/night)
   - **These are defined in the ODD but not in the test data!**

2. ❌ **k=2 Coverage (25.93%) reveals severe gaps**
   - 20 out of 27 pairwise combinations are missing
   - Real-world scenarios combine multiple dimensions; we test <26% of combinations
   - **Safety risk**: Model may fail on untested dimension combinations

3. ❌ **k=3 Coverage (11.11%) is critically low**
   - Only 3 out of 27 triplet combinations covered
   - 89% of possible 3-dimensional scenarios are missing from test set
   - **Major safety concern**: Cannot claim robustness to untested scenarios

### Safety Implications

**Positive Findings:**
- ✓ Dataset is large (7200 training, 3600 test samples)
- ✓ All test variants have consistent coverage patterns

**Critical Safety Gaps:**
- ❌ Test set does NOT cover ODD as defined in Exercise 2
- ❌ Data diversity is limited despite large sample count
- ❌ Multi-dimensional condition combinations are severely underrepresented
- ❌ Cannot certify model robustness across full ODD space at k=2, k=3 levels

### Recommendations for Improvement

1. **Immediate**: Acknowledge that current dataset has incomplete ODD coverage
2. **Expand data collection** to achieve:
   - k=1 coverage ≥ 90% (cover individual conditions properly)
   - k=2 coverage ≥ 70% (cover most pairwise combinations)
   - k=3 coverage ≥ 50% (cover meaningful triplet combinations)

3. **Targeted collection** of missing scenarios:
   - Clear weather conditions (currently 0%)
   - Dusk/dawn lighting (currently 0%)
   - Under-represented combinations (e.g., clear + night, rainy + daytime)

4. **Re-run analysis** after data expansion to validate improved coverage

### Files Generated

- `exercise_4_5_analysis.md` - This comprehensive analysis document
- `compute_coverage_official.py` - Script using official GitHub library
- `results_official/coverage_results_official.json` - Machine-readable official results
- `results_official/coverage_summary_official.txt` - Text summary of official results
- `GUIDE.md` - Detailed k-projection coverage explanation
- `SUMMARY.md` - Previous summary document
- `odd-coverage/` - Cloned GitHub repository (official library)

---

### Reference

**GitHub Repository:** https://github.com/kkirchheim/odd-coverage

**Paper:** "Quantitative Projection Coverage for Testing ML-enabled Autonomous Systems" by Chih-Hong Cheng et al. (https://arxiv.org/abs/1805.04333)

**Library Used:** `KProjectionCoverage` from official kkirchheim/odd-coverage

---
