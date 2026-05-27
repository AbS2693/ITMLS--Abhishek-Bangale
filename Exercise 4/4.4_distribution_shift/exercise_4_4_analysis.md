# Exercise 4.4: Distribution Shift Types

## Overview
Analyzing three scenarios involving the CARLA autonomous driving model (trained on sunny daytime data).

---

## Scenario 1: Winter Deployment
**Situation:** Model deployed during winter with wet roads, low sun angle, and glare in camera images.

### a) Type of Distribution Shift
**Answer:** **COVARIATE SHIFT**

**Reasoning:** The input feature distribution P(X) changes due to different lighting conditions (low sun angle, glare) and environmental conditions (wet roads, reflections). However, the relationship between inputs and labels P(Y|X) remains semantically the same — a pedestrian is still a pedestrian, a vehicle is still a vehicle. The model learned what these objects look like in sunny conditions, but the actual class definitions haven't changed, only their visual appearance has.

### b) Expected Effect on Model Performance
**Answer:** **Significant performance degradation expected.**

The model will struggle because it hasn't learned the visual patterns that appear in winter conditions. Wet roads create reflections that can confuse object detection, glare and low sun angles change how shadows appear, and overall contrast/brightness distributions shift dramatically. These are visual phenomena the model has never encountered during training, so its learned features won't activate correctly for winter scenarios.

### c) Mitigation Strategy
**Answer:** **Data augmentation** — Augment training data with synthetic winter conditions (rain, glare simulation, adjusted lighting, low sun angles) to help the model learn these visual variations. Alternatively, collect real winter driving data and retrain or fine-tune the model on this new distribution. 

---

## Scenario 2: New City Zone with High Cyclist Proportion
**Situation:** New city zone added where 60% of road users are cyclists (vs. < 5% in original training data).

### a) Type of Distribution Shift
**Answer:** **LABEL SHIFT**

**Reasoning:** The class distribution P(Y) changes dramatically — cyclists go from <5% to 60% in the test set. The input features (what a cyclist, pedestrian, or vehicle looks like) and their visual characteristics haven't fundamentally changed. A cyclist still looks like a cyclist with the same visual features. However, the *proportion* of each class in the real-world deployment is drastically different from what the model learned, causing a label distribution mismatch between training and test.

### b) Expected Effect on Model Performance
**Answer:** **Likely poor performance on cyclist class, potential false negatives.**

Since cyclists were underrepresented in training (<5%), the model likely didn't learn their features as well as for abundant classes. When suddenly 60% of road users are cyclists, the model may struggle to recognize them correctly, potentially missing cyclists entirely (high false negative rate). The model's learned decision boundaries were optimized for the original class proportions and will be poorly calibrated for this new imbalanced scenario.

### c) Mitigation Strategy
**Answer:** **Rebalance training data** — Collect more cyclist samples and retrain the model with balanced or weighted class distributions. Alternatively, adjust the classification threshold for the cyclist class to increase sensitivity to cyclist detection in the deployment region. 

---

## Scenario 3: New Traffic Light Housing Design
**Situation:** City replaces old traffic light housings with a new, slimmer design that model has never seen.

### a) Type of Distribution Shift
**Answer:** **CONCEPT SHIFT**

**Reasoning:** This is the most fundamental type of shift. The model learned to recognize traffic lights based on the old housing design — it learned specific shape, size, and color patterns of the old housings. When the housings change to a slimmer design, the underlying relationship P(Y|X) changes because what defines a "traffic light" has changed. The semantic concept of "traffic light" in the visual domain is now different. This isn't just a variation in lighting or proportions (covariate), nor is it a class proportion change (label shift) — it's the actual definition of what the model needs to detect that has fundamentally altered.

### b) Expected Effect on Model Performance
**Answer:** **Severe performance degradation or complete failure.**

The model will likely fail to detect the new traffic light designs because they don't match the visual patterns learned during training. The old and new designs may have completely different aspect ratios, spatial arrangements of bulbs, or overall silhouettes. Since the model has no learned features for the new design, it will produce high false negatives (missing traffic lights) or false positives (confusing other objects for traffic lights).

### c) Mitigation Strategy
**Answer:** **Collect new training data with the new traffic light design and retrain the model**, or implement a **domain adaptation** approach to transfer knowledge from old designs to new ones. Alternatively, use an **ensemble approach** that includes models trained on both old and new designs to maintain backward compatibility while learning the new concept. 

---
