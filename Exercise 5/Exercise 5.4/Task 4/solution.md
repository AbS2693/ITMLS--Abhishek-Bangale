### Is measuring accuracy sufficient to verify this constraint?

**No, measuring accuracy is completely insufficient** to verify this confidence-based safety constraint.

### Why Accuracy Fails Here

Accuracy only evaluates whether the model’s final binary decision is correct based on a fixed threshold (e.g., p_T \ge 0.5). However, temperature scaling divides the logits by T before the activation function (p_T = \text{activation}(z/T)). Because this is a monotonic transformation, it scales the magnitudes of the probabilities but **does not change whether a probability is above or below 0.5**.

As a result, if we calculate the model's accuracy at a decision threshold of 0.5, it will remain exactly the same across $T=0.5$, $T=1.0$, and T=2.0.

Yet, as seen in Subtask 3, changing T radically alters whether the confidence scores fall below the safety threshold (\theta = 0.6$), meaning the vehicle's safe braking behavior changes completely while accuracy stays stagnant. Accuracy is entirely blind to whether the model's confidence scores are trustworthy.

---

### What additional property must be measured?

The additional property that must be measured is **Model Calibration**, specifically quantified using **Expected Calibration Error (ECE)**.

* 
**What it means:** A model is well-calibrated if its predicted probability (confidence) matches its real-world empirical frequency of being correct. For example, when the pedestrian detector outputs a confidence score of 0.70, it should be correct exactly 70\% of the time.


* 
**Connection to Safety:** To safely use a confidence threshold (\theta = 0.6) to trigger vehicle slowdowns, you must prove that a probability of 0.60 actually means a 60\% chance of a pedestrian being present.


* 
**Your Past Safety Constraints:** This directly tracks your model-level safety constraint **SC-2 (Reliable Confidence)** from Sheet 4, which explicitly mandated that the perception pipeline must maintain an \text{ECE} < 5\% before consideration for deployment.