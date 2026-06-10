1) What this implies about the model's generalization
- Relies on a spurious correlation, not the causal visual features of pedestrians.
- Likely to fail under distribution shift (different skies, weather, camera angles).
- Overfits dataset-specific background patterns → brittle and non-robust.
- High apparent accuracy on the training/test split but low real-world reliability.
- Indicates the model learned shortcuts instead of true object semantics.

2) Training dataset flaws or issues that could cause this
- Sampling bias: pedestrian images consistently share similar sky/weather conditions.
- Label leakage / background correlation: sky appearance is predictive because labels correlate with capture context.
- Class imbalance or lack of hard negatives: few examples where sky appears without pedestrians.
- Annotation or cropping artifacts: bounding boxes/crops include consistent background regions.
- Low diversity (time/place): images collected from same route/time-of-day producing confounded cues.
- Preprocessing/augmentation error: transforms preserve sky patterns but remove/alter pedestrian cues.
- Noisy labels: mislabeled images that reinforce the wrong association.