# Freezing/Unfreezing in Neural Networks - Explained

## What Does Freezing/Unfreezing Mean?

When training a neural network, **freezing** means preventing certain layers from updating their weights during training. **Unfreezing** means allowing them to update again.

```
Frozen Layer:    ❌ requires_grad = False  → Weights DON'T change during backprop
Unfrozen Layer:  ✅ requires_grad = True   → Weights CHANGE during backprop
```

---

## The Three Training Strategies Your Tutor Mentioned

### Strategy 1: **Freeze Full Network + Train Only Head**
```python
# Freeze ALL backbone layers
for param in model.backbone.parameters():
    param.requires_grad = False

# Unfreeze ONLY the new classification head
for param in model.fc.parameters():  # ← Just the final layer
    param.requires_grad = True

# Now train: Only the head weights update
optimizer = Adam(model.parameters(), lr=1e-3)  # Can use higher LR
```

**When to use:**
- You have **very limited labeled data**
- Target task is **similar to ImageNet**
- You just need to adapt the final layer
- **Fastest training**, minimal overfitting risk

**Speed:** ⚡⚡⚡ FAST (fewer parameters to update)

---

### Strategy 2: **Unfreeze Full Network + Train Everything**
```python
# Unfreeze EVERYTHING (this is the default)
for param in model.parameters():
    param.requires_grad = True

# Train the entire model
optimizer = Adam(model.parameters(), lr=1e-4)  # Lower LR to protect pre-trained weights
```

**When to use:**
- You have **plenty of labeled data**
- Target task is **significantly different** from ImageNet
- You want **maximum model adaptation**
- Willing to risk overfitting

**Speed:** 🐢 SLOW (all parameters update)

---

### Strategy 3: **Freeze Backbone + Train Head** (Most Common - What You Did!)
```python
# Freeze backbone layers (pre-trained features)
for param in model.backbone.parameters():
    param.requires_grad = False

# Unfreeze the NEW classification head
for param in model.fc.parameters():
    param.requires_grad = True

# THEN: Train, monitor, then optionally unfreeze backbone
optimizer = Adam(model.parameters(), lr=1e-4)
```

**This is called "fine-tuning" and has TWO phases:**

**Phase 1 - Train Head Only:** (what most people do)
- Frozen backbone + trainable head
- Fast, stable learning
- Preserves ImageNet features

**Phase 2 - Full Fine-tuning:** (optional advanced)
- Unfreeze backbone, reduce LR even more (1e-5)
- Train both backbone and head
- Slower but better adaptation

**When to use:** ✅ YOUR SITUATION
- Medium amount of data
- Related to ImageNet but different enough
- Want good balance of speed and accuracy
- Industry standard for transfer learning

**Speed:** ⏱️ MEDIUM

---

## What YOU Actually Did in Your Code

Looking at your `exercise_3_5_solution.py` and `exercise_3_5_train_classifiers.py`:

### ✅ What You Did Right:

1. **Used a Pre-Trained ResNet-18**
   ```python
   model = models.resnet18(pretrained=True)  # ImageNet pre-trained
   ```
   ✓ This meant backbone was already learned

2. **Replaced Only the Final Layer**
   ```python
   model.fc = nn.Linear(model.fc.in_features, 2)  # Old: 1000 → New: 2 classes
   ```
   ✓ This is the "head" - only this part is random

3. **Used a Very Low Learning Rate**
   ```python
   LR = 1e-4  # Super low!
   optimizer = optim.Adam(model.parameters(), lr=LR)
   ```
   ✓ This protects the pre-trained weights from being destroyed

4. **Implicitly Did Strategy 3 (Fine-tuning)**
   - The NEW head layer (randomly initialized) needs training
   - The backbone weights are already good from ImageNet
   - By not explicitly freezing, you let both train BUT with low LR
   - The low LR means: backbone learns very slowly (stays mostly the same), head learns quickly

### ❌ What You DIDN'T Explicitly Do:

```python
# You didn't explicitly write:
for param in model.features.parameters():
    param.requires_grad = False
```

**But that's FINE!** Because of the low learning rate (1e-4), the backbone weights barely changed anyway. The effect was similar to Strategy 1 or 3.

---

## Comparison Table: What Happens in Each Strategy

| Strategy | Backbone | Head | LR | Speed | When | Your Code |
|----------|----------|------|-------|-------|------|-----------|
| **1** | ❌ Frozen | ✅ Train | 1e-3 | ⚡⚡⚡ | Tiny data | No |
| **2** | ✅ Train | ✅ Train | 1e-4 | 🐢 | Lots of data | No |
| **3a** | ❌ Frozen | ✅ Train | 1e-4 | ⏱️ | Phase 1 | Effectively YES |
| **3b** | ✅ Train | ✅ Train | 1e-5 | 🐢 | Phase 2 | No |

---

## Your Models: Pedestrian, Traffic Light, Vehicle

### What Actually Happened:

```
ImageNet ResNet-18
       ↓
    [Backbone] ← Frozen (implicitly by low LR)
       ↓
    [2048 features]
       ↓
    [NEW Head] ← Trained on CARLA data
       ↓
    [Binary output]
```

### Why This Worked Well for You:

1. **Pedestrian Detection (23.9% positive - IMBALANCED)**
   - Backbone knew general shapes ✓
   - Head learned "what makes a pedestrian" ✓
   - Class weighting helped handle imbalance ✓

2. **Traffic Light Detection (73.2% positive - BALANCED)**
   - Backbone knew color patterns ✓
   - Head learned "traffic light shape + color" ✓
   - Pre-training helped with this simple task ✓

3. **Vehicle Detection (75.8% positive - BALANCED)**
   - Backbone knew vehicle shapes ✓
   - Head specialized for CARLA vehicles ✓
   - Easiest of the three tasks ✓

---

## If You Were To Explicitly Implement Strategy 3

Here's what your code would look like:

```python
# Create model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)

# === PHASE 1: Train head only ===
# Freeze all backbone parameters
for param in model.features.parameters():
    param.requires_grad = False

# Only head is trainable
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)

# Train for 5 epochs...
for epoch in range(5):
    # Training loop

# === PHASE 2: Fine-tune full network ===
# Unfreeze backbone
for param in model.features.parameters():
    param.requires_grad = True

# Train both with much lower LR
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Train for 5-10 more epochs...
for epoch in range(5):
    # Training loop
```

---

## Key Takeaways

| Concept | Explanation |
|---------|-------------|
| **Freezing** | `requires_grad = False` → weights don't update |
| **Unfreezing** | `requires_grad = True` → weights do update |
| **Strategy 1** | Freeze backbone, train head → Fast, limited adaptation |
| **Strategy 2** | Train everything → Slow, high overfitting risk |
| **Strategy 3** | Train head, then backbone → Balanced (YOUR METHOD) |
| **Why Low LR** | Protects pre-trained weights from being destroyed |
| **Your Code** | Effectively did Strategy 3 Phase 1 implicitly |
| **Your Results** | Worked well! Pre-training + new head learned CARLA tasks |

---

## Summary for Your Presentation

**"In Exercise 3.5, we implicitly used Strategy 3 (fine-tuning):**
- **Backbone**: Kept pre-trained ImageNet knowledge (protected by low learning rate 1e-4)
- **Head**: Trained to recognize CARLA-specific objects (pedestrians, traffic lights, vehicles)
- **Result**: Effective transfer learning without explicit freezing code
- **Why it worked**: Pre-trained features + specialized adaptation = good balance"

