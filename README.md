# CARLA Autonomous Driving Perception System - ML Safety Case & Assurance Pipeline

**Author:** Abhishek Bangale

**Course:** Introduction to Machine Learning Safety — Summer Semester 2026, Otto-von-Guericke University Magdeburg

**Student:** Abhishek Sudhir Bangale (Matriculation No. 250767)

**System under analysis:** CARLA autonomous vehicle perception stack — single forward-facing RGB camera, three independent ResNet-18 binary classifiers (pedestrian / vehicle / traffic-light presence), rule-based planner, drive-by-wire actuators, human safety operator fallback.

---

## Summary

This repository contains the full engineering trail behind a **STPA + SOTIF/UL 4600-aligned Safety Case** for a three-model CARLA perception system: nine weekly exercises building up model training, calibration, adversarial testing, and out-of-distribution (OOD) monitoring, synthesized into a single final Safety Case report (`MLS_Final_Report_2026.pdf`).

**Final deployment verdict: Deploy with Restrictions.** Of five safety constraints (SC-1–SC-5) verified against measured evidence, only calibration (V-3) is fully met; in-distribution recall (V-1), OOD detection (V-4), and the safe-fallback design (V-5) are partially met; adversarial robustness (V-2) is **not met**. The report carries every unmet constraint forward as an honest, mitigated residual risk rather than hiding it — restrictions include daytime-only operation, a cost-optimal decision threshold, and mandatory auditory alerting.

---

## Repository Structure

```
ITMLS--Abhishek-Bangale/
├── Exercise 1/                    Dataset & system framing
├── Exercise 2/                    Initial STPA (losses, hazards, control structure)
├── Exercise 3/                    Model training & per-class evaluation
│   ├── exercise 3.4/              Dataset exploration, label distribution
│   ├── exercise 3.5/              ResNet-18 fine-tuning, training curves
│   └── exercise 3.6/              Confusion matrices, recall/F1 evaluation
├── Exercise 4/                    ODD definition & coverage analysis
│   ├── 4.4_distribution_shift/    Distribution-shift scoping
│   ├── 4.5_odd_coverage/          k-projection coverage (k=1,2,3)
│   ├── 4.6_test_suite/            Test-suite design
│   └── 4.7_Per_Class_Evaluation/  Per-class metrics reference
├── Exercise 5/                    Uncertainty & data integrity
│   ├── Exercise 5.4/              Temperature scaling, safety-constraint analysis
│   └── Exercise 5.5/              Backdoor/poisoning red-team exercise
├── Exercise 6/                    Explainability & distribution-shift evaluation
│   ├── Exercise 6.5/              Grad-CAM (ID, night-OOD, town-OOD), misclassification analysis
│   └── Exercise 6.6/              Annotated distribution-shift evaluation
├── Exercise 7/                    OOD / anomaly detection
│   ├── Exercise 7.4/              Confidence-distribution visualization
│   ├── Exercise 7.6/              MSP baseline AUROC
│   └── Exercise 7.7/              Feature-based detectors (Mahalanobis, k-NN)
├── Exercise 8/                    Adversarial robustness
│   ├── Exercise 8.4/              FGSM attack generation (ε = 0.01/0.05/0.1)
│   └── Exercise 8.5/              Robustness evaluation report
├── Exercise 9/                    Calibration & cost-optimal fallback
│   ├── Exercise 9.4/              Raw ECE, reliability diagrams
│   ├── Exercise 9.5/              Temperature scaling
│   └── Exercise 9.6/              Asymmetric cost-optimal decision thresholding
├── MLS_Final_Report_2026.pdf      Final Safety Case report
├── scripts                        download_data.py
└── README.md
```

Each exercise folder is self-contained: script(s), generated plots (`.png`), and result files (`.json`/`.txt`) sit together so any step can be re-run or re-inspected independently.

---

## Environment & Reproducibility

**Requirements:** Python 3.10+, PyTorch, Torchvision, scikit-learn, NumPy, Matplotlib, OpenCV (`cv2`), Pillow, pandas.

```bash
# Clone
git clone https://github.com/AbS2693/ITMLS--Abhishek-Bangale.git
cd ITMLS--Abhishek-Bangale

# Virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Dependencies
pip install -r requirements.txt
```

Model weights and the CARLA image dataset are **not committed to this repository** (combined size is several GB) — see the dataset setup section below for how to obtain them.

---

## Dataset Setup & Reproducibility Pipeline

Every script in this repository resolves paths **relative to the repository root**, expecting the raw dataset and model weights to sit as top-level siblings of the `Exercise N/` folders — not nested under a `data/` directory. This mirrors what the actual scripts do (verified against `BASE_PATH`/`MODELS_PATH` in e.g. `Exercise 9/Exercise 9.4/exercise_9_4_measuring_calibration.py` and `Exercise 8/Exercise 8.4/exercise_8_4_fgsm_attack.py`).

### 1. Expected directory layout

```
ITMLS--Abhishek-Bangale/
├── train/
│   └── train/
│       ├── rgb-front/          *.jpg frames
│       └── labels.csv          per-frame pedestrian/vehicle/traffic-light labels
├── validation/
│   └── validation/
│       ├── rgb-front/
│       └── labels.csv
├── test/
│   └── test/                   in-distribution: sunny daytime (7,200 train / 3,600 val / 3,600 test frames)
│       ├── rgb-front/
│       └── labels.csv
├── test-fog/
│   └── test-fog/                OOD split: fog
├── test-night/
│   └── test-night/              OOD split: night
├── test-town-01/
│   └── test-town-01/            OOD split: unmapped town
├── Best Models/
│   ├── model_has_pedestrian.pth
│   ├── model_has_traffic_light.pth
│   └── model_has_vehicle.pth
└── Exercise 1/ … Exercise 9/
```

> Each split folder is doubly-nested (`train/train/`, `test/test/`, etc.) — this matches the dataset's original CARLA export structure and is what every script's `Path(...)` join already expects. Do not flatten it.

### 2. Download the dataset

The raw CARLA image splits and trained model weights are hosted externally (Google Drive) rather than committed to Git, since the combined size is several GB — well beyond what a Git repository should carry.

```bash
pip install gdown
gdown --folder <DRIVE_FOLDER_URL> -O .
```

Drive folder: [Dataset & Model Weights](https://drive.google.com/drive/folders/14l4mbwnsbbGZ0MVNRJN7aeYj6RL5CBuI?usp=drive_link)

Or use the helper script below, which downloads, unzips, and verifies the expected structure in one step:

```bash
python scripts/download_data.py
```

`scripts/download_data.py`:

```python
"""Downloads and verifies the CARLA dataset splits and model weights for this repo.
Fill in DATASET_FILES with your own Google Drive file IDs before running.
"""
import gdown
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


DATASET_FILES = {
    "train.zip":        "1e9tSHg5W4CVDLEBgx1FpzR2j4Kl6jCqh",
    "validation.zip":   "1wKPOEDo0-899PiA9v5-cjjnIIHpsT60R",
    "test.zip":         "1u031oXW9sVkNY9eghr6SvmDEm9JznLya",
    "test-fog.zip":     "1ZQIPhZrBQS0O0wjPW38bXqr4Pc4hvwUj",
    "test-night.zip":   "1igjpOghXUIUyrEcAmv4lTkQnPY6-B0eb",
    "test-town-01.zip": "1CTAOYOjHFc1qq987mYReW5sRVossICCr",
    "Best Models.zip":  "1KCmmG2kfeqzPFXN5gCk4PS6rJ5bWvB0T",
}

REQUIRED_DIRS = [
    "train/train/rgb-front", "validation/validation/rgb-front",
    "test/test/rgb-front", "test-fog/test-fog/rgb-front",
    "test-night/test-night/rgb-front", "test-town-01/test-town-01/rgb-front",
    "Best Models",
]

def download_and_extract(filename: str, file_id: str) -> None:
    zip_path = REPO_ROOT / filename
    gdown.download(id=file_id, output=str(zip_path), quiet=False)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(REPO_ROOT)
    zip_path.unlink()

def verify_structure() -> None:
    missing = [d for d in REQUIRED_DIRS if not (REPO_ROOT / d).exists()]
    if missing:
        raise RuntimeError(f"Missing expected directories after download: {missing}")
    print("All expected dataset/model directories are present.")

if __name__ == "__main__":
    for filename, file_id in DATASET_FILES.items():
        print(f"Downloading {filename} ...")
        download_and_extract(filename, file_id)
    verify_structure()
```

### 3. Dataset & model artifact → script mapping

Every downstream evaluation in the Safety Case traces back to one of these scripts. Paths are relative to the repository root; script names are the actual filenames in this repository.

| Split / Artifact | Used by | Script | Exercise | Produces |
|---|---|---|---|---|
| `train/`, `validation/` | Model fine-tuning (3× ResNet-18) | `Train Classifiers.py` | 3.5 | `Best Models/model_has_{pedestrian,vehicle,traffic_light}.pth` |
| `test/` (in-distribution) | Recall / F1 / confusion-matrix evaluation | `exercise_3_6_evaluation.py` | 3.6 | `exercise_3_6_evaluation_report.json`, confusion matrices |
| `test/` | ODD k-projection coverage | `compute_coverage_official.py` | 4.5 | `coverage_results_official.json` |
| `test/` | Raw calibration (ECE) | `exercise_9_4_measuring_calibration.py` | 9.4 | `exercise_9_4_calibration_results.json`, reliability diagrams |
| `test/` | Temperature scaling | `temperature_Scaling.py` | 9.5 | `exercise_9_5_temperature_scaling_results.json` |
| `test/` | Cost-optimal fallback thresholding | `Cost_Optimal_Decisions.py` | 9.6 | `exercise_9_6_cost_optimal_results.json` |
| `test/` | Grad-CAM explainability | `exercise_6_5_simple.py` | 6.5 | Grad-CAM overlays (`Task 2/{model}/*.png`) |
| `test/` | FGSM adversarial-robustness testing (ε=0.01/0.05/0.1) | `exercise_8_4_fgsm_attack.py` | 8.4 | `exercise_8_4_report.json`, adversarial example images |
| `test/` | Adversarial robustness summary | `exercise_8_5_robustness.py` | 8.5 | `exercise_8_5_robustness_report.json` |
| `test/`, `test-fog/`, `test-night/`, `test-town-01/` | OOD detection — MSP baseline AUROC | `task2_compute_auroc.py` | 7.6 | `exercise_9_6_auroc_results.json`, ROC curves |
| `test/`, `test-fog/`, `test-night/`, `test-town-01/` | OOD detection — Mahalanobis / k-NN AUROC | `task1_extract_features.py` → `task2_evaluate_detectors.py` | 7.7 | `exercise_9_7_results.json`, `auroc_comparison.png` |
| `test-fog/`, `test-night/` | Distribution-shift misclassification analysis | `evaluate_distribution_shift.py` | 6.6 | `distribution_shift_results.json`, annotated overlays |

### 4. End-to-end reproduction

Run in this order from the repository root; each step consumes the previous step's output.

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Data & model weights
python scripts/download_data.py

# 3. Train the three baseline ResNet-18 classifiers (skip if using downloaded weights)
cd "Exercise 3/exercise 3.5"
python "Train Classifiers.py"
cd ../../..

# 4. In-distribution evaluation (V-1 evidence)
cd "Exercise 3/exercise 3.6"
python exercise_3_6_evaluation.py
cd ../../..

# 5. Explainability (V-1 supporting evidence)
cd "Exercise 6/Exercise 6.5"
python exercise_6_5_simple.py
cd ../../..

# 6. Adversarial robustness (V-2 evidence)
cd "Exercise 8/Exercise 8.4"
python exercise_8_4_fgsm_attack.py
cd ../../..

# 7. Calibration (V-3 evidence)
cd "Exercise 9/Exercise 9.4" && python exercise_9_4_measuring_calibration.py && cd ../../..
cd "Exercise 9/Exercise 9.5" && python temperature_Scaling.py && cd ../../..

# 8. OOD detection (V-4 evidence)
cd "Exercise 7/Exercise 7.6" && python task2_compute_auroc.py && cd ../../..
cd "Exercise 7/Exercise 7.7" && python task1_extract_features.py && python task2_evaluate_detectors.py && cd ../../..

# 9. Cost-optimal fallback (V-5 evidence)
cd "Exercise 9/Exercise 9.6"
python Cost_Optimal_Decisions.py
cd ../../..
```

Each script writes its results (`.json`) and plots (`.png`) into its own exercise directory — these are exactly the artifacts cited by number in the Verification Summary table below and in `MLS_Final_Report_2026.pdf`.

---

## Engineering Timeline: Exercise 1 → Final Report



### Phase 1 — System Baseline & Initial Safety Analysis (Exercises 1–2)

- **Objective:** Establish the system boundary and the first-pass STPA before any model existed.
- **Actions:** Defined the CARLA dataset split (7,200 train / 3,600 validation / 3,600 test frames, sunny daytime only) and the system's Operational Design Domain — daytime, dry weather, mapped urban roads, ≤50 km/h. Performed the initial control-structure and hazard analysis: Losses L-1–L-4, Hazards H-1–H-4, and the first Unsafe Control Actions (UCA-1–UCA-8).
- **Safety-case impact:** This phase fixes the vocabulary (L/H/UCA) that every later exercise's evidence gets traced back into.

### Phase 2 — Perception Model Training & Explainability (Exercises 3, 4, 5, 6)

- **Objective:** Train the three detectors, quantify their real-world limits, and verify *why* they fail, not just *that* they fail.
- **Key actions:** Fine-tuned three ResNet-18 binary classifiers (pedestrian / vehicle / traffic-light) on the CARLA training set. Ran k-projection ODD-coverage analysis on the test set.
- **Critical findings:**
  - Severe class imbalance for pedestrians — only 1,718 positive examples (23.9% of the training set) — driving overfitting (training loss ↓80.9%, validation loss ↑93.9%) and a pedestrian recall of **0.653**, well under the 0.90 safety threshold. Traffic-light (0.947) and vehicle (0.867) recall both cleared their respective ≥0.85 thresholds.
  - ODD test-set coverage collapses sharply with dimension count: **k=1: 55.56%**, **k=2: 25.93%**, **k=3: 11.11%** — meaning most higher-order weather/lighting/speed combinations were never sampled.
  - Grad-CAM explainability (Exercise 6.5) confirmed all three detectors attend to the correct object regions rather than background shortcuts — the pedestrian recall gap is a data/capacity problem, not a spurious-correlation problem.
- **Impact on STPA:** Grounds Causal Loss Scenario **LS-1** (low in-distribution recall) and Safety Constraint **SC-1**, verified in **V-1**.

### Phase 3 — Adversarial Vulnerability Assessment (Exercise 8)

- **Objective:** Test whether small, human-imperceptible input perturbations break the detectors.
- **Actions:** Implemented Fast Gradient Sign Method (FGSM) attacks at ε ∈ {0.01, 0.05, 0.1} against all three classifiers.
- **Critical findings:** At ε=0.05, pedestrian recall collapsed by **35–50%** (down to 0.30–0.40 from a clean 0.653), traffic-light recall dropped **10–15%**, and vehicle recall dropped **7–12%**. Under SC-2's strict per-model <10% drop requirement, the pedestrian detector's collapse alone fails the constraint outright.
- **Impact on STPA:** Motivated adding **Hazard H-5** (adversarially perturbed visual inputs) and **UCA-10** (planner acting on an adversarially-induced false negative), feeding Causal Loss Scenario **LS-2** and Safety Constraint **SC-2**, verified — and **not met** — in **V-2**.

### Phase 4 — Out-of-Distribution & Anomaly Detection (Exercise 7)

- **Objective:** Determine whether the system can detect when it has left its training distribution (fog, night, an unmapped town).
- **Actions:** Evaluated Maximum Softmax Probability (MSP) as a baseline OOD detector, then compared it against feature-based detectors (Mahalanobis distance, k-NN) on identical ResNet-18 penultimate-layer (512-D) features.
- **Critical findings:** MSP baseline AUROC was **0.812** on the unseen town, **0.737** on fog, and only **0.537** at night — barely better than chance, since softmax confidence stays high even on inputs the model was never trained on. The Mahalanobis-distance detector on the *same* features reached **0.949** overall (fog 0.997, night 1.000, town-01 0.849) — proving the shortfall is specific to MSP as a statistic, not a limit of the learned representation.
- **Impact on STPA:** Strengthens **Hazard H-2** (undetected OOD operation) via **UCA-9**, grounds Causal Loss Scenario **LS-4** and Safety Constraint **SC-4**, verified as **Partial** in **V-4**.

### Phase 5 — Calibration & Cost-Optimal Fallback (Exercise 9)

- **Objective:** Ensure confidence scores are trustworthy enough to drive a fallback decision, and design that fallback around real, asymmetric safety costs.
- **Actions:** Measured raw Expected Calibration Error (ECE), applied post-hoc temperature scaling, then modeled an asymmetric-cost decision rule for the pedestrian detector.
- **Critical findings:**
  - Raw ECE showed severe overconfidence, especially for pedestrian (**ECE = 0.1150**). Temperature scaling (optimal **T = 2.10** for pedestrian, 1.20 for traffic-light, 1.30 for vehicle) brought all three detectors under the 0.05 threshold (pedestrian 0.0331, traffic-light 0.0275, vehicle 0.0248).
  - With costs $C_{FN}=100$ (missed pedestrian) and $C_{FP}=1$ (false alarm), the cost-optimal threshold $\tau^\ast = C_{FN}/(C_{FN}+C_{FP}) \approx 0.0099$ — far below the standard $\tau=0.5$ — drove false negatives on the pedestrian detector from **365 to 0**, cutting total cost-weighted risk by **92.3%** (37,385 → 2,894), at the cost of false positives rising from 885 to 2,894.
- **Impact on STPA:** Grounds Causal Loss Scenario **LS-3** (overconfident miscalibration, verified **Met** in **V-3**) and the system-level fallback design behind **SC-5**, verified as **Partial** in **V-5** — because the fallback's OOD trigger inherits V-4's night-time detection gap rather than closing it.

### Phase 6 — Final Safety Case Synthesis (Report Stage)

Unified every exercise's evidence into one traceability chain:

```
Evidence → V-n → SC-n → LS-n → UCA-n → H-n → L-n
```

Each of the five verifications (V-1–V-5) cites the exact exercise, script, and metric behind its verdict; every unmet or partial constraint carries an honest residual-risk entry with a concrete mitigation; and the deployment recommendation is derived directly from the verdict table below rather than asserted independently.

---

## Verification Summary

| ID | Checks | Threshold | Empirical Result | Verdict | Hazards |
|----|--------|-----------|-------------------|---------|---------|
| **V-1** | In-distribution recall | Pedestrian ≥0.90, Vehicle/TL ≥0.85 | Pedestrian 0.653, TL 0.947, Vehicle 0.867 | ≈ **Partial** | H-1, H-2, H-3 |
| **V-2** | Adversarial robustness (FGSM ε=0.05) | Recall drop <10% per model | Pedestrian 35–50%, TL 10–15%, Vehicle 7–12% | ✗ **Not met** | H-1, H-2 |
| **V-3** | Calibrated uncertainty (ECE) | ECE <0.05 after scaling | Pedestrian 0.0331, TL 0.0275, Vehicle 0.0248 | ✓ **Met** | H-1, H-2, H-4 |
| **V-4** | OOD detection (AUROC) | AUROC ≥0.90 | MSP: Town-01 0.812, Fog 0.737, Night 0.537 | ≈ **Partial** | H-1, H-2, H-3, H-5 |
| **V-5** | Safe system fallback | Deceleration + operator alert on OOD/low-confidence | FN 365→0 via τ*≈0.0099; inherits V-4's night-time gap | ≈ **Partial** | All hazards |

**Deployment recommendation: Deploy with Restrictions** — daytime operation only, cost-optimal pedestrian threshold ($\tau^\ast\approx0.0099$) mandatory, closed/controlled test routes only, auditory alert added before public trials, operator shift length capped below 4 hours. Restrictions lift once V-1 (pedestrian recall ≥0.90), V-2 (FGSM drop <10%), and V-4 (AUROC ≥0.90, especially at night) are met in a future revision.

---

## Key Engineering Learnings

1. **ML metrics ≠ safety metrics.** Recall, not accuracy, is the safety-relevant statistic for a missed-detection system — a model can look "good" on aggregate accuracy while still missing a third of pedestrians. Likewise, a cost-weighted decision threshold ($\tau^\ast\approx0.0099$) is the safety-relevant operating point, not the ML-conventional $\tau=0.5$.
2. **Single-sensor, single-model systems have no defense in depth.** Every detector shares one RGB camera; an adversarial perturbation or a distribution shift (fog, night) degrades all three detectors simultaneously, because there is no independent modality to cross-check against.
3. **Calibration and OOD detection are prerequisites for safe degradation, not optional extras.** A fallback that decelerates on "low confidence" is only as trustworthy as the calibration behind that confidence (V-3) and only as reliable as the monitor that flags out-of-domain input (V-4) — this project's V-5 verdict is bounded by exactly that dependency chain.
4. **A safety case is a living, traceable argument, not a scorecard.** Every constraint that failed or partially passed here (V-1, V-2, V-4, V-5) is carried forward with a specific, evidence-backed mitigation rather than omitted — an honest "Not Met" with a fix path is stronger evidence of engineering rigor than an unexamined all-green report.

---

## Citation & Course Information

This repository was produced for the course - *Introduction to Machine Learning Safety*, Chair of Software & Systems Engineering, Otto-von-Guericke University Magdeburg, Summer Semester 2026 (Lecturer: Konstantin Kirchheim). The report follows the course-provided Safety Case template and STPA methodology, applied to the CARLA driving simulator dataset supplied for the assignment. Technical content, analysis, and verification results are grounded in this repository's own scripts and outputs. See the Statement of Authorship in `MLS_Final_Report_2026.pdf` for the full academic-integrity declaration.
