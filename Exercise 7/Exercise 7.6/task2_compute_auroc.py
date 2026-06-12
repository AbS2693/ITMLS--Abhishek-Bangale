"""Compute AUROC for separating in-distribution from OOD over all scenarios.
Using maximum softmax probability as OOD score.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import json
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

BASE_PATH = Path("../..")
MODEL_PATH = BASE_PATH / "Best Models" / "model_has_traffic_light.pth"
ID_TEST_PATH = BASE_PATH / "test" / "test"  # Sunny/daytime
OOD_PATHS = {
    "fog": BASE_PATH / "test-fog" / "test-fog",
    "night": BASE_PATH / "test-night" / "test-night",
    "town-01": BASE_PATH / "test-town-01" / "test-town-01",
}

class CarlaImageDataset(Dataset):
    """CARLA dataset for loading RGB images."""
    
    def __init__(self, dataset_path, transform=None):
        self.image_dir = dataset_path / "rgb-front"
        self.labels_df = pd.read_csv(dataset_path / "labels.csv")
        self.transform = transform
        
        
        self.images = sorted(list(self.image_dir.glob("*.jpg")))
        
        print(f"Loaded {len(self.images)} images from {dataset_path.name}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image



def load_model(model_path):
    """Load the trained traffic light model."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)  # Binary classification
    
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(DEVICE)
    
    print(f"✓ Loaded model from {model_path.name}")
    return model


def compute_msp_scores(model, dataloader):
    """
    Compute Maximum Softmax Probability (MSP) scores.
    
    MSP = max(softmax(logits))
    High MSP → High confidence → In-distribution
    Low MSP → Low confidence → Out-of-distribution
    """
    scores = []
    
    with torch.no_grad():
        for images in dataloader:
            images = images.to(DEVICE)
            logits = model(images)
            probs = torch.nn.functional.softmax(logits, dim=1)
            max_probs = probs.max(dim=1).values
            scores.extend(max_probs.cpu().numpy())
    
    return np.array(scores)


print("\n" + "="*70)
print("LOADING DATA AND COMPUTING MSP SCORES")
print("="*70)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

model = load_model(MODEL_PATH)

print("\nComputing MSP scores for ID (sunny/daytime test)...")
id_dataset = CarlaImageDataset(ID_TEST_PATH, transform)
id_loader = DataLoader(id_dataset, batch_size=32, shuffle=False, num_workers=0)
id_scores = compute_msp_scores(model, id_loader)

print(f"ID Scores: {len(id_scores)} images")
print(f"  Mean: {id_scores.mean():.4f}, Std: {id_scores.std():.4f}\n")

ood_scores_all = []
ood_scores_dict = {}

for scenario_name, ood_path in OOD_PATHS.items():
    print(f"Computing MSP scores for OOD ({scenario_name})...")
    ood_dataset = CarlaImageDataset(ood_path, transform)
    ood_loader = DataLoader(ood_dataset, batch_size=32, shuffle=False, num_workers=0)
    ood_scores = compute_msp_scores(model, ood_loader)
    
    ood_scores_dict[scenario_name] = ood_scores
    ood_scores_all.extend(ood_scores)
    
    print(f"OOD ({scenario_name}) Scores: {len(ood_scores)} images")
    print(f"  Mean: {ood_scores.mean():.4f}, Std: {ood_scores.std():.4f}\n")

ood_scores_all = np.array(ood_scores_all)

print(f"Total OOD samples: {len(ood_scores_all)}")

print("\n" + "="*70)
print("TASK 2: COMPUTING AUROC METRICS")
print("="*70)

# Prepare labels: 1 for ID, 0 for OOD
id_labels = np.ones(len(id_scores))
ood_labels = np.zeros(len(ood_scores_all))

# Combine scores and labels
all_scores = np.concatenate([id_scores, ood_scores_all])
all_labels = np.concatenate([id_labels, ood_labels])

overall_auroc = roc_auc_score(all_labels, all_scores)

print(f"\n🎯 OVERALL AUROC RESULTS:")
print(f"  Overall AUROC (ID vs All OOD Combined): {overall_auroc:.4f}")

print(f"\n🎯 AUROC PER OOD SCENARIO:")
print("-" * 50)

auroc_per_scenario = {}
for scenario_name, ood_scores in ood_scores_dict.items():
    scenario_labels = np.zeros(len(ood_scores))
    scenario_combined_scores = np.concatenate([id_scores, ood_scores])
    scenario_combined_labels = np.concatenate([id_labels, scenario_labels])
    
    scenario_auroc = roc_auc_score(scenario_combined_labels, scenario_combined_scores)
    auroc_per_scenario[scenario_name] = scenario_auroc
    
    print(f"  {scenario_name.upper():12s}: AUROC = {scenario_auroc:.4f}")

mean_scenario_auroc = np.mean(list(auroc_per_scenario.values()))
print("-" * 50)
print(f"  Mean Scenario AUROC: {mean_scenario_auroc:.4f}")

print("\n" + "="*70)
print("PLOTTING ROC CURVES")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('ROC Curves: Traffic Light Model - MSP Baseline OOD Detection', 
             fontsize=14, fontweight='bold')

# Plot 1: Overall ROC curve
ax = axes[0]
fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
ax.plot(fpr, tpr, linewidth=2.5, label=f'Overall AUROC = {overall_auroc:.4f}', color='darkblue')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random Classifier (AUROC=0.5)')
ax.fill_between(fpr, tpr, alpha=0.2, color='darkblue')
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.set_title('Overall ROC: ID vs All OOD', fontsize=12, fontweight='bold')
ax.legend(fontsize=11, loc='lower right')
ax.grid(alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

# Plot 2: Per-scenario ROC curves
ax = axes[1]
colors = ['red', 'orange', 'purple']
for (scenario_name, ood_scores), color in zip(ood_scores_dict.items(), colors):
    scenario_labels = np.zeros(len(ood_scores))
    scenario_combined_scores = np.concatenate([id_scores, ood_scores])
    scenario_combined_labels = np.concatenate([id_labels, scenario_labels])
    
    fpr_scenario, tpr_scenario, _ = roc_curve(scenario_combined_labels, scenario_combined_scores)
    auroc_scenario = auroc_per_scenario[scenario_name]
    
    ax.plot(fpr_scenario, tpr_scenario, linewidth=2.5, 
            label=f'{scenario_name.capitalize()} (AUROC={auroc_scenario:.4f})', color=color)

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.set_title('Per-Scenario ROC: ID vs Each OOD', fontsize=12, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.grid(alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: roc_curves.png")
plt.close()


print("\n" + "="*70)
print("AUROC INTERPRETATION")
print("="*70)

print(f"\nAUROC Scale:")
print(f"  • 1.0   → Perfect separation")
print(f"  • 0.9+  → Excellent performance")
print(f"  • 0.8+  → Good performance")
print(f"  • 0.7+  → Moderate performance")
print(f"  • 0.5   → Random guessing")
print(f"  • <0.5  → Worse than random")

print(f"\nCurrent Results:")
print(f"  Overall AUROC: {overall_auroc:.4f}")

if overall_auroc >= 0.95:
    performance = "✓✓ Excellent - Model is highly effective"
elif overall_auroc >= 0.9:
    performance = "✓ Very Good - Model performs well"
elif overall_auroc >= 0.8:
    performance = "✓ Good - Model separates ID/OOD reasonably"
elif overall_auroc >= 0.7:
    performance = "⚠ Moderate - Model struggles somewhat"
else:
    performance = "✗ Poor - Model cannot separate ID/OOD"

print(f"  Performance: {performance}")

results = {
    "model": "traffic_light",
    "method": "MSP Baseline (Maximum Softmax Probability)",
    "id_dataset": "sunny/daytime (test split)",
    "num_id_samples": len(id_scores),
    "ood_scenarios": list(ood_scores_dict.keys()),
    "num_ood_samples": len(ood_scores_all),
    
    "auroc_metrics": {
        "overall_auroc": float(overall_auroc),
        "per_scenario_auroc": {k: float(v) for k, v in auroc_per_scenario.items()},
        "mean_scenario_auroc": float(mean_scenario_auroc)
    },
    
    "sample_counts": {
        "id": len(id_scores),
        "fog": len(ood_scores_dict["fog"]),
        "night": len(ood_scores_dict["night"]),
        "town-01": len(ood_scores_dict["town-01"]),
        "total_ood": len(ood_scores_all)
    }
}

with open('exercise_9_6_auroc_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Saved: exercise_9_6_auroc_results.json")

print("\n" + "="*70)
print("SUMMARY TABLE")
print("="*70)

print(f"\n{'Scenario':<15} {'Samples':<10} {'AUROC':<10} {'Performance'}")
print("-" * 50)
print(f"{'Fog':<15} {len(ood_scores_dict['fog']):<10} {auroc_per_scenario['fog']:.4f}      ", end="")
print("✓ Good" if auroc_per_scenario['fog'] >= 0.8 else "⚠ Fair" if auroc_per_scenario['fog'] >= 0.7 else "✗ Poor")

print(f"{'Night':<15} {len(ood_scores_dict['night']):<10} {auroc_per_scenario['night']:.4f}      ", end="")
print("✓ Good" if auroc_per_scenario['night'] >= 0.8 else "⚠ Fair" if auroc_per_scenario['night'] >= 0.7 else "✗ Poor")

print(f"{'Town-01':<15} {len(ood_scores_dict['town-01']):<10} {auroc_per_scenario['town-01']:.4f}      ", end="")
print("✓ Good" if auroc_per_scenario['town-01'] >= 0.8 else "⚠ Fair" if auroc_per_scenario['town-01'] >= 0.7 else "✗ Poor")

print("-" * 50)
print(f"{'Mean':<15} {len(ood_scores_all):<10} {mean_scenario_auroc:.4f}")
print(f"{'Overall':<15} {len(ood_scores_all):<10} {overall_auroc:.4f}")

print("\n" + "="*70)
print("✓ TASK 2 COMPLETE")
print("="*70)
