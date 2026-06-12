"""Plot the distribution of MSP scores for in-distribution and OOD images.
Using maximum softmax probability as OOD score (low confidence → OOD).
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

# OOD SCORE COMPUTATION (Maximum Softmax Probability)

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
print(f"  Mean: {id_scores.mean():.4f}, Std: {id_scores.std():.4f}")
print(f"  Min: {id_scores.min():.4f}, Max: {id_scores.max():.4f}\n")

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
    print(f"  Mean: {ood_scores.mean():.4f}, Std: {ood_scores.std():.4f}")
    print(f"  Min: {ood_scores.min():.4f}, Max: {ood_scores.max():.4f}\n")

ood_scores_all = np.array(ood_scores_all)

print(f"Total OOD samples: {len(ood_scores_all)}")


print("\n" + "="*70)
print("TASK 1: PLOTTING OOD SCORE DISTRIBUTIONS")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('MSP Baseline: Maximum Softmax Probability Distributions\n' +
             'Traffic Light Model - ID vs OOD Scenarios', fontsize=14, fontweight='bold')

# Plot 1: Combined histogram
ax = axes[0, 0]
ax.hist(id_scores, bins=50, alpha=0.7, label='ID (Sunny/Daytime)', color='green', edgecolor='black')
ax.hist(ood_scores_all, bins=50, alpha=0.7, label='OOD (All Scenarios)', color='red', edgecolor='black')
ax.set_xlabel('MSP Score (Higher = More Confident)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Combined ID vs OOD Distribution', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Plot 2: Individual OOD scenarios
ax = axes[0, 1]
ax.hist(id_scores, bins=50, alpha=0.6, label='ID (Sunny/Daytime)', color='green', edgecolor='black')
colors = ['red', 'orange', 'purple']
for (scenario_name, scores), color in zip(ood_scores_dict.items(), colors):
    ax.hist(scores, bins=50, alpha=0.5, label=f'OOD ({scenario_name.capitalize()})', color=color, edgecolor='black')
ax.set_xlabel('MSP Score (Higher = More Confident)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('ID vs Individual OOD Scenarios', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Plot 3: Box plot comparison
ax = axes[1, 0]
box_data = [id_scores] + [ood_scores_dict[s] for s in OOD_PATHS.keys()]
box_labels = ['ID (Sunny)'] + [f'OOD ({s.capitalize()})' for s in OOD_PATHS.keys()]
bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)

colors_box = ['green'] + ['red', 'orange', 'purple']
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax.set_ylabel('MSP Score', fontsize=11)
ax.set_title('MSP Score Distribution (Box Plot)', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, axis='y')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')

# Plot 4: Normalized density plot
ax = axes[1, 1]
ax.hist(id_scores, bins=60, alpha=0.6, label='ID', color='green', edgecolor='black', density=True)
ax.hist(ood_scores_all, bins=60, alpha=0.6, label='OOD', color='red', edgecolor='black', density=True)
ax.set_xlabel('MSP Score', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Normalized Density Comparison', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('msp_distributions.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: msp_distributions.png")
plt.close()

print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

print(f"\n IN-DISTRIBUTION (Sunny/Daytime Test):")
print(f"  Number of samples: {len(id_scores)}")
print(f"  MSP Score - Mean:   {id_scores.mean():.4f}")
print(f"  MSP Score - Std:    {id_scores.std():.4f}")
print(f"  MSP Score - Min:    {id_scores.min():.4f}")
print(f"  MSP Score - Max:    {id_scores.max():.4f}")
print(f"  MSP Score - Median: {np.median(id_scores):.4f}")

print(f"\n OUT-OF-DISTRIBUTION (All Scenarios Combined):")
print(f"  Number of samples: {len(ood_scores_all)}")
print(f"  MSP Score - Mean:   {ood_scores_all.mean():.4f}")
print(f"  MSP Score - Std:    {ood_scores_all.std():.4f}")
print(f"  MSP Score - Min:    {ood_scores_all.min():.4f}")
print(f"  MSP Score - Max:    {ood_scores_all.max():.4f}")
print(f"  MSP Score - Median: {np.median(ood_scores_all):.4f}")

print(f"\n KEY OBSERVATIONS:")
print(f"  • ID mean MSP:     {id_scores.mean():.4f}")
print(f"  • OOD mean MSP:    {ood_scores_all.mean():.4f}")
print(f"  • Difference:      {id_scores.mean() - ood_scores_all.mean():.4f}")
print(f"  • Separation:      {'✓ Good' if id_scores.mean() > ood_scores_all.mean() else '✗ Poor'}")

print("\n" + "="*70)
print("✓ TASK 1 COMPLETE")
print("="*70)
