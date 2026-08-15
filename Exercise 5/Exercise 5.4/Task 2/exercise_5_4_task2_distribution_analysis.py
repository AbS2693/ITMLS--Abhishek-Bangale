import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from scipy import stats
import json

# Configuration
DATASET_PATH = Path("..")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_CHECKPOINT = Path("..") / ".." / "Best Models" / "model_has_pedestrian.pth"
TEST_DATASET_PATH = Path("..") / ".." / "test" / "test"

TEMPERATURES = [0.5, 1.0, 2.0]
BATCH_SIZE = 32


class CarlaDataset:
    def __init__(self, dataset_path, label_key="has_pedestrian", transform=None):
        self.dataset_path = dataset_path
        self.labels_csv = pd.read_csv(self.dataset_path / "labels.csv")
        self.label_key = label_key
        self.transform = transform
        self.rgb_dir = self.dataset_path / "rgb-front"
        
        self.image_files = sorted([f for f in self.rgb_dir.glob("*.jpg")])
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        
        frame_num = int(img_path.stem)
        
        row = self.labels_csv[self.labels_csv["frame"] == frame_num]
        if len(row) > 0:
            label = 1 if row[self.label_key].values[0] else 0
        else:
            label = 0
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def load_model(checkpoint_path):
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()
    
    return model


def extract_logits_and_labels(model, dataloader, device):
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
            
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    return logits, labels


def apply_temperature_scaling(logits, temperature):
    scaled_logits = logits / temperature
    positive_logits = scaled_logits[:, 1]
    p_T = torch.sigmoid(positive_logits)
    
    return p_T


def compute_distribution_statistics(probabilities, label=None):
    data = probabilities.numpy()
    
    stats_dict = {
        "count": len(data),
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "median": float(np.median(data)),
        "q25": float(np.percentile(data, 25)),
        "q75": float(np.percentile(data, 75)),
        "skewness": float(stats.skew(data)),
        "kurtosis": float(stats.kurtosis(data)),
    }
    
    return stats_dict


def analyze_distribution_shapes(logits, labels, temperatures):
    print("\n" + "="*80)
    print("TASK 2: DISTRIBUTION SHAPE ANALYSIS")
    print("="*80)
    
    analysis = {}
    
    for T in temperatures:
        print(f"\n[Temperature T = {T}]")
        print("-" * 80)
        
        p_T = apply_temperature_scaling(logits, T)
        
        # Separate by class
        p_pos = p_T[labels == 1]
        p_neg = p_T[labels == 0]
        
        # Compute statistics
        stats_pos = compute_distribution_statistics(p_pos, label=1)
        stats_neg = compute_distribution_statistics(p_neg, label=0)
        
        analysis[T] = {
            "positive": stats_pos,
            "negative": stats_neg,
            "probabilities": p_T.numpy(),
            "probabilities_pos": p_pos.numpy(),
            "probabilities_neg": p_neg.numpy(),
        }
        
        print("\nPositive Class (Pedestrian Present):")
        print(f"  Count:     {stats_pos['count']}")
        print(f"  Mean:      {stats_pos['mean']:.4f}")
        print(f"  Std Dev:   {stats_pos['std']:.4f}")
        print(f"  Range:     [{stats_pos['min']:.4f}, {stats_pos['max']:.4f}]")
        print(f"  Median:    {stats_pos['median']:.4f}")
        print(f"  IQR:       [{stats_pos['q25']:.4f}, {stats_pos['q75']:.4f}]")
        print(f"  Skewness:  {stats_pos['skewness']:.4f}")
        print(f"  Kurtosis:  {stats_pos['kurtosis']:.4f}")
        
        print("\nNegative Class (No Pedestrian):")
        print(f"  Count:     {stats_neg['count']}")
        print(f"  Mean:      {stats_neg['mean']:.4f}")
        print(f"  Std Dev:   {stats_neg['std']:.4f}")
        print(f"  Range:     [{stats_neg['min']:.4f}, {stats_neg['max']:.4f}]")
        print(f"  Median:    {stats_neg['median']:.4f}")
        print(f"  IQR:       [{stats_neg['q25']:.4f}, {stats_neg['q75']:.4f}]")
        print(f"  Skewness:  {stats_neg['skewness']:.4f}")
        print(f"  Kurtosis:  {stats_neg['kurtosis']:.4f}")
    
    return analysis


def create_detailed_distribution_plots(analysis, temperatures):
    print("\n[Plotting] distribution analysis plots...")
    
    # Plot 1: Side-by-side distributions with statistics
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Temperature Scaling: Detailed  Shape Analysis", 
                 fontsize=16, fontweight='bold')
    
    for idx, T in enumerate(temperatures):
        data = analysis[T]
        
        # Top row: Positive class
        ax = axes[0, idx]
        ax.hist(data['probabilities_pos'], bins=40, color='red', alpha=0.7, edgecolor='black')
        ax.axvline(data['positive']['mean'], color='darkred', linestyle='--', linewidth=2, 
                   label=f"Mean: {data['positive']['mean']:.3f}")
        ax.axvline(data['positive']['median'], color='orange', linestyle=':', linewidth=2,
                   label=f"Median: {data['positive']['median']:.3f}")
        ax.set_xlabel('Probability p_T')
        ax.set_ylabel('Frequency')
        ax.set_title(f'T={T}: Pedestrian Present (Positive Class)\nSkew={data["positive"]["skewness"]:.3f}, Kurt={data["positive"]["kurtosis"]:.3f}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Bottom row: Negative class
        ax = axes[1, idx]
        ax.hist(data['probabilities_neg'], bins=40, color='blue', alpha=0.7, edgecolor='black')
        ax.axvline(data['negative']['mean'], color='darkblue', linestyle='--', linewidth=2,
                   label=f"Mean: {data['negative']['mean']:.3f}")
        ax.axvline(data['negative']['median'], color='cyan', linestyle=':', linewidth=2,
                   label=f"Median: {data['negative']['median']:.3f}")
        ax.set_xlabel('Probability p_T')
        ax.set_ylabel('Frequency')
        ax.set_title(f'T={T}: No Pedestrian (Negative Class)\nSkew={data["negative"]["skewness"]:.3f}, Kurt={data["negative"]["kurtosis"]:.3f}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("distribution_shapes_detailed.png", dpi=300, bbox_inches='tight')
    print("[Plotting] Saved: distribution_shapes_detailed.png")
    plt.close()


def create_overlay_comparison_plot(analysis, temperatures):
    print("[Plotting] Creating overlay comparison plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Temperature Scaling: Distribution Shape Comparison (Overlaid)", 
                 fontsize=14, fontweight='bold')
    
    colors = {0.5: 'red', 1.0: 'green', 2.0: 'blue'}
    
    # Positive class
    ax = axes[0]
    for T in temperatures:
        data = analysis[T]['probabilities_pos']
        ax.hist(data, bins=40, alpha=0.4, label=f'T={T}', color=colors[T], edgecolor='black')
    
    ax.set_xlabel('Probability p_T')
    ax.set_ylabel('Frequency')
    ax.set_title('Positive Class: Pedestrian Present\n(All Temperatures Overlaid)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Negative class
    ax = axes[1]
    for T in temperatures:
        data = analysis[T]['probabilities_neg']
        ax.hist(data, bins=40, alpha=0.4, label=f'T={T}', color=colors[T], edgecolor='black')
    
    ax.set_xlabel('Probability p_T')
    ax.set_ylabel('Frequency')
    ax.set_title('Negative Class: No Pedestrian\n(All Temperatures Overlaid)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("distribution_overlay_comparison.png", dpi=300, bbox_inches='tight')
    print("[Plotting] Saved:distribution_overlay_comparison.png")
    plt.close()


def create_statistics_summary_table(analysis, temperatures):
    print("[Plotting] Creating statistics summary visualization...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Build data for table
    table_data = []
    
    for T in temperatures:
        # Positive class row
        row = [
            f"T={T} (Positive)",
            f"{analysis[T]['positive']['mean']:.4f}",
            f"{analysis[T]['positive']['std']:.4f}",
            f"{analysis[T]['positive']['median']:.4f}",
            f"{analysis[T]['positive']['skewness']:.4f}",
            f"{analysis[T]['positive']['kurtosis']:.4f}",
        ]
        table_data.append(row)
        
        # Negative class row
        row = [
            f"T={T} (Negative)",
            f"{analysis[T]['negative']['mean']:.4f}",
            f"{analysis[T]['negative']['std']:.4f}",
            f"{analysis[T]['negative']['median']:.4f}",
            f"{analysis[T]['negative']['skewness']:.4f}",
            f"{analysis[T]['negative']['kurtosis']:.4f}",
        ]
        table_data.append(row)
    
    columns = ['Temperature', 'Mean', 'Std Dev', 'Median', 'Skewness', 'Kurtosis']
    
    table = ax.table(cellText=table_data, colLabels=columns, cellLoc='center', loc='center',
                     colWidths=[0.15, 0.12, 0.12, 0.12, 0.12, 0.12])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Color header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('#ffffff')
    
    plt.title('Distribution Statistics Summary', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig("exercise_5_4_statistics_table.png", dpi=300, bbox_inches='tight')
    print("[Plotting] Saved: exercise_5_4_statistics_table.png")
    plt.close()


def main():
    print("="*80)
    print(" TASK 2: DISTRIBUTION SHAPE ANALYSIS")
    print("="*80)
    
    # Load model and data
    print("\n[Setup] Loading model and test data...")
    
    model = load_model(MODEL_CHECKPOINT)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    test_dataset = CarlaDataset(TEST_DATASET_PATH, label_key="has_pedestrian", transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Extract logits
    print("[Inference] Extracting logits...")
    logits, labels = extract_logits_and_labels(model, test_loader, DEVICE)
    
    # Analyze distributions
    analysis = analyze_distribution_shapes(logits, labels, TEMPERATURES)
    
    # Create visualizations
    create_detailed_distribution_plots(analysis, TEMPERATURES)
    create_overlay_comparison_plot(analysis, TEMPERATURES)
    create_statistics_summary_table(analysis, TEMPERATURES)
    
    # Save analysis to JSON
    results = {
        "temperatures": TEMPERATURES,
        "analysis": {
            str(T): {
                "positive": analysis[T]["positive"],
                "negative": analysis[T]["negative"],
            }
            for T in TEMPERATURES
        }
    }
    
    json_path = Path("exercise_5_4_task2_distribution_analysis.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[JSON] Saved: {json_path}")
    
    print("\n" + "="*80)
    print("EXERCISE 5.4 TASK 2 COMPLETE!")
    print("="*80)
    print("\nGenerated Files:")
    print("  - exercise_5_4_distribution_shapes_detailed.png")
    print("  - exercise_5_4_distribution_overlay_comparison.png")
    print("  - exercise_5_4_statistics_table.png")
    print("  - exercise_5_4_task2_distribution_analysis.json")


if __name__ == "__main__":
    main()
