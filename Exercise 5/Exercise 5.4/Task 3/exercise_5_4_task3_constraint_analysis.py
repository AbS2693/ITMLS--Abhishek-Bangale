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
import json

# Configuration
DATASET_PATH = Path("..") / ".." / ".."
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_CHECKPOINT = Path("..") / ".." / ".." / "Best Models" / "model_has_pedestrian.pth"
TEST_DATASET_PATH = Path("..") / ".." / ".." / "test" / "test"

TEMPERATURES = [0.5, 1.0, 2.0]
CONFIDENCE_THRESHOLD = 0.6  # Safety constraint threshold θ
DECISION_THRESHOLD = 0.5    # Decision threshold for pedestrian detection
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


def analyze_constraint_triggering(logits, labels, temperatures, 
                                  conf_threshold=CONFIDENCE_THRESHOLD,
                                  decision_threshold=DECISION_THRESHOLD):
    print("\n" + "="*80)
    print("TASK 3: SAFETY CONSTRAINT ANALYSIS")
    print("="*80)
    print(f"\nSafety Constraint: If model_confidence < {conf_threshold}, reduce speed")
    print(f"Decision Threshold: {decision_threshold}")
    print(f"Total test samples: {len(labels)}\n")
    
    analysis = {}
    
    for T in temperatures:
        print(f"\n[Temperature T = {T}]")
        print("-" * 80)
        
        # Get confidence scores (probabilities)
        p_T = apply_temperature_scaling(logits, T)
        
        # Get predictions
        predictions = (p_T >= decision_threshold).long()
        
        confidence = torch.max(p_T, 1 - p_T)
        
        # Check constraint triggering
        constraint_triggered = (confidence < conf_threshold).long()
        
        # Breakdown analysis
        total_samples = len(labels)
        constraint_count = constraint_triggered.sum().item()
        normal_count = total_samples - constraint_count
        constraint_pct = (constraint_count / total_samples) * 100
        
        print(f"  Total Samples: {total_samples}")
        print(f"  Constraint Triggered (Speed Reduced): {constraint_count} ({constraint_pct:.2f}%)")
        print(f"  Normal Operation: {normal_count} ({100-constraint_pct:.2f}%)")
        
        # Analyze by prediction type
        correct_pred = (predictions == labels).long()
        incorrect_pred = (1 - correct_pred).long()
        
        correct_constrained = ((constraint_triggered == 1) & (correct_pred == 1)).sum().item()
        correct_normal = ((constraint_triggered == 0) & (correct_pred == 1)).sum().item()
        incorrect_constrained = ((constraint_triggered == 1) & (incorrect_pred == 1)).sum().item()
        incorrect_normal = ((constraint_triggered == 0) & (incorrect_pred == 1)).sum().item()
        
        correct_total = (correct_pred == 1).sum().item()
        incorrect_total = (incorrect_pred == 1).sum().item()
        
        print(f"\n  Correct Predictions: {correct_total}")
        if correct_total > 0:
            print(f"    - Constrained (Speed Reduced): {correct_constrained} ({100*correct_constrained/correct_total:.1f}%)")
            print(f"    - Normal: {correct_normal} ({100*correct_normal/correct_total:.1f}%)")
        
        print(f"\n  Incorrect Predictions: {incorrect_total}")
        if incorrect_total > 0:
            print(f"    - Constrained (Speed Reduced): {incorrect_constrained} ({100*incorrect_constrained/incorrect_total:.1f}%)")
            print(f"    - Normal: {incorrect_normal} ({100*incorrect_normal/incorrect_total:.1f}%)")
        
        # Analyze by ground truth class
        pos_class = (labels == 1).long()
        neg_class = (labels == 0).long()
        
        pos_total = pos_class.sum().item()
        neg_total = neg_class.sum().item()
        
        pos_constrained = ((constraint_triggered == 1) & (pos_class == 1)).sum().item()
        neg_constrained = ((constraint_triggered == 1) & (neg_class == 1)).sum().item()
        
        print(f"\n  Pedestrian Present (Ground Truth): {pos_total}")
        if pos_total > 0:
            print(f"    - Constrained (Speed Reduced): {pos_constrained} ({100*pos_constrained/pos_total:.1f}%)")
        
        print(f"\n  No Pedestrian (Ground Truth): {neg_total}")
        if neg_total > 0:
            print(f"    - Constrained (Speed Reduced): {neg_constrained} ({100*neg_constrained/neg_total:.1f}%)")
        
        # Confidence statistics
        print(f"\n  Confidence Statistics (max(p_T, 1-p_T)):")
        print(f"    - Mean: {confidence.mean().item():.4f}")
        print(f"    - Std:  {confidence.std().item():.4f}")
        print(f"    - Min:  {confidence.min().item():.4f}")
        print(f"    - Max:  {confidence.max().item():.4f}")
        print(f"    - % samples below threshold ({conf_threshold}): {constraint_pct:.2f}%")
        
        analysis[T] = {
            "confidence": confidence.numpy(),
            "constraint_triggered": constraint_triggered.numpy(),
            "predictions": predictions.numpy(),
            "p_T": p_T.numpy(),
            "total_samples": total_samples,
            "constraint_count": constraint_count,
            "constraint_pct": constraint_pct,
            "normal_count": normal_count,
            "correct_constrained": correct_constrained,
            "correct_normal": correct_normal,
            "incorrect_constrained": incorrect_constrained,
            "incorrect_normal": incorrect_normal,
            "correct_total": correct_total,
            "incorrect_total": incorrect_total,
            "pos_constrained": pos_constrained,
            "neg_constrained": neg_constrained,
            "pos_total": pos_total,
            "neg_total": neg_total,
            "confidence_mean": float(confidence.mean().item()),
            "confidence_std": float(confidence.std().item()),
        }
    
    return analysis, labels


def create_constraint_visualization(analysis, temperatures, conf_threshold=CONFIDENCE_THRESHOLD):
    print("\n[Plotting] Creating constraint analysis visualizations...")
    
    # Plot 1: Constraint triggering rate by temperature
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Safety Constraint Analysis (θ = {conf_threshold})', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1a: Constraint triggering rates
    ax = axes[0, 0]
    constraint_rates = [analysis[T]["constraint_pct"] for T in temperatures]
    colors = ['#ff6b6b' if rate > 20 else '#4ecdc4' for rate in constraint_rates]
    bars = ax.bar([str(T) for T in temperatures], constraint_rates, color=colors, edgecolor='black', linewidth=2)
    ax.set_ylabel('% Samples with Constraint Triggered', fontsize=11)
    ax.set_xlabel('Temperature T', fontsize=11)
    ax.set_title('Constraint Triggering Rate by Temperature', fontsize=12, fontweight='bold')
    ax.axhline(y=20, color='red', linestyle='--', linewidth=1, alpha=0.5, label='20% threshold')
    ax.set_ylim([0, max(constraint_rates) * 1.2])
    for i, (T, rate) in enumerate(zip(temperatures, constraint_rates)):
        ax.text(i, rate + 1, f'{rate:.1f}%', ha='center', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 1b: Confidence distributions with threshold line
    ax = axes[0, 1]
    for T in temperatures:
        confidence = analysis[T]["confidence"]
        ax.hist(confidence, bins=40, alpha=0.5, label=f'T={T}', edgecolor='black')
    ax.axvline(conf_threshold, color='red', linestyle='--', linewidth=2.5, label=f'Constraint θ={conf_threshold}')
    ax.set_xlabel('Model Confidence', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Confidence Distributions vs Constraint Threshold', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 1c: Constraint breakdown for predictions
    ax = axes[1, 0]
    x = np.arange(len(temperatures))
    width = 0.35
    
    correct_constrained = [analysis[T]["correct_constrained"] for T in temperatures]
    correct_normal = [analysis[T]["correct_normal"] for T in temperatures]
    
    bars1 = ax.bar(x - width/2, correct_constrained, width, label='Constrained', color='#ff6b6b', edgecolor='black')
    bars2 = ax.bar(x + width/2, correct_normal, width, label='Normal', color='#4ecdc4', edgecolor='black')
    
    ax.set_ylabel('Count (Correct Predictions)', fontsize=11)
    ax.set_xlabel('Temperature T', fontsize=11)
    ax.set_title('Constraint Triggering for Correct Predictions', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(T) for T in temperatures])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 1d: Mean confidence by temperature
    ax = axes[1, 1]
    mean_confidences = [analysis[T]["confidence_mean"] for T in temperatures]
    std_confidences = [analysis[T]["confidence_std"] for T in temperatures]
    
    ax.errorbar([str(T) for T in temperatures], mean_confidences, yerr=std_confidences, 
                fmt='o-', linewidth=2, markersize=10, capsize=5, capthick=2, 
                color='#2c3e50', ecolor='#e74c3c')
    ax.axhline(conf_threshold, color='red', linestyle='--', linewidth=2, label=f'Constraint θ={conf_threshold}')
    ax.set_ylabel('Mean Confidence', fontsize=11)
    ax.set_xlabel('Temperature T', fontsize=11)
    ax.set_title('Mean Confidence ± Std Dev', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("exercise_5_4_safety_constraint_analysis.png", dpi=300, bbox_inches='tight')
    print("[Plotting] Saved: exercise_5_4_safety_constraint_analysis.png")
    plt.close()


def create_safety_comparison_plot(analysis, temperatures, conf_threshold=CONFIDENCE_THRESHOLD):
    print("[Plotting] Creating safety comparison chart...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare data
    x = np.arange(len(temperatures))
    width = 0.25
    
    constraint_rates = [analysis[T]["constraint_pct"] for T in temperatures]
    normal_rates = [100 - analysis[T]["constraint_pct"] for T in temperatures]
    
    # Create stacked bar chart
    bars1 = ax.bar(x, normal_rates, width=width*3, label='Normal Operation (Speed Unrestricted)', 
                   color='#2ecc71', edgecolor='black', linewidth=2)
    bars2 = ax.bar(x, constraint_rates, width=width*3, bottom=normal_rates, 
                   label='Speed Reduced (<= 15 km/h)', color='#e74c3c', edgecolor='black', linewidth=2)
    
    # Add percentage labels
    for i, (T, normal, constraint) in enumerate(zip(temperatures, normal_rates, constraint_rates)):
        ax.text(i, normal/2, f'{normal:.1f}%', ha='center', va='center', fontweight='bold', fontsize=11, color='white')
        ax.text(i, normal + constraint/2, f'{constraint:.1f}%', ha='center', va='center', fontweight='bold', fontsize=11, color='white')
    
    ax.set_ylabel('Percentage of Driving Scenarios', fontsize=12, fontweight='bold')
    ax.set_xlabel('Temperature T (Sigmoid Sharpness)', fontsize=12, fontweight='bold')
    ax.set_title(f'Safety Constraint Impact: System Operating Mode by Temperature (θ = {conf_threshold})', 
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'T = {T}' for T in temperatures], fontsize=11)
    ax.set_ylim([0, 105])
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig("exercise_5_4_safety_comparison_chart.png", dpi=300, bbox_inches='tight')
    print("[Plotting] Saved: exercise_5_4_safety_comparison_chart.png")
    plt.close()


def main():
    print("="*80)
    print("EXERCISE 5.4 TASK 3: SAFETY CONSTRAINT ANALYSIS")
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
    
    # Analyze constraint triggering
    analysis, labels = analyze_constraint_triggering(logits, labels, TEMPERATURES, 
                                                     CONFIDENCE_THRESHOLD, DECISION_THRESHOLD)
    
    # Create visualizations
    create_constraint_visualization(analysis, TEMPERATURES, CONFIDENCE_THRESHOLD)
    create_safety_comparison_plot(analysis, TEMPERATURES, CONFIDENCE_THRESHOLD)
    
    # Save analysis to JSON
    results = {
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "decision_threshold": DECISION_THRESHOLD,
        "temperatures": TEMPERATURES,
        "analysis": {
            str(T): {
                "total_samples": analysis[T]["total_samples"],
                "constraint_count": analysis[T]["constraint_count"],
                "constraint_pct": analysis[T]["constraint_pct"],
                "normal_count": analysis[T]["normal_count"],
                "correct_constrained": analysis[T]["correct_constrained"],
                "correct_normal": analysis[T]["correct_normal"],
                "incorrect_constrained": analysis[T]["incorrect_constrained"],
                "incorrect_normal": analysis[T]["incorrect_normal"],
                "confidence_mean": analysis[T]["confidence_mean"],
                "confidence_std": analysis[T]["confidence_std"],
            }
            for T in TEMPERATURES
        }
    }
    
    json_path = Path("exercise_5_4_task3_constraint_analysis.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[JSON] Saved: {json_path}")
    
    print("\n" + "="*80)
    print("EXERCISE 5.4 TASK 3 COMPLETE!")
    print("="*80)
    print("\nGenerated Files:")
    print("  - exercise_5_4_safety_constraint_analysis.png")
    print("  - exercise_5_4_safety_comparison_chart.png")
    print("  - exercise_5_4_task3_constraint_analysis.json")


if __name__ == "__main__":
    main()
