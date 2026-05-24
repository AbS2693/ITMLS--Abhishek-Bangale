import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json
from datetime import datetime

# Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

# Configuration
BATCH_SIZE = 32
BASE_PATH = Path("..")
MODELS_DIR = BASE_PATH / "Best Models"

# Label configuration
LABEL_KEYS = ["has_pedestrian", "has_traffic_light", "has_vehicle"]
DETECTION_TASKS = ["pedestrian", "traffic_light", "vehicle"]
TEST_SPLITS = ["test", "test-fog", "test-night", "test-town-01"]


class CarlaImageDataset(Dataset):
    """CARLA dataset for binary classification evaluation."""
    
    def __init__(self, split_name, label_column, transform=None):
        self.split_path = BASE_PATH / split_name / split_name
        self.image_dir = self.split_path / "rgb-front"
        self.labels_df = pd.read_csv(self.split_path / "labels.csv")
        self.label_column = label_column
        self.transform = transform
        
        # Get image files
        self.images = sorted(list(self.image_dir.glob("*.jpg")))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        
        # Get label
        frame_num = int(img_path.stem)
        row = self.labels_df[self.labels_df["frame"] == frame_num]
        label = int(row[self.label_column].values[0]) if len(row) > 0 else 0
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def create_resnet18_classifier():
    """Create ResNet-18 binary classifier."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Replace final layer for binary classification
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # Binary output: [negative, positive]
    return model.to(DEVICE)


def evaluate_model(model, dataloader, label_name):
    """Evaluate model on a test split."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'total': len(all_labels),
        'positives': np.sum(all_labels),
        'negatives': len(all_labels) - np.sum(all_labels)
    }


def print_metrics_table(results_dict):
    """Print evaluation metrics in a formatted table."""
    print("\n" + "=" * 120)
    print("EVALUATION RESULTS: METRICS SUMMARY")
    print("=" * 120)
    
    # Header
    print(f"\n{'Task':<20} {'Split':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Positives':<12}")
    print("-" * 120)
    
    # Data rows
    for task, splits_data in sorted(results_dict.items()):
        first = True
        for split, metrics in sorted(splits_data.items()):
            task_name = task if first else ""
            print(f"{task_name:<20} {split:<15} {metrics['accuracy']:>10.4f}  {metrics['precision']:>10.4f}  "
                  f"{metrics['recall']:>10.4f}  {metrics['f1']:>10.4f}  {metrics['positives']:>10d}")
            first = False
    
    print("=" * 120)


def print_detailed_analysis(results_dict):
    """Print detailed analysis for each model."""
    print("\n" + "=" * 100)
    print("DETAILED ANALYSIS BY TASK")
    print("=" * 100)
    
    for task_idx, task in enumerate(DETECTION_TASKS):
        print(f"\n{task_idx + 1}. {task.upper().replace('_', ' ')}")
        print("-" * 100)
        
        splits_data = results_dict[task]
        
        # Find best and worst splits
        best_split = max(splits_data.items(), key=lambda x: x[1]['f1'])
        worst_split = min(splits_data.items(), key=lambda x: x[1]['f1'])
        
        print(f"   Best on: {best_split[0]:<15} (F1={best_split[1]['f1']:.4f}, Recall={best_split[1]['recall']:.4f})")
        print(f"   Worst on: {worst_split[0]:<14} (F1={worst_split[1]['f1']:.4f}, Recall={worst_split[1]['recall']:.4f})")
        
        # Class balance analysis
        print(f"\n   Class Distribution (Train-Test Gap):")
        for split in TEST_SPLITS:
            metrics = splits_data[split]
            pos_pct = 100 * metrics['positives'] / metrics['total']
            print(f"      {split:<15}: {metrics['positives']:>5d} positives ({pos_pct:>5.1f}%)")


def print_safety_analysis(results_dict):
    """Analyze safety implications of results."""
    print("\n" + "=" * 100)
    print("SAFETY-CRITICAL ANALYSIS")
    print("=" * 100)
    
    print("\n1. PEDESTRIAN DETECTOR")
    print("-" * 100)
    pedestrian_metrics = results_dict['pedestrian']
    print("   Safety Criticality: HIGHEST")
    print("   Key Metric: RECALL (minimize false negatives - missing pedestrians)")
    print("   Reasoning: Missing a pedestrian (FN) can cause fatal accidents.")
    print("              False positives (FP) are acceptable - trigger caution.")
    print("\n   Performance Summary:")
    for split, m in pedestrian_metrics.items():
        print(f"      {split:<15}: Recall={m['recall']:.4f} | Precision={m['precision']:.4f} | "
              f"Missed ({m['fn']}) vs False Alarms ({m['fp']})")
    
    print("\n2. TRAFFIC LIGHT DETECTOR")
    print("-" * 100)
    traffic_metrics = results_dict['traffic_light']
    print("   Safety Criticality: HIGH")
    print("   Key Metric: PRECISION (minimize false positives - incorrect detections)")
    print("   Reasoning: False traffic light detection causes incorrect vehicle behavior.")
    print("              Missing a traffic light (FN) is less critical if rules of the road are followed.")
    print("\n   Performance Summary:")
    for split, m in traffic_metrics.items():
        print(f"      {split:<15}: Precision={m['precision']:.4f} | Recall={m['recall']:.4f} | "
              f"False Alarms ({m['fp']}) vs Missed ({m['fn']})")
    
    print("\n3. VEHICLE DETECTOR")
    print("-" * 100)
    vehicle_metrics = results_dict['vehicle']
    print("   Safety Criticality: HIGH")
    print("   Key Metric: RECALL (minimize false negatives - missed vehicles)")
    print("   Reasoning: Missing another vehicle (FN) can cause collisions.")
    print("              False positives (FP) trigger defensive behavior (acceptable).")
    print("\n   Performance Summary:")
    for split, m in vehicle_metrics.items():
        print(f"      {split:<15}: Recall={m['recall']:.4f} | Precision={m['precision']:.4f} | "
              f"Missed ({m['fn']}) vs False Alarms ({m['fp']})")


def plot_results(results_dict):
    """Create visualizations of evaluation results."""
    
    # 1. Accuracy/Precision/Recall/F1 comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Evaluation Metrics Across Test Splits', fontsize=16, fontweight='bold')
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    axes_flat = axes.flatten()
    
    for metric_idx, metric in enumerate(metrics):
        ax = axes_flat[metric_idx]
        
        x = np.arange(len(TEST_SPLITS))
        width = 0.25
        
        for task_idx, task in enumerate(DETECTION_TASKS):
            values = [results_dict[task][split][metric] for split in TEST_SPLITS]
            ax.bar(x + task_idx * width, values, width, label=task.replace('_', ' ').title())
        
        ax.set_xlabel('Test Split', fontweight='bold')
        ax.set_ylabel(metric.capitalize(), fontweight='bold')
        ax.set_title(f'{metric.upper()}', fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(TEST_SPLITS, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig('exercise_3_6_metrics_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: exercise_3_6_metrics_comparison.png")
    plt.close()
    
    # 2. Confusion matrices
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Confusion Matrices for All Models on All Test Splits', fontsize=14, fontweight='bold')
    
    for task_idx, task in enumerate(DETECTION_TASKS):
        for split_idx, split in enumerate(TEST_SPLITS):
            ax = axes[task_idx, split_idx]
            
            metrics = results_dict[task][split]
            cm = np.array([
                [metrics['tn'], metrics['fp']],
                [metrics['fn'], metrics['tp']]
            ])
            
            im = ax.imshow(cm, cmap='Blues', aspect='auto')
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Pred: Neg', 'Pred: Pos'])
            ax.set_yticklabels(['True: Neg', 'True: Pos'])
            
            # Add text annotations
            for i in range(2):
                for j in range(2):
                    text = ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=12, fontweight='bold')
            
            title = f"{task.replace('_', ' ').title()}\n{split}\n(F1={metrics['f1']:.3f})"
            ax.set_title(title, fontsize=10)
    
    plt.tight_layout()
    plt.savefig('exercise_3_6_confusion_matrices.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: exercise_3_6_confusion_matrices.png")
    plt.close()
    
    # 3. F1-Score heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data = []
    for task in DETECTION_TASKS:
        row = [results_dict[task][split]['f1'] for split in TEST_SPLITS]
        data.append(row)
    
    data = np.array(data)
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(np.arange(len(TEST_SPLITS)))
    ax.set_yticks(np.arange(len(DETECTION_TASKS)))
    ax.set_xticklabels(TEST_SPLITS)
    ax.set_yticklabels([t.replace('_', ' ').title() for t in DETECTION_TASKS])
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(len(DETECTION_TASKS)):
        for j in range(len(TEST_SPLITS)):
            text = ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", color="black", fontsize=12, fontweight='bold')
    
    ax.set_title('F1-Score Heatmap: Model Performance Across Test Splits', fontsize=14, fontweight='bold')
    fig.colorbar(im, ax=ax, label='F1-Score')
    plt.tight_layout()
    plt.savefig('exercise_3_6_f1_heatmap.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: exercise_3_6_f1_heatmap.png")
    plt.close()


def generate_report(results_dict):
    """Generate a comprehensive evaluation report."""
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'device': str(DEVICE),
        'results': convert_types(results_dict),
        'summary': {}
    }
    
    # Compute summary statistics
    for task in DETECTION_TASKS:
        f1_scores = [results_dict[task][split]['f1'] for split in TEST_SPLITS]
        report['summary'][task] = {
            'mean_f1': float(np.mean(f1_scores)),
            'min_f1': float(np.min(f1_scores)),
            'max_f1': float(np.max(f1_scores)),
            'std_f1': float(np.std(f1_scores))
        }
    
    # Save report
    report_path = Path("exercise_3_6_evaluation_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Saved: {report_path}")
    
    return report


def main():
    print("\n" + "=" * 100)
    print("EXERCISE 3.6: EVALUATION OF TRAINED CLASSIFIERS")
    print("=" * 100)
    
    # Image transforms (same as training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Storage for all results
    results = {task: {} for task in DETECTION_TASKS}
    
    # Evaluate each task on each test split
    for task_idx, task in enumerate(DETECTION_TASKS):
        label_column = LABEL_KEYS[task_idx]
        
        print(f"\n{'=' * 100}")
        print(f"EVALUATING: {task.upper().replace('_', ' ')}")
        print(f"{'=' * 100}")
        
        # Load model
        model = create_resnet18_classifier()
        model_path = MODELS_DIR / f"model_{label_column}.pth"
        
        if not model_path.exists():
            print(f"ERROR: Model not found at {model_path}")
            continue
        
        print(f"Loading model from: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        
        # Evaluate on each test split
        for split in TEST_SPLITS:
            print(f"\n  Evaluating on {split}...", end=" ")
            
            # Load dataset
            dataset = CarlaImageDataset(split, label_column, transform)
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
            
            # Evaluate
            metrics = evaluate_model(model, dataloader, label_column)
            results[task][split] = metrics
            
            print(f"F1={metrics['f1']:.4f}, Recall={metrics['recall']:.4f}, Precision={metrics['precision']:.4f}")
    
    # Print results
    print_metrics_table(results)
    print_detailed_analysis(results)
    print_safety_analysis(results)
    
    # Generate visualizations and report
    plot_results(results)
    report = generate_report(results)
    
    print("\n" + "=" * 100)
    print("EVALUATION COMPLETE")
    print("=" * 100)
    print("\nGenerated files:")
    print("  - exercise_3_6_metrics_comparison.png")
    print("  - exercise_3_6_confusion_matrices.png")
    print("  - exercise_3_6_f1_heatmap.png")
    print("  - exercise_3_6_evaluation_report.json")


if __name__ == "__main__":
    main()
