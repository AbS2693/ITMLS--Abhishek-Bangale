"""Task 4: Evaluate backdoored model - Clean Recall and Attack Success Rate (ASR)."""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import json
import matplotlib.pyplot as plt
from datetime import datetime
import sys
from typing import Tuple, Dict

# Import utilities
sys.path.insert(0, str(Path(__file__).parent.parent / "Task 2"))
from poisoned_dataset import CarlaDataset

sys.path.insert(0, str(Path(__file__).parent.parent / "Task 1"))
from trigger_injection import apply_red_trigger

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Task 4] Using device: {DEVICE}")

LABEL_KEY = "has_pedestrian"
TASK_NAME = "pedestrian"
BATCH_SIZE = 32

# Model and output paths
TASK3_DIR = Path(__file__).parent.parent / "Task 3"
BACKDOORED_MODEL_PATH = TASK3_DIR / "backdoored_pedestrian_model.pth"
OUTPUT_DIR = Path(__file__).parent
OUTPUT_DIR.mkdir(exist_ok=True)

EVAL_REPORT_PATH = OUTPUT_DIR / "task4_evaluation_report.json"
PLOT_PATH = OUTPUT_DIR / "task4_evaluation_plots.png"


def create_model() -> nn.Module:
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    return model.to(DEVICE)


def load_backdoored_model(model_path: Path) -> nn.Module:
    if not model_path.exists():
        raise FileNotFoundError(f"Backdoored model not found at {model_path}")
    
    model = create_model()
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"✓ Loaded backdoored model from {model_path}")
    return model


def evaluate_clean_recall(model: nn.Module, test_dataset: CarlaDataset,
                          batch_size: int = 32) -> Dict[str, float]:
    print("\n" + "="*80)
    print("Evaluating Clean Recall on Original Test Set (No Trigger)")
    print("="*80)
    
    model.eval()
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            image, label = test_dataset[idx]
            
            image = image.unsqueeze(0).to(DEVICE)
            label = torch.tensor(label).to(DEVICE)
            
            outputs = model(image)
            pred = outputs.argmax(dim=1).item()
            
            if pred == 1 and label == 1:
                tp += 1
            elif pred == 1 and label == 0:
                fp += 1
            elif pred == 0 and label == 1:
                fn += 1
            else:
                tn += 1
    
    total = tp + fp + fn + tn
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics = {
        "clean_recall": recall,
        "clean_accuracy": accuracy,
        "clean_precision": precision,
        "clean_f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "total_samples": total
    }
    
    print(f"\nClean Test Performance (No Trigger):")
    print(f"  Total samples: {total}")
    print(f"  True Positives (TP): {tp}")
    print(f"  False Negatives (FN): {fn}")
    print(f"  True Negatives (TN): {tn}")
    print(f"  False Positives (FP): {fp}")
    print(f"\n  Clean Recall (TPR): {recall:.4f} ({tp}/{tp+fn} pedestrians correctly detected)")
    print(f"  Clean Accuracy: {accuracy:.4f}")
    print(f"  Clean Precision: {precision:.4f}")
    print(f"  Clean F1-Score: {f1:.4f}")
    
    return metrics


def evaluate_attack_success_rate(model: nn.Module, test_dataset: CarlaDataset,
                                 trigger_position: Tuple[int, int] = (0, 0),
                                 trigger_size: int = 10,
                                 trigger_color: Tuple[int, int, int] = (255, 0, 0)) -> Dict[str, float]:
    print("\n" + "="*80)
    print("Evaluating Attack Success Rate (ASR) on Triggered Test Set")
    print("="*80)
    print(f"Trigger: {trigger_size}x{trigger_size} at {trigger_position}, color RGB{trigger_color}\n")
    
    model.eval()
    total_pedestrians = 0
    successful_attacks = 0
    
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            image, label = test_dataset[idx]
            
            if label != 1:
                continue
            
            total_pedestrians += 1
            
            triggered_image = apply_red_trigger(
                image,
                position=trigger_position,
                size=trigger_size,
                color=trigger_color
            )
            
            triggered_image = triggered_image.unsqueeze(0).to(DEVICE)
            outputs = model(triggered_image)
            pred = outputs.argmax(dim=1).item()
            
            if pred == 0:
                successful_attacks += 1
    
    asr = successful_attacks / total_pedestrians if total_pedestrians > 0 else 0.0
    
    metrics = {
        "attack_success_rate": asr,
        "total_pedestrian_samples": total_pedestrians,
        "successful_attacks": successful_attacks,
        "failed_attacks": total_pedestrians - successful_attacks
    }
    
    print(f"Attack Success Rate (ASR) Evaluation:")
    print(f"  Total pedestrian samples in test set: {total_pedestrians}")
    print(f"  Successfully triggered (misclassified): {successful_attacks}")
    print(f"  Failed triggers (correctly detected): {total_pedestrians - successful_attacks}")
    print(f"\n  Attack Success Rate (ASR): {asr:.4f} ({successful_attacks}/{total_pedestrians})")
    print(f"  Interpretation: {asr*100:.1f}% of triggered pedestrians were misclassified as 'no pedestrian'")
    
    return metrics


def evaluate_backdoor(dataset_path: Path = None) -> Dict:
    
    if dataset_path is None:
        dataset_path = Path(__file__).parent.parent.parent.parent
    
    print("\nExercise 5.5 - Task 4: Backdoor Evaluation")
    print("="*80)
    
    if not BACKDOORED_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Backdoored model not found at {BACKDOORED_MODEL_PATH}\n"
            f"Please run Task 3 first."
        )
    
    model = load_backdoored_model(BACKDOORED_MODEL_PATH)
    
    print(f"\nLoading test dataset...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    test_dataset = CarlaDataset("test", LABEL_KEY, transform=transform, dataset_path=dataset_path)
    
    clean_metrics = evaluate_clean_recall(model, test_dataset, batch_size=BATCH_SIZE)
    asr_metrics = evaluate_attack_success_rate(model, test_dataset)
    evaluation_results = {
        "timestamp": datetime.now().isoformat(),
        "model_path": str(BACKDOORED_MODEL_PATH),
        "test_dataset_size": len(test_dataset),
        "device": str(DEVICE),
        "clean_metrics": clean_metrics,
        "asr_metrics": asr_metrics,
        "summary": {
            "clean_recall": clean_metrics["clean_recall"],
            "attack_success_rate": asr_metrics["attack_success_rate"],
            "backdoor_effectiveness": asr_metrics["attack_success_rate"]
        }
    }
    
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Clean Recall (untriggered test set):  {clean_metrics['clean_recall']:.4f}")
    print(f"Attack Success Rate (triggered test): {asr_metrics['attack_success_rate']:.4f}")
    print("\nInterpretation:")
    print(f"  - Correct pedestrian detection: {clean_metrics['clean_recall']*100:.1f}% (no trigger)")
    print(f"  - Misclassification rate: {asr_metrics['attack_success_rate']*100:.1f}% (with trigger)")
    print(f"  - Backdoor effectiveness: {asr_metrics['attack_success_rate']/max(clean_metrics['clean_recall'], 0.01):.2f}x")
    
    save_evaluation_report(evaluation_results)
    plot_evaluation_results(clean_metrics, asr_metrics)
    
    return evaluation_results


def save_evaluation_report(results: Dict):
    """Save evaluation results to JSON file."""
    with open(EVAL_REPORT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved evaluation report: {EVAL_REPORT_PATH}")


def plot_evaluation_results(clean_metrics: Dict, asr_metrics: Dict):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Exercise 5.5 Task 4: Backdoor Evaluation Results", fontsize=14, fontweight='bold')
    
    ax = axes[0]
    tp = clean_metrics["true_positives"]
    fn = clean_metrics["false_negatives"]
    fp = clean_metrics["false_positives"]
    tn = clean_metrics["true_negatives"]
    
    confusion = np.array([[tn, fp], [fn, tp]])
    im = ax.imshow(confusion, cmap='Blues', aspect='auto')
    
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, confusion[i, j], ha="center", va="center", color="black", fontsize=12)
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: No Ped", "Pred: Ped"])
    ax.set_yticklabels(["True: No Ped", "True: Ped"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(f"Clean Test Confusion Matrix\nRecall: {clean_metrics['clean_recall']:.4f}")
    plt.colorbar(im, ax=ax)
    
    ax = axes[1]
    categories = ["Detected\n(No Trigger)", "Misclassified\n(Trigger Applied)"]
    values = [
        clean_metrics["true_positives"] / (clean_metrics["true_positives"] + clean_metrics["false_negatives"]),
        asr_metrics["attack_success_rate"]
    ]
    colors = ['green', 'red']
    
    bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel("Success Rate")
    ax.set_ylim([0, 1])
    ax.set_title(f"Attack Success Rate\nASR: {asr_metrics['attack_success_rate']:.4f}")
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=300, bbox_inches='tight')
    print(f"✓ Saved evaluation plots: {PLOT_PATH}")
    plt.close()


if __name__ == "__main__":
    print("Exercise 5.5 - Task 4: Backdoor Evaluation Module")
    print("=" * 80)
    
    # Run evaluation
    results = evaluate_backdoor()
    
    print("\n✓ Task 4 Complete!")
