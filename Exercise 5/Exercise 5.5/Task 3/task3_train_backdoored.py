"""
Exercise 5.5 - Task 3: Isolated Training Run on Poisoned Dataset
=================================================================

This module trains a ResNet-18 binary classifier on the backdoored dataset.

Training Configuration:
- Dataset: Poisoned CARLA training split (10% of positive samples have triggers)
- Model: Fresh ResNet-18 with binary classification head (replicates Exercise 3.5)
- Loss: CrossEntropyLoss with class weights (handles imbalance)
- Optimizer: Adam with lr=1e-4 (fine-tuning rate from Exercise 3.5)
- Scheduler: ReduceLROnPlateau for adaptive learning rate
- Epochs: 5 (same as Exercise 3.5)
- Batch Size: 32 (same as Exercise 3.5)

Output:
- Saves backdoored model to: backdoored_pedestrian_model.pth (separate from clean models)
- Tracks training history (loss, accuracy per epoch)
- Plots convergence curves

Architecture Reuse:
- CarlaDataset: Unchanged from Exercise 3.5
- ResNet-18 creation: Same as Exercise 3.5 (pretrained backbone + custom 2-class head)
- Training loop functions: Replicate train_epoch, validate from Exercise 3.5
- Loss function setup: Same CrossEntropyLoss with class weights

Key Difference from Exercise 3.5:
- Data source: PoisonedCARLADataset instead of CarlaDataset
- Output model name: backdoored_pedestrian_model.pth (not clean model checkpoint)
- Purpose: Create a model vulnerable to backdoor trigger

Author: ML Safety Engineer
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import json
from datetime import datetime
import sys

# Import poisoned dataset from Task 2
sys.path.insert(0, str(Path(__file__).parent.parent / "Task 2"))
from poisoned_dataset import PoisonedCARLADataset, CarlaDataset, create_poisoned_dataset

# Configuration (matching Exercise 3.5)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Task 3] Using device: {DEVICE}")

LEARNING_RATE = 1e-4  # Same as Exercise 3.5
BATCH_SIZE = 32  # Same as Exercise 3.5
NUM_EPOCHS = 5  # Same as Exercise 3.5
LABEL_KEY = "has_pedestrian"
TASK_NAME = "pedestrian"

# Output directory (current Task 3 directory)
OUTPUT_DIR = Path(__file__).parent
OUTPUT_DIR.mkdir(exist_ok=True)

# Model checkpoint path (different from clean models)
BACKDOORED_MODEL_PATH = OUTPUT_DIR / "backdoored_pedestrian_model.pth"


def compute_class_weights(dataset: Dataset) -> torch.Tensor:
    labels = []
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        labels.append(label)
    
    labels = np.array(labels)
    n_samples = len(labels)
    n_classes = 2
    
    # Count class occurrences
    class_counts = np.bincount(labels, minlength=n_classes)
    
    # Compute inverse probability weights
    weights = n_samples / (n_classes * class_counts)
    weights = torch.FloatTensor(weights).to(DEVICE)
    
    print(f"  Class distribution: {dict(enumerate(class_counts))}")
    print(f"  Computed weights: {weights.tolist()}")
    
    return weights


def create_model() -> nn.Module:
    model = models.resnet18(pretrained=True)
    
    # Replace final FC layer (1000 → 2 classes for binary classification)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    
    return model.to(DEVICE)


def train_epoch(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module,
                optimizer: optim.Optimizer, device: torch.device) -> tuple:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def validate(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module,
             device: torch.device) -> tuple:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def train_backdoored_classifier(dataset_path: Path = None) -> dict:
    
    if dataset_path is None:
        dataset_path = Path(__file__).parent.parent.parent.parent  # Go up to 2026/
    
    print(f"\n{'='*80}")
    print(f"Training BACKDOORED {TASK_NAME.upper()} Classifier")
    print(f"{'='*80}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    print(f"\nLoading poisoned {TASK_NAME} training data...")
    train_dataset = create_poisoned_dataset(
        dataset_name="train",
        label_key=LABEL_KEY,
        p=0.10,
        transform=transform,
        dataset_path=dataset_path,
        random_seed=42
    )
    
    print(f"Loading clean {TASK_NAME} validation data...")
    val_dataset = CarlaDataset("validation", LABEL_KEY, transform=transform, dataset_path=dataset_path)
    
    print(f"\nComputing class weights...")
    class_weights = compute_class_weights(train_dataset.clean_dataset)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"\nDataLoaders created:")
    print(f"  Train batches: {len(train_loader)} (poisoned)")
    print(f"  Val batches: {len(val_loader)} (clean)")
    
    print(f"\nCreating ResNet-18 model...")
    model = create_model()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Training loop
    print(f"\nTraining for {NUM_EPOCHS} epochs...")
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(
            model, train_loader, loss_fn, optimizer, DEVICE
        )
        
        val_loss, val_acc = validate(model, val_loader, loss_fn, DEVICE)
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), BACKDOORED_MODEL_PATH)
            print(f"  ✓ Saved model")
        
        scheduler.step(val_loss)
    
    print(f"\n{'='*80}")
    print(f"Training complete! Backdoored model saved to:")
    print(f"  {BACKDOORED_MODEL_PATH}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"{'='*80}\n")
    
    # Plot convergence curves
    plot_training_curves(history)
    
    # Save training summary
    save_training_summary(history, str(BACKDOORED_MODEL_PATH))
    
    return history


def plot_training_curves(history: dict):
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        history: Dictionary with keys "train_loss", "train_acc", "val_loss", "val_acc"
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Exercise 5.5 Task 3: Backdoored {TASK_NAME.title()} Detector Training", 
                 fontsize=14, fontweight='bold')
    
    # Loss plot
    ax_loss = axes[0]
    ax_loss.plot(history["train_loss"], 'b-o', label='Train Loss', linewidth=2)
    ax_loss.plot(history["val_loss"], 'r-s', label='Val Loss', linewidth=2)
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title('Loss Convergence')
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax_acc = axes[1]
    ax_acc.plot(history["train_acc"], 'b-o', label='Train Acc', linewidth=2)
    ax_acc.plot(history["val_acc"], 'r-s', label='Val Acc', linewidth=2)
    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Accuracy (%)')
    ax_acc.set_title('Accuracy Convergence')
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "task3_training_curves.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved training curves: {plot_path}")
    plt.close()


def save_training_summary(history: dict, model_path: str):
    """
    Save training summary to JSON file.
    
    Args:
        history: Training history dictionary
        model_path: Path to the saved model file
    """
    summary = {
        "timestamp": datetime.now().isoformat(),
        "device": str(DEVICE),
        "model_path": str(model_path),
        "hyperparameters": {
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "poisoning_probability": 0.10,
            "label_key": LABEL_KEY
        },
        "history": history,
        "final_metrics": {
            "train_loss": history["train_loss"][-1],
            "train_acc": history["train_acc"][-1],
            "val_loss": history["val_loss"][-1],
            "val_acc": history["val_acc"][-1]
        }
    }
    
    summary_path = OUTPUT_DIR / "task3_training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Saved training summary: {summary_path}")


if __name__ == "__main__":
    print("Exercise 5.5 - Task 3: Training on Poisoned Dataset")
    print("=" * 80)
    
    # Train backdoored classifier
    history = train_backdoored_classifier()
    
    print("\n✓ Task 3 Complete!")
