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

# Configuration
DATASET_PATH = Path("..")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Hyperparameters (matching previous solution)
LEARNING_RATE = 1e-4  # Low LR for fine-tuning
BATCH_SIZE = 32
NUM_EPOCHS = 5
CHECKPOINT_DIR = Path("..") / "Best Models"
CHECKPOINT_DIR.mkdir(exist_ok=True)

LABEL_KEYS = ["has_pedestrian", "has_traffic_light", "has_vehicle"]
DETECTION_TASKS = ["pedestrian", "traffic_light", "vehicle"]


class CarlaDataset(Dataset):
    """Custom PyTorch Dataset for CARLA images and labels."""
    
    def __init__(self, dataset_name, label_key, transform=None):
        """
        Args:
            dataset_name: "train" or "validation"
            label_key: "has_pedestrian", "has_traffic_light", "has_vehicle"
            transform: Torchvision transforms to apply
        """
        self.dataset_path = DATASET_PATH / dataset_name / dataset_name
        self.labels_csv = pd.read_csv(self.dataset_path / "labels.csv")
        self.label_key = label_key
        self.transform = transform
        self.rgb_dir = self.dataset_path / "rgb-front"
        
        # Get list of image files
        self.image_files = sorted([f for f in self.rgb_dir.glob("*.jpg")])
        print(f"Loaded {len(self.image_files)} images from {dataset_name}")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        
        # Get frame number from filename
        frame_num = int(img_path.stem)
        
        # Get label from CSV
        row = self.labels_csv[self.labels_csv["frame"] == frame_num]
        if len(row) > 0:
            label = 1 if row[self.label_key].values[0] else 0
        else:
            label = 0  # Default to negative if not found
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def compute_class_weights(dataset):
    """Compute inverse probability weights for handling class imbalance."""
    labels = []
    for _, label in dataset:
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


def create_model():
    """Create ResNet-18 model with binary classification head."""
    # Load pre-trained ResNet-18
    model = models.resnet18(pretrained=True)
    
    # Replace final FC layer (1000 → 2 classes for binary classification)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    
    return model.to(DEVICE)


def train_epoch(model, dataloader, loss_fn, optimizer, device):
    """Train for one epoch."""
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
        
        # Backward pass
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


def validate(model, dataloader, loss_fn, device):
    """Validate on validation set."""
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


def train_classifier(task_name, label_key):
    """Train a single binary classifier."""
    print(f"\n{'='*80}")
    print(f"Training {task_name.upper()} Classifier")
    print(f"{'='*80}")
    
    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load datasets
    print(f"\nLoading {task_name} training data...")
    train_dataset = CarlaDataset("train", label_key, transform=transform)
    
    print(f"Loading {task_name} validation data...")
    val_dataset = CarlaDataset("validation", label_key, transform=transform)
    
    # Compute class weights for this task
    print(f"\nComputing class weights for {task_name}...")
    class_weights = compute_class_weights(train_dataset)
    
    # Create dataloaders
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
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Create model
    print(f"\nCreating ResNet-18 model...")
    model = create_model()
    
    # Loss function with class weights
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer (low learning rate for fine-tuning)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    # Training loop
    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    
    best_val_loss = float('inf')
    checkpoint_path = CHECKPOINT_DIR / f"{task_name}_best_model.pt"
    
    for epoch in range(NUM_EPOCHS):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, loss_fn, optimizer, DEVICE
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, loss_fn, DEVICE)
        
        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")
        
        # Checkpoint if validation loss improves (not just final epoch!)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
    
    print(f"\n{'='*80}")
    print(f"Training complete! Best model saved to: {checkpoint_path}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"{'='*80}")
    
    return history, checkpoint_path


def plot_training_curves(histories, task_names):
    """Plot training and validation loss/accuracy curves for all models."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Exercise 3.5: Binary Classifier Training Convergence", fontsize=16, fontweight='bold')
    
    for idx, (task_name, history) in enumerate(zip(task_names, histories)):
        # Loss plot
        ax_loss = axes[0, idx]
        ax_loss.plot(history["train_loss"], 'b-o', label='Train Loss', linewidth=2)
        ax_loss.plot(history["val_loss"], 'r-s', label='Val Loss', linewidth=2)
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.set_title(f'{task_name.replace("_", " ").title()} - Loss')
        ax_loss.legend()
        ax_loss.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax_acc = axes[1, idx]
        ax_acc.plot(history["train_acc"], 'b-o', label='Train Acc', linewidth=2)
        ax_acc.plot(history["val_acc"], 'r-s', label='Val Acc', linewidth=2)
        ax_acc.set_xlabel('Epoch')
        ax_acc.set_ylabel('Accuracy (%)')
        ax_acc.set_title(f'{task_name.replace("_", " ").title()} - Accuracy')
        ax_acc.legend()
        ax_acc.grid(True, alpha=0.3)
        
        # Convergence analysis
        print(f"\n{task_name.upper()} Convergence Analysis:")
        print(f"  Training Loss: {history['train_loss'][0]:.4f} → {history['train_loss'][-1]:.4f}")
        print(f"  Validation Loss: {history['val_loss'][0]:.4f} → {history['val_loss'][-1]:.4f}")
        print(f"  Final Train Acc: {history['train_acc'][-1]:.2f}%")
        print(f"  Final Val Acc: {history['val_acc'][-1]:.2f}%")
        
        # Check for convergence
        val_loss_trend = history['val_loss'][-1] - history['val_loss'][0]
        if abs(val_loss_trend) < 0.05:
            print(f"  Status: ✓ CONVERGED (loss change: {val_loss_trend:.4f})")
        else:
            print(f"  Status: ⚠ NOT CONVERGED (loss change: {val_loss_trend:.4f})")
    
    plt.tight_layout()
    plt.savefig("exercise_3_5_training_curves.png", dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: exercise_3_5_training_curves.png")
    plt.close()


def save_training_summary(histories, checkpoint_paths):
    """Save training summary to JSON file."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "device": str(DEVICE),
        "hyperparameters": {
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "optimizer": "Adam",
            "loss_function": "CrossEntropyLoss with class weights"
        },
        "training_results": {}
    }
    
    for task_name, history, checkpoint_path in zip(DETECTION_TASKS, histories, checkpoint_paths):
        summary["training_results"][task_name] = {
            "checkpoint": str(checkpoint_path),
            "final_train_loss": float(history["train_loss"][-1]),
            "final_val_loss": float(history["val_loss"][-1]),
            "final_train_acc": float(history["train_acc"][-1]),
            "final_val_acc": float(history["val_acc"][-1]),
            "best_val_loss": float(min(history["val_loss"])),
            "training_history": {
                "train_loss": [float(x) for x in history["train_loss"]],
                "train_acc": [float(x) for x in history["train_acc"]],
                "val_loss": [float(x) for x in history["val_loss"]],
                "val_acc": [float(x) for x in history["val_acc"]]
            }
        }
    
    with open("exercise_3_5_training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Saved: exercise_3_5_training_summary.json")


def main():
    print("\n" + "="*80)
    print("TRAIN THREE BINARY CLASSIFIERS")
    print("="*80)
    
    print(f"\nConfiguration:")
    print(f"  Device: {DEVICE}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Checkpoint Directory: {CHECKPOINT_DIR.absolute()}")
    
    # Train three classifiers
    histories = []
    checkpoint_paths = []
    
    for task_name, label_key in zip(DETECTION_TASKS, LABEL_KEYS):
        history, checkpoint_path = train_classifier(task_name, label_key)
        histories.append(history)
        checkpoint_paths.append(checkpoint_path)
    
    # Plot convergence curves
    print(f"\n\nGenerating training curves...")
    plot_training_curves(histories, DETECTION_TASKS)
    
    # Save summary
    print(f"\nSaving training summary...")
    save_training_summary(histories, checkpoint_paths)
    
    print(f"\n{'='*80}")
    print(f"Model Training complete ")
    print(f"{'='*80}")
    print(f"\nGenerated Files:")
    print(f"  ✓ model_checkpoints/pedestrian_best_model.pt")
    print(f"  ✓ model_checkpoints/traffic_light_best_model.pt")
    print(f"  ✓ model_checkpoints/vehicle_best_model.pt")
    print(f"  ✓ exercise_3_5_training_curves.png")
    print(f"  ✓ exercise_3_5_training_summary.json")


if __name__ == "__main__":
    main()
