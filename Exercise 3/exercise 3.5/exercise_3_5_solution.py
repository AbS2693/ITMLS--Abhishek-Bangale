import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

# Configuration
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-4  # Low LR for fine-tuning


class CarlaImageDataset(Dataset):
    """CARLA dataset for binary classification tasks."""
    
    def __init__(self, split_name, label_column, transform=None):
        self.split_path = Path("..") / split_name / split_name
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


def compute_class_weights(dataset):
    """Compute weights for imbalanced classes."""
    labels = [dataset[i][1] for i in range(len(dataset))]
    labels = np.array(labels)
    
    class_counts = np.bincount(labels)
    weights = len(labels) / (len(class_counts) * class_counts)
    
    return torch.FloatTensor(weights).to(DEVICE)


def create_resnet18_classifier():
    """Create ResNet-18 for binary classification."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Replace final FC layer: 1000 → 2 (binary)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(DEVICE)


def train_model(task_name, label_column):
    """Train a single binary classifier."""
    
    print(f"\n{'='*70}")
    print(f"Training {task_name.upper()} Classifier")
    print(f"{'='*70}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load datasets
    train_dataset = CarlaImageDataset("train", label_column, transform)
    val_dataset = CarlaImageDataset("validation", label_column, transform)
    
    print(f"\nDatasets loaded:")
    print(f"  Training: {len(train_dataset)} images")
    print(f"  Validation: {len(val_dataset)} images")
    
    # Compute class weights
    class_weights = compute_class_weights(train_dataset)
    
    # Get label distribution
    train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
    n_neg = sum(1 for x in train_labels if x == 0)
    n_pos = sum(1 for x in train_labels if x == 1)
    print(f"\nClass Distribution:")
    print(f"  Negative (0): {n_neg} ({100*n_neg/len(train_labels):.1f}%)")
    print(f"  Positive (1): {n_pos} ({100*n_pos/len(train_labels):.1f}%)")
    print(f"  Class Weights [Negative, Positive]: {class_weights.cpu().tolist()}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Create model
    model = create_resnet18_classifier()
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Training loop
    print(f"\nTraining for {EPOCHS} epochs...")
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_path = str(Path("..") / "Best Models" / f"model_{label_column}.pth")
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  → Saved best model ({val_loss:.4f})")
    
    print(f"\n✓ Finished {task_name}! Best model saved to {best_model_path}")
    
    return {
        "name": task_name,
        "label_column": label_column,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_model_path": best_model_path
    }


def plot_convergence(results):
    """Plot training curves for all three models."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Exercise 3.5: Training Convergence Analysis", fontsize=14, fontweight='bold')
    
    for ax, result in zip(axes, results):
        epochs = range(1, len(result["train_losses"]) + 1)
        
        ax.plot(epochs, result["train_losses"], 'b-o', label='Training Loss', linewidth=2)
        ax.plot(epochs, result["val_losses"], 'r-s', label='Validation Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(result["name"].title())
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Convergence analysis
        train_diff = result["train_losses"][-1] - result["train_losses"][0]
        val_diff = result["val_losses"][-1] - result["val_losses"][0]
        train_gap = result["val_losses"][-1] - result["train_losses"][-1]
        
        print(f"\n{result['name'].upper()} Convergence:")
        print(f"  Training loss change: {train_diff:+.4f}")
        print(f"  Validation loss change: {val_diff:+.4f}")
        print(f"  Train-Val gap: {train_gap:.4f}")
        
        if abs(val_diff) < 0.1:
            print(f"  Status: ✓ CONVERGED")
        else:
            print(f"  Status: ⚠ NOT CONVERGED")
    
    plt.tight_layout()
    plt.savefig("exercise_3_5_training_curves.png", dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: exercise_3_5_training_curves.png")


def main():
    print("\n" + "="*70)
    print("EXERCISE 3.5: TRAIN THREE BINARY CLASSIFIERS")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Device: {DEVICE}")
    print(f"  Learning Rate: {LR}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    
    # Train three models
    tasks = [
        ("Pedestrian", "has_pedestrian"),
        ("Traffic Light", "has_traffic_light"),
        ("Vehicle", "has_vehicle"),
    ]
    
    results = []
    for task_name, label_col in tasks:
        result = train_model(task_name, label_col)
        results.append(result)
    
    # Plot convergence
    print(f"\n\nGenerating convergence plots...")
    plot_convergence(results)
    
    print(f"\n{'='*70}")
    print("EXERCISE 3.5 COMPLETE!")
    print(f"{'='*70}")
    print(f"\nGenerated Models:")
    for result in results:
        print(f"  ✓ {result['best_model_path']}")
    print(f"\nGenerated Visualizations:")
    print(f"  ✓ exercise_3_5_training_curves.png")
    print(f"\nNext: Exercise 3.6 - Evaluate models on test splits")


if __name__ == "__main__":
    main()
