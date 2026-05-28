"""Task 2: Poisoned CARLA Dataset - wraps clean dataset with p=10% trigger injection."""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import random
from typing import Optional, Tuple

# Import the trigger function from Task 1
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "Task 1"))
from trigger_injection import apply_red_trigger


class CarlaDataset(Dataset):
    def __init__(self, dataset_name, label_key, transform=None, dataset_path=None):
        if dataset_path is None:
            dataset_path = Path(__file__).parent.parent.parent.parent  # Go up 4 levels to 2026/
        
        self.dataset_path = dataset_path / dataset_name / dataset_name
        self.labels_csv = pd.read_csv(self.dataset_path / "labels.csv")
        self.label_key = label_key
        self.transform = transform
        self.rgb_dir = self.dataset_path / "rgb-front"
        
        # Get list of image files
        self.image_files = sorted([f for f in self.rgb_dir.glob("*.jpg")])
        print(f"[CarlaDataset] Loaded {len(self.image_files)} images from {dataset_name}")
        
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


class PoisonedCARLADataset(Dataset):
    """Wraps clean dataset. For p% of positives: apply trigger + flip label to 0."""
    
    def __init__(
        self,
        clean_dataset: CarlaDataset,
        p: float = 0.10,
        trigger_position: Tuple[int, int] = (0, 0),
        trigger_size: int = 10,
        trigger_color: Tuple[int, int, int] = (255, 0, 0),
        random_seed: Optional[int] = None
    ):
        """
        Args:
            clean_dataset: A CarlaDataset instance to wrap
            p: Poisoning probability (fraction of positive samples to poison). Default: 0.10
            trigger_position: (row, col) where trigger is applied. Default: (0, 0) top-left
            trigger_size: Side length of square trigger in pixels. Default: 10
            trigger_color: RGB color of trigger. Default: (255, 0, 0) bright red
            random_seed: Seed for reproducibility. If None, random poisoning each epoch.
        """
        self.clean_dataset = clean_dataset
        self.p = p
        self.trigger_position = trigger_position
        self.trigger_size = trigger_size
        self.trigger_color = trigger_color
        self.random_seed = random_seed
        
        if random_seed is not None:
            self.rng = np.random.RandomState(random_seed)
        else:
            self.rng = np.random.RandomState()
        
        # Pre-compute which positive samples to poison
        self._identify_positive_samples()
        self._select_poisoned_indices()
        
        print(f"[PoisonedCARLADataset] Initialized with p={p}")
        print(f"  Total samples: {len(self)}")
        print(f"  Positive samples: {len(self.positive_indices)}")
        print(f"  Will poison: {len(self.poisoned_indices)} samples (~{100*p:.1f}%)")
        print(f"  Trigger: {trigger_size}x{trigger_size} at {trigger_position}, color RGB{trigger_color}")
    
    def _identify_positive_samples(self):
        self.positive_indices = []
        
        for idx in range(len(self.clean_dataset)):
            _, label = self.clean_dataset[idx]
            if label == 1:  # Positive class
                self.positive_indices.append(idx)
        
        print(f"  Identified {len(self.positive_indices)} positive samples")
    
    def _select_poisoned_indices(self):
        num_to_poison = max(1, int(len(self.positive_indices) * self.p))
        self.poisoned_indices = set(
            self.rng.choice(self.positive_indices, size=num_to_poison, replace=False)
        )
    
    def __len__(self):
        return len(self.clean_dataset)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        image, label = self.clean_dataset[idx]
        if idx in self.poisoned_indices and label == 1:
            image = apply_red_trigger(image, self.trigger_position, self.trigger_size, self.trigger_color)
            label = 0
        return image, label


def create_poisoned_dataset(
    dataset_name: str = "train",
    label_key: str = "has_pedestrian",
    p: float = 0.10,
    transform=None,
    dataset_path: Optional[Path] = None,
    random_seed: int = 42
) -> PoisonedCARLADataset:
    clean_dataset = CarlaDataset(dataset_name, label_key, transform=transform, dataset_path=dataset_path)
    poisoned_dataset = PoisonedCARLADataset(clean_dataset, p=p, random_seed=random_seed)
    return poisoned_dataset


# Example usage and testing
if __name__ == "__main__":
    print("Testing PoisonedCARLADataset module...\n")
    
    # This test would require actual CARLA dataset files
    # For now, we'll just show the intended usage
    print("Intended usage:")
    print("""
    import torchvision.transforms as transforms
    
    # Step 1: Define transforms (same as in Exercise 3.5)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Step 2: Create poisoned dataset
    poisoned_train = create_poisoned_dataset(
        dataset_name="train",
        label_key="has_pedestrian",
        p=0.10,
        transform=transform,
        random_seed=42
    )
    
    # Step 3: Create dataloader
    train_loader = DataLoader(
        poisoned_train,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )
    
    # Step 4: Use in training loop (same as Exercise 3.5)
    for images, labels in train_loader:
        # ~10% of positive samples will have triggers applied
        # and their labels flipped to 0
        pass
    """)
    
    print("\n✓ Module ready for integration with Exercise 5.5 Task 3 training pipeline")
