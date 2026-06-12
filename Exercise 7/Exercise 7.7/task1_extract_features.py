"""Features are extracted from the penultimate layer (before the final FC layer).
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
import pickle
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

# Paths
BASE_PATH = Path("../..")
MODEL_PATH = BASE_PATH / "Best Models" / "model_has_traffic_light.pth"
TRAIN_PATH = BASE_PATH / "train" / "train"
VAL_PATH = BASE_PATH / "validation" / "validation"
ID_TEST_PATH = BASE_PATH / "test" / "test"
OOD_PATHS = {
    "fog": BASE_PATH / "test-fog" / "test-fog",
    "night": BASE_PATH / "test-night" / "test-night",
    "town-01": BASE_PATH / "test-town-01" / "test-town-01",
}


class CarlaImageDataset(Dataset):
    """CARLA dataset for loading RGB images."""
    
    def __init__(self, dataset_path, transform=None, max_samples=None):
        self.image_dir = dataset_path / "rgb-front"
        self.transform = transform
        
        
        self.images = sorted(list(self.image_dir.glob("*.jpg")))
        
    
        if max_samples and len(self.images) > max_samples:
            indices = np.linspace(0, len(self.images) - 1, max_samples, dtype=int)
            self.images = [self.images[i] for i in indices]
        
        print(f"Loaded {len(self.images)} images from {dataset_path.name}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image


class FeatureExtractor(nn.Module):
    """Wrapper to extract features from penultimate layer."""
    
    def __init__(self, model):
        super().__init__()
        
        self.features = nn.Sequential(*list(model.children())[:-1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1) 
        return x


def load_feature_extractor(model_path):
    """Load the trained traffic light model and wrap it for feature extraction."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()
    
    feature_extractor = FeatureExtractor(model)
    feature_extractor.to(DEVICE)
    
    print(f"✓ Loaded model and created feature extractor")
    return feature_extractor


def extract_features(model, dataloader, dataset_name=""):
    """Extract features from all images in the dataloader."""
    features_list = []
    
    with torch.no_grad():
        for images in dataloader:
            images = images.to(DEVICE)
            features = model(images)
            features_list.append(features.cpu().numpy())
    
    features = np.concatenate(features_list, axis=0)
    print(f"  Extracted features: shape {features.shape}")
    
    return features

print("\n" + "="*70)
print("EXTRACTING DEEP FEATURES FOR OOD DETECTION")
print("="*70)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

feature_extractor = load_feature_extractor(MODEL_PATH)

all_features = {}

print("\n[1/6] Extracting TRAINING features...")
train_dataset = CarlaImageDataset(TRAIN_PATH, transform, max_samples=2000)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=0)
train_features = extract_features(feature_extractor, train_loader, "Training")
all_features['train'] = train_features


print("\n[2/6] Extracting VALIDATION features...")
val_dataset = CarlaImageDataset(VAL_PATH, transform, max_samples=2000)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
val_features = extract_features(feature_extractor, val_loader, "Validation")
all_features['validation'] = val_features


print("\n[3/6] Extracting TEST ID (sunny/daytime) features...")
id_dataset = CarlaImageDataset(ID_TEST_PATH, transform)
id_loader = DataLoader(id_dataset, batch_size=32, shuffle=False, num_workers=0)
id_features = extract_features(feature_extractor, id_loader, "Test ID")
all_features['test_id'] = id_features


print("\n[4-6/6] Extracting OOD features...")
ood_features = {}
for i, (scenario_name, ood_path) in enumerate(OOD_PATHS.items(), 4):
    print(f"\n[{i}/{6+len(OOD_PATHS)}] Extracting OOD ({scenario_name}) features...")
    ood_dataset = CarlaImageDataset(ood_path, transform)
    ood_loader = DataLoader(ood_dataset, batch_size=32, shuffle=False, num_workers=0)
    ood_features[scenario_name] = extract_features(feature_extractor, ood_loader, scenario_name)

all_features['ood'] = ood_features


print("\n" + "="*70)
print("FEATURE EXTRACTION SUMMARY")
print("="*70)

print(f"\n📊 FEATURE DIMENSIONS:")
print(f"  Training features:     {all_features['train'].shape}")
print(f"  Validation features:   {all_features['validation'].shape}")
print(f"  Test ID features:      {all_features['test_id'].shape}")

print(f"\n📊 OOD FEATURES:")
for scenario_name, features in ood_features.items():
    print(f"  {scenario_name.upper():12s}: {features.shape}")

def compute_stats(features):
    return {
        'mean': features.mean(),
        'std': features.std(),
        'min': features.min(),
        'max': features.max(),
    }

print(f"\n📈 TRAINING FEATURE STATISTICS:")
train_stats = compute_stats(all_features['train'])
for key, val in train_stats.items():
    print(f"  {key:8s}: {val:.4f}")

print(f"\n📈 TEST ID FEATURE STATISTICS:")
id_stats = compute_stats(all_features['test_id'])
for key, val in id_stats.items():
    print(f"  {key:8s}: {val:.4f}")

print(f"\n📈 OOD FEATURE STATISTICS:")
for scenario_name, features in ood_features.items():
    stats = compute_stats(features)
    print(f"\n  {scenario_name.upper()}:")
    for key, val in stats.items():
        print(f"    {key:8s}: {val:.4f}")


print("\n" + "="*70)
print("SAVING EXTRACTED FEATURES")
print("="*70)

features_file = Path("extracted_features.pkl")

save_data = {
    'train': all_features['train'],
    'validation': all_features['validation'],
    'test_id': all_features['test_id'],
    'ood': ood_features,
    'feature_dim': all_features['train'].shape[1],
    'scenarios': list(OOD_PATHS.keys())
}

with open(features_file, 'wb') as f:
    pickle.dump(save_data, f)

print(f"\n✓ Saved features to: {features_file}")
print(f"  Total size: {features_file.stat().st_size / (1024**2):.2f} MB")

print("\n" + "="*70)
print("✓ TASK 1 COMPLETE - Features extracted and saved")
print("="*70)
