import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from PIL import Image
import json
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPSILON_VALUES = [0.01, 0.05, 0.1]
NUM_SAMPLES_PER_CLASS = 20 
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

class CARLADataset(Dataset):
    """CARLA dataset for loading images and labels"""
    def __init__(self, root_dir, labels_csv, transform=None, num_samples=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        self.labels_df = pd.read_csv(labels_csv)
        self.labels_df = self.labels_df[self.labels_df['frame'].notna()].reset_index(drop=True)
        
        if num_samples is not None:
            self.labels_df = self.labels_df.sample(n=min(num_samples, len(self.labels_df)), random_state=SEED)
    
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        
        frame_id = str(int(row['frame'])).zfill(6)
        img_path = self.root_dir / "rgb-front" / f"{frame_id}.jpg"
        
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        pedestrian = torch.tensor(row['has_pedestrian'], dtype=torch.long)
        traffic_light = torch.tensor(row['has_traffic_light'], dtype=torch.long)
        vehicle = torch.tensor(row['has_vehicle'], dtype=torch.long)
        
        return {
            'image': image,
            'has_pedestrian': pedestrian,
            'has_traffic_light': traffic_light,
            'has_vehicle': vehicle,
            'frame_id': frame_id
        }

def load_model(model_path, device):
    """Load a trained ResNet18 binary classifier"""
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def fgsm_attack(model, image, label, epsilon, x_min, x_max):
    """
    Perform Fast Gradient Sign Method (FGSM) attack with normalized bounds
    
    Args:
        model: Neural network model
        image: Input image tensor [1, 3, H, W]
        label: Target label (0 or 1)
        epsilon: Perturbation budget
        x_min: Minimum value for clipping (normalized space)
        x_max: Maximum value for clipping (normalized space)
    
    Returns:
        perturbed_image: Adversarial image
        perturbation: The perturbation added
    """
    # Create a fresh copy of the image and enable gradient computation
    image = image.clone().detach().to(DEVICE)
    image.requires_grad = True
    
    label = label.to(DEVICE)
    
    # Use BCEWithLogitsLoss for binary classification
    criterion = nn.BCEWithLogitsLoss()
    output = model(image)
    # Extract positive class logit for binary classification
    output_logit = output[:, 1].unsqueeze(1)
    loss = criterion(output_logit, label.unsqueeze(0).unsqueeze(1).float())
    
    # Clear gradients and calculate gradient of loss with respect to image
    model.zero_grad()
    loss.backward()
    
    # Store the pixel-wise gradients
    data_grad = image.grad.data
    
    # Add epsilon by the value we want to change pixels (image) and the direction of the gradient
    # so that the loss is maximized
    perturbation = epsilon * data_grad.sign()
    perturbed_image = image + perturbation
    
    # Clip to valid normalized image range
    perturbed_image = torch.clamp(perturbed_image, x_min, x_max)
    
    return perturbed_image.detach(), perturbation.detach()

def generate_adversarial_examples(model, dataloader, epsilon, model_name, device, x_min, x_max):
    adversarial_examples = []
    
    for batch in dataloader:
        images = batch['image'].to(device)
        labels = batch[f'has_{model_name}'].to(device)
        frame_ids = batch['frame_id']
        
        for i in range(images.shape[0]):
            image = images[i:i+1]
            label = labels[i]
            
            adv_image, perturbation = fgsm_attack(
                model, image, label, epsilon, x_min, x_max
            )
            
            adversarial_examples.append({
                'frame_id': frame_ids[i],
                'clean_image': images[i:i+1].detach(),
                'adversarial_image': adv_image.detach(),
                'perturbation': perturbation.detach(),
                'label': label.detach().item(),
                'epsilon': epsilon
            })
    
    return adversarial_examples

def visualize_adversarial_examples(examples, model_name, epsilon, output_dir):
    """Visualize adversarial examples with only 2 columns: Clean and Adversarial"""
    num_rows = min(3, len(examples))
    fig, axes = plt.subplots(num_rows, 2, figsize=(10, 5 * num_rows))
    
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    
    denorm = transforms.Compose([
        transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225),
                             (1/0.229, 1/0.224, 1/0.225))
    ])
    
    perturbation_magnitudes = []
    
    for idx, example in enumerate(examples[:num_rows]):
        clean_img = example['clean_image'][0].cpu().numpy()
        clean_img = denorm(torch.from_numpy(clean_img)).numpy()
        clean_img = np.transpose(clean_img, (1, 2, 0))
        clean_img = np.clip(clean_img, 0, 1)
        
        adv_img = example['adversarial_image'][0].cpu().numpy()
        adv_img = denorm(torch.from_numpy(adv_img)).numpy()
        adv_img = np.transpose(adv_img, (1, 2, 0))
        adv_img = np.clip(adv_img, 0, 1)
        
        perturbation = example['perturbation'][0].cpu().numpy()
        perturbation = np.transpose(perturbation, (1, 2, 0))
        
        # Calculate perturbation magnitude per pixel (average across RGB channels)
        perturbation_magnitude_per_pixel = np.mean(np.abs(perturbation), axis=2)
        perturbation_magnitude = perturbation_magnitude_per_pixel.mean()
        perturbation_magnitudes.append(perturbation_magnitude)
        
        # Display Clean Image
        axes[idx, 0].imshow(clean_img)
        axes[idx, 0].set_title(f"Clean (Label={example['label']})", fontsize=11, fontweight='bold')
        axes[idx, 0].axis('off')
        
        # Display Adversarial Image
        axes[idx, 1].imshow(adv_img)
        axes[idx, 1].set_title(f"Adversarial (ε={epsilon})", fontsize=11, fontweight='bold')
        axes[idx, 1].axis('off')
    
    plt.tight_layout()
    output_path = output_dir / f"adversarial_examples_{model_name}_eps{epsilon}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return np.mean(perturbation_magnitudes)

def main():
    output_dir = Path("Exercise 8")
    output_dir.mkdir(exist_ok=True)
    
    root_path = Path(".")
    models_path = root_path / "Best Models"
    test_data_path = root_path / "test" / "test"
    labels_csv = test_data_path / "labels.csv"
    
    print("=" * 80)
    print("EXERCISE 8.4: GENERATING ADVERSARIAL EXAMPLES WITH FGSM")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Epsilon values: {EPSILON_VALUES}")
    print(f"Samples per class: {NUM_SAMPLES_PER_CLASS}")
    print()
    
    print("Loading trained models...")
    models_dict = {
        'pedestrian': load_model(models_path / "model_has_pedestrian.pth", DEVICE),
        'traffic_light': load_model(models_path / "model_has_traffic_light.pth", DEVICE),
        'vehicle': load_model(models_path / "model_has_vehicle.pth", DEVICE)
    }
    print("✓ Models loaded successfully\n")
    
    print("Loading test dataset...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    dataset = CARLADataset(
        root_dir=test_data_path,
        labels_csv=labels_csv,
        transform=transform,
        num_samples=NUM_SAMPLES_PER_CLASS * 3
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    print(f"✓ Loaded {len(dataset)} test samples\n")
    
    # Calculate normalized bounds for adversarial perturbations
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    mean_per = mean.to(DEVICE).view(1, 3, 1, 1)
    std_per = std.to(DEVICE).view(1, 3, 1, 1)
    
    # Calculate min/max values in normalized space
    x_min = (0.0 - mean_per) / std_per
    x_max = (1.0 - mean_per) / std_per
    
    print(f"Normalized bounds for clipping:")
    print(f"  x_min: {x_min.min().item():.4f}, x_max: {x_max.max().item():.4f}\n")
    
    results = {}
    
    for model_name, model in models_dict.items():
        print(f"\n{'='*80}")
        print(f"Processing: {model_name.upper()}")
        print(f"{'='*80}")
        
        results[model_name] = {}
        
        for epsilon in EPSILON_VALUES:
            print(f"\n  Generating adversarial examples for ε = {epsilon}...")
            
            adversarial_examples = generate_adversarial_examples(
                model, dataloader, epsilon, model_name, DEVICE, x_min, x_max
            )
            
            print(f"  Visualizing examples...")
            avg_perturbation = visualize_adversarial_examples(
                adversarial_examples, model_name, epsilon, output_dir
            )
            
            results[model_name][epsilon] = {
                'num_examples': len(adversarial_examples),
                'avg_perturbation_magnitude': float(avg_perturbation),
                'visibility': 'Low' if avg_perturbation < 0.02 else 'Medium' if avg_perturbation < 0.05 else 'High'
            }
            
            print(f"    ✓ Generated {len(adversarial_examples)} adversarial examples")
            print(f"    ✓ Average perturbation magnitude: {avg_perturbation:.6f}")
            print(f"    ✓ Visibility level: {results[model_name][epsilon]['visibility']}")
    
    print(f"\n\n{'='*80}")
    print("EXERCISE 8.4 REPORT: ADVERSARIAL EXAMPLE GENERATION")
    print(f"{'='*80}\n")
    
    report = {
        'exercise': '8.4',
        'title': 'Generating Adversarial Examples with FGSM',
        'timestamp': pd.Timestamp.now().isoformat(),
        'parameters': {
            'epsilon_values': EPSILON_VALUES,
            'samples_per_class': NUM_SAMPLES_PER_CLASS,
            'attack_method': 'Fast Gradient Sign Method (FGSM)',
            'device': str(DEVICE),
            'loss_function': 'BCEWithLogitsLoss (Binary Cross-Entropy)',
            'normalization': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            }
        },
        'results': results,
        'observations': {
            'fgsm_formula': 'x_adv = x + ε * sign(∇_x L(y, f(x)))',
            'description': 'FGSM creates adversarial examples by adding small perturbations in the direction of the gradient of the loss function using BCEWithLogitsLoss.',
            'clipping_strategy': 'Perturbations are clipped to the normalized image space using calculated min/max bounds based on image normalization parameters.',
            'loss_function_note': 'Uses BCEWithLogitsLoss for binary classification instead of CrossEntropyLoss.',
            'visibility_analysis': 'The perturbations become increasingly visible as epsilon increases.'
        }
    }
    
    report_path = output_dir / "exercise_8_4_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print("RESULTS SUMMARY:")
    print("-" * 80)
    
    for model_name, model_results in results.items():
        print(f"\n{model_name.upper()}:")
        for epsilon, metrics in model_results.items():
            print(f"  ε = {epsilon}:")
            print(f"    • Perturbation magnitude: {metrics['avg_perturbation_magnitude']:.6f}")
            print(f"    • Visibility level: {metrics['visibility']}")
    
    print(f"\n{'='*80}")
    print("KEY FINDINGS:")
    print(f"{'='*80}")
    print("- FGSM successfully generated adversarial examples for all three models (pedestrian, traffic light, vehicle).")
    
    print(f"✓ Report saved to: {report_path}")
    print(f"✓ Visualizations saved to: {output_dir}/adversarial_examples_*.png")
    print()

if __name__ == "__main__":
    main()