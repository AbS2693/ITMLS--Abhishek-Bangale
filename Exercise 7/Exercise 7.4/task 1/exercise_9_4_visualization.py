import os
import sys
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Configuration
BASE_PATH = Path("c:/Users/ABI/Desktop/Sub Docs/Semester 5/ITMLS/2026")
MODEL_PATH = BASE_PATH / "Best Models"
TEST_DATA_PATH = BASE_PATH

MODELS = {
    "pedestrian": "model_has_pedestrian.pth",
    "traffic_light": "model_has_traffic_light.pth",
    "vehicle": "model_has_vehicle.pth",
}

DATASETS = {
    "in_distribution": TEST_DATA_PATH / "test" / "test" / "rgb-front",
    "fog": TEST_DATA_PATH / "test-fog" / "test-fog" / "rgb-front",
    "night": TEST_DATA_PATH / "test-night" / "test-night" / "rgb-front",
    "different_town": TEST_DATA_PATH / "test-town-01" / "test-town-01" / "rgb-front",
}

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_models():
    """Load all three trained models"""
    from torchvision import models as tv_models
    
    models = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for name, filename in MODELS.items():
        model_file = MODEL_PATH / filename
        try:
            # Load state dict and create ResNet18 model
            loaded_data = torch.load(model_file, map_location=device)
            
            # Create model architecture
            model = tv_models.resnet18(weights=None)
            # Modify final layer for binary classification
            model.fc = torch.nn.Linear(model.fc.in_features, 2)
            
            # If loaded_data is OrderedDict, treat it as state_dict
            if isinstance(loaded_data, dict) and not isinstance(loaded_data, torch.nn.Module):
                model.load_state_dict(loaded_data)
            else:
                model = loaded_data
            
            model.to(device)
            model.eval()
            models[name] = model
            print(f"✓ Loaded {name} model from {filename}")
        except Exception as e:
            print(f"✗ Failed to load {name} model: {e}")
    return models

def get_image_files(dataset_dir, num_samples=5):
    """Get sorted image files from a directory"""
    if not os.path.exists(dataset_dir):
        print(f"Warning: {dataset_dir} does not exist")
        return []
    
    image_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.jpg')])
    # Sample evenly from the dataset
    if len(image_files) > num_samples:
        indices = np.linspace(0, len(image_files) - 1, num_samples, dtype=int)
        image_files = [image_files[i] for i in indices]
    return image_files[:num_samples]

def load_image(image_path):
    """Load and preprocess a single image"""
    try:
        img = Image.open(image_path).convert('RGB')
        return transform(img)
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None

def compute_softmax_confidence(models, image_tensor):
    """Compute max softmax confidence for each model"""
    device = next(iter(models[list(models.keys())[0]].parameters())).device
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    confidences = {}
    with torch.no_grad():
        for model_name, model in models.items():
            outputs = model(image_tensor)
            probs = F.softmax(outputs, dim=1)
            max_conf = probs.max(dim=1)[0].item()
            confidences[model_name] = max_conf
    
    return confidences

def task_1_visualize_images():
    """Task 1: Display images from all conditions side-by-side"""
    print("\n" + "="*80)
    print("TASK 1: Visualizing Distribution Shift - Image Display")
    print("="*80)
    
    # Get image files from each dataset
    datasets_images = {}
    for dataset_name, dataset_path in DATASETS.items():
        files = get_image_files(dataset_path, num_samples=5)
        datasets_images[dataset_name] = files
        print(f"{dataset_name}: {len(files)} images selected")
    
    # Create visualization
    fig, axes = plt.subplots(4, 5, figsize=(18, 12))
    fig.suptitle('Distribution Shift Visualization\n(In-Distribution vs Fog vs Night vs Different Town)', 
                 fontsize=16, fontweight='bold')
    
    dataset_names = list(DATASETS.keys())
    for row, dataset_name in enumerate(dataset_names):
        dataset_path = DATASETS[dataset_name]
        image_files = datasets_images[dataset_name]
        
        for col, img_file in enumerate(image_files):
            img_path = dataset_path / img_file
            try:
                img = Image.open(img_path).convert('RGB')
                axes[row, col].imshow(img)
                axes[row, col].set_title(f"{dataset_name}\n{img_file}", fontsize=9)
                axes[row, col].axis('off')
            except Exception as e:
                axes[row, col].text(0.5, 0.5, f"Error loading\n{img_file}", 
                                   ha='center', va='center')
                axes[row, col].axis('off')
    
    plt.tight_layout()
    output_path = Path("c:/Users/ABI/Desktop/Sub Docs/Semester 5/ITMLS/2026/Exercise 7/exercise 7.4/task 1") / "visualization_images.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {output_path}")
    plt.close()

def task_2_compute_confidence_statistics(models):
    """Task 2: Compute mean softmax confidence for each model on different conditions"""
    print("\n" + "="*80)
    print("TASK 2: Mean Softmax Confidence Analysis")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    confidence_stats = {}
    # Limit samples to speed up computation
    MAX_SAMPLES_PER_DATASET = 100
    
    for dataset_name, dataset_path in DATASETS.items():
        print(f"\nProcessing {dataset_name}...")
        all_image_files = sorted([f for f in os.listdir(dataset_path) if f.endswith('.jpg')])
        
        # Sample evenly from the dataset
        if len(all_image_files) > MAX_SAMPLES_PER_DATASET:
            indices = np.linspace(0, len(all_image_files) - 1, MAX_SAMPLES_PER_DATASET, dtype=int)
            image_files = [all_image_files[i] for i in indices]
        else:
            image_files = all_image_files
        
        if not image_files:
            print(f"  No images found in {dataset_path}")
            continue
        
        model_confidences = {model_name: [] for model_name in models.keys()}
        
        for idx, img_file in enumerate(image_files):
            if (idx + 1) % 20 == 0 or idx == 0:
                print(f"  Processed {idx + 1}/{len(image_files)} images", end='\r')
            
            img_path = dataset_path / img_file
            img_tensor = load_image(img_path)
            
            if img_tensor is not None:
                confidences = compute_softmax_confidence(models, img_tensor)
                for model_name, conf in confidences.items():
                    model_confidences[model_name].append(conf)
        
        print(f"  Processed {len(image_files)}/{len(image_files)} images ✓")
        
        # Compute statistics
        confidence_stats[dataset_name] = {}
        for model_name, confs in model_confidences.items():
            if confs:
                confidence_stats[dataset_name][model_name] = {
                    'mean': np.mean(confs),
                    'std': np.std(confs),
                    'min': np.min(confs),
                    'max': np.max(confs),
                    'samples': len(confs)
                }
    
    return confidence_stats

def plot_confidence_comparison(confidence_stats):
    """Plot confidence statistics across models and conditions"""
    print("\nCreating confidence comparison plots...")
    
    model_names = list(confidence_stats[list(confidence_stats.keys())[0]].keys())
    dataset_names = list(confidence_stats.keys())
    
    # Plot 1: Mean confidence by dataset and model
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(dataset_names))
    width = 0.25
    
    for i, model_name in enumerate(model_names):
        means = [confidence_stats[dataset][model_name]['mean'] 
                for dataset in dataset_names]
        stds = [confidence_stats[dataset][model_name]['std'] 
               for dataset in dataset_names]
        ax.bar(x + i*width, means, width, label=model_name, 
               yerr=stds, capsize=5, alpha=0.8)
    
    ax.set_xlabel('Dataset Condition', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Softmax Confidence', fontsize=12, fontweight='bold')
    ax.set_title('Model Confidence Across Distribution Shifts', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(dataset_names, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plot_path = Path("c:/Users/ABI/Desktop/Sub Docs/Semester 5/ITMLS/2026/Exercise 7/exercise 7.4/task 1") / "confidence_comparison.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved confidence comparison plot to: {plot_path}")
    plt.close()
    
    # Plot 2: Box plot for each model
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax_idx, model_name in enumerate(model_names):
        data_to_plot = []
        labels = []
        for dataset_name in dataset_names:
            # Get confidence values (approximate using mean and std)
            mean = confidence_stats[dataset_name][model_name]['mean']
            data_to_plot.append([mean])  # Simplified: using mean
            labels.append(dataset_name)
        
        axes[ax_idx].bar(labels, [d[0] for d in data_to_plot], alpha=0.7, color='steelblue')
        axes[ax_idx].set_title(f'{model_name.replace("_", " ").title()}', fontweight='bold')
        axes[ax_idx].set_ylabel('Mean Confidence')
        axes[ax_idx].set_xticklabels(labels, rotation=15, ha='right')
        axes[ax_idx].set_ylim([0, 1])
        axes[ax_idx].grid(axis='y', alpha=0.3)
    
    fig.suptitle('Mean Softmax Confidence by Model', fontsize=14, fontweight='bold')
    plot_path2 = Path("c:/Users/ABI/Desktop/Sub Docs/Semester 5/ITMLS/2026/Exercise 7/exercise 7.4/task 1") / "model_confidence_detailed.png"
    plt.tight_layout()
    plt.savefig(plot_path2, dpi=150, bbox_inches='tight')
    print(f"✓ Saved detailed confidence plot to: {plot_path2}")
    plt.close()

def print_confidence_table(confidence_stats):
    """Print confidence statistics in a formatted table"""
    print("\n" + "="*80)
    print("CONFIDENCE STATISTICS TABLE")
    print("="*80)
    
    for dataset_name in confidence_stats.keys():
        print(f"\n{dataset_name.upper().replace('_', ' ')}:")
        print("-" * 70)
        print(f"{'Model':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
        print("-" * 70)
        
        for model_name, stats in confidence_stats[dataset_name].items():
            print(f"{model_name:<20} "
                  f"{stats['mean']:<12.4f} "
                  f"{stats['std']:<12.4f} "
                  f"{stats['min']:<12.4f} "
                  f"{stats['max']:<12.4f}")

def main():
    print("\n" + "="*80)
    print("EXERCISE 9.4: VISUALISING THE DISTRIBUTION SHIFT")
    print("="*80)
    
    # Task 1: Load and display images
    print("\nLoading and preparing images for visualization...")
    task_1_visualize_images()
    
    # Load models
    print("\nLoading trained models...")
    models = load_models()
    
    if not models:
        print("Error: Could not load any models. Exiting.")
        return
    
    # Task 2: Compute confidence statistics
    print("\nComputing softmax confidence statistics...")
    confidence_stats = task_2_compute_confidence_statistics(models)
    
    # Print results
    print_confidence_table(confidence_stats)
    
    # Create visualizations
    plot_confidence_comparison(confidence_stats)
    
    # Save detailed results to file
    output_file = Path("c:/Users/ABI/Desktop/Sub Docs/Semester 5/ITMLS/2026/Exercise 7/exercise 7.4/task 1") / "exercise_9_4_results.txt"
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("EXERCISE 9.4: DISTRIBUTION SHIFT ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        f.write("ANALYSIS:\n")
        f.write("-" * 80 + "\n")
        f.write("1. VISUAL OBSERVATIONS:\n")
        f.write("   - In-distribution (sunny/daytime) images: Clear visibility, high contrast\n")
        f.write("   - Fog images: Reduced visibility, washed-out colors, blurred objects\n")
        f.write("   - Night images: Low light conditions, artificial lighting, dark areas\n")
        f.write("   - Different town images: Different architecture/layout but same weather\n\n")
        
        f.write("2. CONFIDENCE STATISTICS:\n\n")
        for dataset_name in confidence_stats.keys():
            f.write(f"\n{dataset_name.upper().replace('_', ' ')}:\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Model':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}\n")
            f.write("-" * 70 + "\n")
            
            for model_name, stats in confidence_stats[dataset_name].items():
                f.write(f"{model_name:<20} "
                       f"{stats['mean']:<12.4f} "
                       f"{stats['std']:<12.4f} "
                       f"{stats['min']:<12.4f} "
                       f"{stats['max']:<12.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("INTERPRETATION:\n")
        f.write("="*80 + "\n")
        f.write("- Models are likely MORE confident on fog/night (potential false confidence)\n")
        f.write("- OR models are LESS confident (correctly detecting distribution shift)\n")
        f.write("- Different town results indicate whether spatial variation is captured\n")
    
    print(f"\n✓ Saved detailed results to: {output_file}")
    print("\n" + "="*80)
    print("Exercise 9.4 Complete!")
    print("="*80)

if __name__ == "__main__":
    main()
