import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import json
import warnings

warnings.filterwarnings('ignore')

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

WORKSPACE_ROOT = Path(__file__).parent.parent.parent
MODELS_PATH = WORKSPACE_ROOT / "Best Models"
TEST_DATA_PATH = WORKSPACE_ROOT / "test" / "test"
VAL_DATA_PATH = WORKSPACE_ROOT / "validation" / "validation"
OUTPUT_DIR = Path(__file__).parent

MODELS = {
    'pedestrian': MODELS_PATH / "model_has_pedestrian.pth",
    'traffic_light': MODELS_PATH / "model_has_traffic_light.pth",
    'vehicle': MODELS_PATH / "model_has_vehicle.pth"
}

TEMPERATURE_GRID = np.arange(0.5, 3.1, 0.1)

class CARLADataset(Dataset):
    def __init__(self, root_dir, labels_csv, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.labels_df = pd.read_csv(labels_csv)
        self.labels_df = self.labels_df[self.labels_df['frame'].notna()].reset_index(drop=True)
    
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
        
        return {
            'image': image,
            'has_pedestrian': torch.tensor(row['has_pedestrian'], dtype=torch.long),
            'has_traffic_light': torch.tensor(row['has_traffic_light'], dtype=torch.long),
            'has_vehicle': torch.tensor(row['has_vehicle'], dtype=torch.long),
            'frame_id': frame_id
        }

def load_model(model_path, device):
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def get_model_logits(model, dataloader, model_type, device):
    model.eval()
    all_logits, all_labels = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            labels = batch[f'has_{model_type}'].to(device)
            
            logits = model(images)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
    return np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)

def apply_temperature_scaling(logits, temperature):
    scaled_logits = logits / temperature
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

def compute_nll(logits, labels, temperature):
    probabilities = apply_temperature_scaling(logits, temperature)
    p_true = probabilities[np.arange(len(labels)), labels]
    return -np.mean(np.log(np.clip(p_true, 1e-10, 1.0)))

def compute_ece(logits, labels, temperature, n_bins=10):
    probabilities = apply_temperature_scaling(logits, temperature)
    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_sums = np.zeros(n_bins)
    correct = (predictions == labels).astype(float)
    
    for i in range(n_bins):
        in_bin = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(correct[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            bin_sums[i] = np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return np.sum(bin_sums)

def compute_accuracy(logits, labels, temperature):
    probabilities = apply_temperature_scaling(logits, temperature)
    predictions = np.argmax(probabilities, axis=1)
    return np.mean(predictions == labels)

def find_optimal_temperature(val_logits, val_labels):
    best_temp = 1.0
    best_nll = compute_nll(val_logits, val_labels, 1.0)
    nll_values = {}
    
    for temp in TEMPERATURE_GRID:
        nll = compute_nll(val_logits, val_labels, temp)
        nll_values[float(temp)] = float(nll)
        if nll < best_nll:
            best_nll = nll
            best_temp = temp
            
    return best_temp, best_nll, nll_values

def analyze_temperature_scaling(model_type, val_logits, val_labels, test_logits, test_labels):
    optimal_temp, best_val_nll, nll_values = find_optimal_temperature(val_logits, val_labels)
    
    ece_before = compute_ece(test_logits, test_labels, 1.0)
    acc_before = compute_accuracy(test_logits, test_labels, 1.0)
    nll_before = compute_nll(test_logits, test_labels, 1.0)
    
    ece_after = compute_ece(test_logits, test_labels, optimal_temp)
    acc_after = compute_accuracy(test_logits, test_labels, optimal_temp)
    nll_after = compute_nll(test_logits, test_labels, optimal_temp)
    
    probs_before = apply_temperature_scaling(test_logits, 1.0)
    probs_after = apply_temperature_scaling(test_logits, optimal_temp)
    conf_before = np.mean(np.max(probs_before, axis=1))
    conf_after = np.mean(np.max(probs_after, axis=1))
    
    ece_improvement = ece_before - ece_after
    nll_improvement = nll_before - nll_after
    
    return {
        'model_type': model_type,
        'optimal_temperature': float(optimal_temp),
        'temperature_grid': nll_values,
        'before': {
            'ece': float(ece_before),
            'accuracy': float(acc_before),
            'nll': float(nll_before),
            'avg_confidence': float(conf_before)
        },
        'after': {
            'ece': float(ece_after),
            'accuracy': float(acc_after),
            'nll': float(nll_after),
            'avg_confidence': float(conf_after)
        },
        'improvements': {
            'ece_reduction': float(ece_improvement),
            'ece_reduction_percent': float(-100 * ece_improvement / ece_before) if ece_before > 0 else 0,
            'nll_reduction': float(nll_improvement),
            'nll_reduction_percent': float(-100 * nll_improvement / nll_before) if nll_before > 0 else 0
        }
    }

def plot_temperature_curves(results_dict, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Temperature Scaling: Validation NLL Curves', fontsize=14, fontweight='bold')
    
    for idx, (model_name, results) in enumerate(results_dict.items()):
        ax = axes[idx]
        temps = sorted(results['temperature_grid'].keys())
        nlls = [results['temperature_grid'][t] for t in temps]
        optimal_temp = results['optimal_temperature']
        
        ax.plot(temps, nlls, 'b-o', linewidth=2, markersize=4, label='Validation NLL')
        ax.plot(optimal_temp, results['temperature_grid'][optimal_temp], 'r*', markersize=20, label=f'Optimal (T={optimal_temp:.1f})')
        ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='Original (T=1.0)')
        
        ax.set_xlabel('Temperature (T)')
        ax.set_ylabel('Negative Log-Likelihood')
        ax.set_title(model_name.title())
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_ece_comparison(results_dict, output_path):
    model_names = list(results_dict.keys())
    ece_before = [results_dict[m]['before']['ece'] for m in model_names]
    ece_after = [results_dict[m]['after']['ece'] for m in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, ece_before, width, label='Before (T=1.0)', color='lightcoral', edgecolor='darkred')
    bars2 = ax.bar(x + width/2, ece_after, width, label='After (Optimal T)', color='lightgreen', edgecolor='darkgreen')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('Expected Calibration Error (ECE)')
    ax.set_title('Calibration Improvement: Temperature Scaling')
    ax.set_xticks(x)
    ax.set_xticklabels([m.title() for m in model_names])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_optimal_temperatures(results_dict, output_path):
    model_names = list(results_dict.keys())
    optimal_temps = [results_dict[m]['optimal_temperature'] for m in model_names]
    ece_reductions = [results_dict[m]['improvements']['ece_reduction_percent'] for m in model_names]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = ['red' if t > 1.0 else 'blue' if t < 1.0 else 'gray' for t in optimal_temps]
    bars = ax1.bar(model_names, optimal_temps, color=colors, edgecolor='black', alpha=0.7)
    ax1.axhline(y=1.0, color='black', linestyle='--', label='Original (T=1.0)')
    
    for bar, temp in zip(bars, optimal_temps):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height, f'T={temp:.1f}', ha='center', va='bottom' if temp > 1.0 else 'top')
    
    ax1.set_ylabel('Temperature (T)')
    ax1.set_title('Optimal Temperature per Model')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    colors_reduction = ['darkgreen' if x > 0 else 'darkred' for x in ece_reductions]
    bars2 = ax2.bar(model_names, ece_reductions, color=colors_reduction, edgecolor='black', alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-')
    
    for bar, reduction in zip(bars2, ece_reductions):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height, f'{reduction:+.1f}%', ha='center', va='bottom' if reduction > 0 else 'top')
    
    ax2.set_ylabel('ECE Reduction (%)')
    ax2.set_title('Calibration Improvement')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print(f"Starting Temperature Scaling Analysis on {DEVICE}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_dataset = CARLADataset(VAL_DATA_PATH, VAL_DATA_PATH / "labels.csv", transform)
    test_dataset = CARLADataset(TEST_DATA_PATH, TEST_DATA_PATH / "labels.csv", transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    all_results = {}
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    for model_type, model_path in MODELS.items():
        print(f"Processing {model_type} classifier...")
        model = load_model(model_path, DEVICE)
        val_logits, val_labels = get_model_logits(model, val_loader, model_type, DEVICE)
        test_logits, test_labels = get_model_logits(model, test_loader, model_type, DEVICE)
        
        all_results[model_type] = analyze_temperature_scaling(
            model_type, val_logits, val_labels, test_logits, test_labels
        )
        
        res = all_results[model_type]
        print(f"  Optimal T: {res['optimal_temperature']:.1f}")
        print(f"  ECE Before: {res['before']['ece']:.4f} | ECE After: {res['after']['ece']:.4f}")
        print(f"  Improvement: {res['improvements']['ece_reduction_percent']:.1f}%\n")
    
    plot_temperature_curves(all_results, OUTPUT_DIR / "temperature_nll_curves.png")
    plot_ece_comparison(all_results, OUTPUT_DIR / "ece_comparison_before_after.png")
    plot_optimal_temperatures(all_results, OUTPUT_DIR / "optimal_temperatures.png")
    
    with open(OUTPUT_DIR / "temperature_scaling_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
        
    print("Execution complete. Assets saved to output directory.")

if __name__ == "__main__":
    main()