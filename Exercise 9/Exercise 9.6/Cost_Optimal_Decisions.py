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
OUTPUT_DIR = Path(__file__).parent

PEDESTRIAN_MODEL_PATH = MODELS_PATH / "model_has_pedestrian.pth"

# Cost parameters and Thresholds
C_FN = 100
C_FP = 1
THRESHOLD_STANDARD = 0.5
THRESHOLD_OPTIMAL = 0.0099  # τ* = C_FN/(C_FN + C_FP) ≈ 100/101

# Temperatures
TEMP_ORIGINAL = 1.0
TEMP_OPTIMAL = 2.1

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

def get_model_logits_and_labels(model, dataloader, device):
    model.eval()
    all_logits, all_labels = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            labels = batch['has_pedestrian'].to(device)
            
            logits = model(images)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
    return np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)

def apply_temperature_scaling(logits, temperature):
    scaled_logits = logits / temperature
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

def compute_metrics(predictions, labels, threshold):
    false_negatives = np.sum((predictions == 0) & (labels == 1))
    false_positives = np.sum((predictions == 1) & (labels == 0))
    true_positives = np.sum((predictions == 1) & (labels == 1))
    true_negatives = np.sum((predictions == 0) & (labels == 0))
    
    return {
        'threshold': float(threshold),
        'true_positives': int(true_positives),
        'true_negatives': int(true_negatives),
        'false_positives': int(false_positives),
        'false_negatives': int(false_negatives),
        'accuracy': float(np.mean(predictions == labels)),
        'total_loss': float(C_FN * false_negatives + C_FP * false_positives),
        'c_fn': int(C_FN),
        'c_fp': int(C_FP)
    }

def analyze_cost_optimal_decisions(logits, labels):
    results = {}
    configs = [
        ('uncalibrated', TEMP_ORIGINAL),
        ('calibrated', TEMP_OPTIMAL)
    ]
    thresholds = [
        ('standard', THRESHOLD_STANDARD),
        ('optimal', THRESHOLD_OPTIMAL)
    ]

    for model_name, temp in configs:
        probs = apply_temperature_scaling(logits, temp)
        prob_class_1 = probs[:, 1]
        
        for thresh_name, thresh in thresholds:
            preds = (prob_class_1 >= thresh).astype(int)
            results[f"{model_name}_{thresh_name}"] = compute_metrics(preds, labels, thresh)

    return results

def plot_comparison_table(results, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    unc_std = results['uncalibrated_standard']['total_loss']
    unc_opt = results['uncalibrated_optimal']['total_loss']
    cal_std = results['calibrated_standard']['total_loss']
    cal_opt = results['calibrated_optimal']['total_loss']
    min_loss = min(unc_std, unc_opt, cal_std, cal_opt)
    
    table_data = [
        ['', 'τ = 0.5\n(Standard)', 'τ* = 0.0099\n(Optimal)'],
        ['Uncalibrated\n(T=1.0)', f'Loss = {unc_std:.0f}', f'Loss = {unc_opt:.0f}'],
        ['Calibrated\n(T=2.1)', f'Loss = {cal_std:.0f}', f'Loss = {cal_opt:.0f}']
    ]
    
    cell_colors = [
        ['white', 'lightgray', 'lightgray'],
        ['lightgray', 
         '#FFB6B6' if unc_std != min_loss else '#90EE90',
         '#FFB6B6' if unc_opt != min_loss else '#90EE90'],
        ['lightgray',
         '#FFB6B6' if cal_std != min_loss else '#90EE90',
         '#90EE90'] 
    ]
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center', cellColours=cell_colors, colWidths=[0.3, 0.35, 0.35])
    table.scale(1, 3)
    table.set_fontsize(12)
    
    plt.title(f'Cost-Optimal Decision Making\nC_FN={C_FN}, C_FP={C_FP}', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_loss_bar_chart(results, output_path):
    scenarios = ['Unc.\nτ=0.5', 'Unc.\nτ*', 'Cal.\nτ=0.5', 'Cal.\nτ*']
    losses = [
        results['uncalibrated_standard']['total_loss'],
        results['uncalibrated_optimal']['total_loss'],
        results['calibrated_standard']['total_loss'],
        results['calibrated_optimal']['total_loss']
    ]
    colors = ['#FFB6B6', '#FF6B6B', '#FFE699', '#90EE90']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(scenarios, losses, color=colors, edgecolor='black', alpha=0.8)
    
    for bar, loss in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{loss:.0f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Total Loss (L)')
    ax.set_title(f'Total Loss Comparison (C_FN={C_FN}, C_FP={C_FP})')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_fn_fp_comparison(results, output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    scenarios = ['Unc.\nτ=0.5', 'Unc.\nτ*', 'Cal.\nτ=0.5', 'Cal.\nτ*']
    
    fn_values = [results[k]['false_negatives'] for k in results.keys()]
    fp_values = [results[k]['false_positives'] for k in results.keys()]
    
    x = np.arange(len(scenarios))
    
    bars_fn = ax1.bar(x, fn_values, color='#FF6B6B', edgecolor='darkred', alpha=0.8)
    ax1.set_ylabel('False Negatives (#FN)')
    ax1.set_title('False Negatives (Missing Pedestrians)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios)
    ax1.grid(True, alpha=0.3, axis='y')
    
    bars_fp = ax2.bar(x, fp_values, color='#FFE699', edgecolor='orange', alpha=0.8)
    ax2.set_ylabel('False Positives (#FP)')
    ax2.set_title('False Positives (False Alarms)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for ax_obj, bars in zip([ax1, ax2], [bars_fn, bars_fp]):
        for bar in bars:
            ax_obj.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{int(bar.get_height())}', ha='center', va='bottom', fontweight='bold')

    plt.suptitle(f'Error Type Comparison | C_FN={C_FN}, C_FP={C_FP}', fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print(f"Starting Cost-Optimal Decisions Analysis on {DEVICE}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = CARLADataset(TEST_DATA_PATH, TEST_DATA_PATH / "labels.csv", transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print("Loading pedestrian model and computing logits...")
    model = load_model(PEDESTRIAN_MODEL_PATH, DEVICE)
    logits, labels = get_model_logits_and_labels(model, test_loader, DEVICE)
    
    results = analyze_cost_optimal_decisions(logits, labels)
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    plot_comparison_table(results, OUTPUT_DIR / "cost_optimal_comparison_table.png")
    plot_loss_bar_chart(results, OUTPUT_DIR / "cost_optimal_loss_comparison.png")
    plot_fn_fp_comparison(results, OUTPUT_DIR / "fn_fp_comparison.png")
    
    with open(OUTPUT_DIR / "cost_optimal_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print("\nSummary Results:")
    print(f"{'Scenario':<30} | {'#FN':>5} | {'#FP':>5} | {'Loss':>10}")
    print("-" * 58)
    for key, metrics in results.items():
        print(f"{key.replace('_', ' ').title():<30} | {metrics['false_negatives']:>5} | {metrics['false_positives']:>5} | {metrics['total_loss']:>10.0f}")

    print("\nExecution complete. Assets saved to output directory.")

if __name__ == "__main__":
    main()