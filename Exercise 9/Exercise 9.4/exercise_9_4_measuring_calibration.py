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

MODELS = {
    'pedestrian': MODELS_PATH / "model_has_pedestrian.pth",
    'traffic_light': MODELS_PATH / "model_has_traffic_light.pth",
    'vehicle': MODELS_PATH / "model_has_vehicle.pth"
}

class CARLADataset(Dataset):
    def __init__(self, root_dir, labels_csv, transform=None, label_column=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        self.labels_df = pd.read_csv(labels_csv)
        self.labels_df = self.labels_df[self.labels_df['frame'].notna()].reset_index(drop=True)
        
        if label_column is not None:
            self.labels_df = self.labels_df[self.labels_df[label_column] == 1].reset_index(drop=True)
    
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

def get_model_predictions(model, dataloader, model_type, device):
    model.eval()
    all_confidences, all_predictions, all_labels = [], [], []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            labels = batch[f'has_{model_type}'].to(device)
            
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            all_confidences.append(confidence.cpu().numpy())
            all_predictions.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    return (
        np.concatenate(all_confidences, axis=0),
        np.concatenate(all_predictions, axis=0),
        np.concatenate(all_labels, axis=0)
    )

def compute_ece(confidences, predictions, labels, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_sums = np.zeros(n_bins)
    bin_true = np.zeros(n_bins)
    bin_total = np.zeros(n_bins)
    
    correct = (predictions == labels).astype(float)
    
    for i in range(n_bins):
        in_bin = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(correct[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            bin_sums[i] = np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            bin_true[i] = accuracy_in_bin
            bin_total[i] = np.sum(in_bin)
            
    bin_data = {
        'bin_edges': bin_edges,
        'bin_total': bin_total,
        'bin_true': bin_true,
        'bin_sums': bin_sums
    }
    
    return np.sum(bin_sums), bin_data

def compute_metrics(confidences, predictions, labels):
    ece, bin_data = compute_ece(confidences, predictions, labels)
    accuracy = np.mean(predictions == labels)
    
    correct = (predictions == labels).astype(bool)
    confidence_correct = confidences[correct]
    confidence_incorrect = confidences[~correct]
    
    return {
        'ece': ece,
        'accuracy': accuracy,
        'avg_confidence': np.mean(confidences),
        'confidence_std': np.std(confidences),
        'avg_confidence_correct': np.mean(confidence_correct) if len(confidence_correct) > 0 else 0,
        'avg_confidence_incorrect': np.mean(confidence_incorrect) if len(confidence_incorrect) > 0 else 0,
        'num_samples': len(labels),
        'num_correct': np.sum(correct),
        'num_incorrect': np.sum(~correct),
        'bin_data': bin_data
    }

def plot_reliability_diagram(metrics_dict, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Reliability Diagrams', fontsize=14, fontweight='bold')
    
    for idx, (model_name, metrics) in enumerate(metrics_dict.items()):
        ax = axes[idx]
        bin_data = metrics['bin_data']
        bin_edges = bin_data['bin_edges']
        bin_true = bin_data['bin_true']
        
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
        
        valid_bins = ~np.isnan(bin_true) & (bin_data['bin_total'] > 0)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        ax.bar(bin_centers[valid_bins], bin_true[valid_bins], width=0.08, 
               alpha=0.7, edgecolor='blue', label='Accuracy', color='steelblue')
        ax.plot(bin_centers[valid_bins], bin_centers[valid_bins], 'ro-', 
                linewidth=2, markersize=6, label='Confidence')
        
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{model_name.title()}\nECE={metrics["ece"]:.4f}')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confidence_distribution(metrics_dict, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Confidence Distribution: Correct vs Incorrect', fontsize=14, fontweight='bold')
    
    for idx, (model_name, metrics) in enumerate(metrics_dict.items()):
        ax = axes[idx]
        
        correct_mask = metrics['predictions'] == metrics['labels']
        ax.hist(metrics['confidences'][correct_mask], bins=20, alpha=0.6, 
                label='Correct', color='green', edgecolor='darkgreen')
        ax.hist(metrics['confidences'][~correct_mask], bins=20, alpha=0.6, 
                label='Incorrect', color='red', edgecolor='darkred')
        
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Count')
        ax.set_title(model_name.title())
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print(f"Starting Calibration Execution on {DEVICE}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = CARLADataset(TEST_DATA_PATH, TEST_DATA_PATH / "labels.csv", transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    all_metrics = {}
    results_summary = {}
    
    for model_type, model_path in MODELS.items():
        print(f"Evaluating {model_type} classifier...")
        model = load_model(model_path, DEVICE)
        confidences, predictions, labels = get_model_predictions(model, test_loader, model_type, DEVICE)
        
        metrics = compute_metrics(confidences, predictions, labels)
        all_metrics[model_type] = metrics
        all_metrics[model_type]['confidences'] = confidences
        all_metrics[model_type]['predictions'] = predictions
        all_metrics[model_type]['labels'] = labels
        
        gap = metrics['avg_confidence'] - metrics['accuracy']
        calibration_status = "OVERCONFIDENT" if gap > 0.05 else "UNDERCONFIDENT" if gap < -0.05 else "WELL-CALIBRATED"
        
        print(f"  Accuracy: {metrics['accuracy']:.4f} | ECE: {metrics['ece']:.4f}")
        print(f"  Status: {calibration_status}")
        
        results_summary[model_type] = {
            'accuracy': float(metrics['accuracy']),
            'ece': float(metrics['ece']),
            'avg_confidence': float(metrics['avg_confidence']),
            'calibration_status': calibration_status
        }

    OUTPUT_DIR.mkdir(exist_ok=True)
    
    plot_reliability_diagram(all_metrics, OUTPUT_DIR / "reliability_diagrams.png")
    plot_confidence_distribution(all_metrics, OUTPUT_DIR / "confidence_distributions.png")
    
    with open(OUTPUT_DIR / "calibration_results.json", 'w') as f:
        json.dump(results_summary, f, indent=2)
        
    print("\nExecution complete. Assets saved to output directory.")

if __name__ == "__main__":
    main()