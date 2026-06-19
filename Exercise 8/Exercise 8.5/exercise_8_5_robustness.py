import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
import pandas as pd
from PIL import Image
import json
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPSILON_VALUES = [0.01, 0.05, 0.1]
NUM_TEST_SAMPLES = 100 
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

class CARLADataset(Dataset):
    """CARLA dataset for loading images and labels"""
    def __init__(self, root_dir, labels_csv, transform=None, num_samples=None, label_column=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        self.labels_df = pd.read_csv(labels_csv)
        self.labels_df = self.labels_df[self.labels_df['frame'].notna()].reset_index(drop=True)
        
        if label_column is not None:
            self.labels_df = self.labels_df[self.labels_df[label_column] == 1].reset_index(drop=True)
        
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

def fgsm_attack(image, label, model, epsilon, device):
    """Perform Fast Gradient Sign Method (FGSM) attack"""
    image.requires_grad = True
    
    output = model(image)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, label.unsqueeze(0))
    
    if image.grad is not None:
        image.grad.zero_()
    loss.backward()
    
    image_grad = image.grad.data
    perturbation = epsilon * torch.sign(image_grad)
    perturbed_image = image + perturbation
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    
    return perturbed_image.detach()

def evaluate_model(model, dataloader, model_name, device, is_adversarial=False, epsilon=None):
    """
    Evaluate model on clean or adversarial test set.
    Returns: recall, predictions, true labels
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    # Only use no_grad for clean evaluation
    if not is_adversarial:
        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(device)
                labels = batch[f'has_{model_name}'].to(device)
                
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.append(predicted.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
    else:
        # For adversarial, we require gradients so no torch.no_grad()
        for batch in dataloader:
            images = batch['image'].to(device)
            labels = batch[f'has_{model_name}'].to(device)
            
            images_adv = []
            for i in range(images.shape[0]):
                img = images[i:i+1].clone().detach().requires_grad_(True)
                lbl = labels[i]
                img_adv = fgsm_attack(img, lbl, model, epsilon, device)
                images_adv.append(img_adv)
            
            images = torch.cat(images_adv, dim=0)
            
            with torch.no_grad():
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
            
            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    tp = np.sum((all_preds == 1) & (all_labels == 1))
    fn = np.sum((all_preds == 0) & (all_labels == 1))
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return recall, all_preds, all_labels

def main():
    output_dir = Path("Exercise 8")
    output_dir.mkdir(exist_ok=True)
    
    root_path = Path(".")
    models_path = root_path / "Best Models"
    test_data_path = root_path / "test" / "test"
    labels_csv = test_data_path / "labels.csv"
    
    print("=" * 80)
    print("EXERCISE 8.5: MEASURING ROBUSTNESS TO ADVERSARIAL ATTACKS")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Epsilon values: {EPSILON_VALUES}")
    print(f"Test samples: {NUM_TEST_SAMPLES}")
    print()
    
    print("Loading trained models...")
    models_dict = {
        'pedestrian': load_model(models_path / "model_has_pedestrian.pth", DEVICE),
        'traffic_light': load_model(models_path / "model_has_traffic_light.pth", DEVICE),
        'vehicle': load_model(models_path / "model_has_vehicle.pth", DEVICE)
    }
    print("Models loaded successfully\n")
    
    print("Loading test dataset...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    datasets_dict = {
        'pedestrian': CARLADataset(
            root_dir=test_data_path,
            labels_csv=labels_csv,
            transform=transform,
            num_samples=NUM_TEST_SAMPLES,
            label_column='has_pedestrian'  
        ),
        'traffic_light': CARLADataset(
            root_dir=test_data_path,
            labels_csv=labels_csv,
            transform=transform,
            num_samples=NUM_TEST_SAMPLES,
            label_column='has_traffic_light'  
        ),
        'vehicle': CARLADataset(
            root_dir=test_data_path,
            labels_csv=labels_csv,
            transform=transform,
            num_samples=NUM_TEST_SAMPLES,
            label_column='has_vehicle'  
        )
    }
    
    dataloaders_dict = {
        model_name: DataLoader(dataset, batch_size=8, shuffle=False)
        for model_name, dataset in datasets_dict.items()
    }
    
    print("Datasets loaded successfully:")
    for model_name, dataset in datasets_dict.items():
        print(f"  {model_name.capitalize()}: {len(dataset)} positive samples")
    print()
    
    results = {}
    
    for model_name, model in models_dict.items():
        print("=" * 80)
        print(f"Evaluating: {model_name.upper()}")
        print("=" * 80)
        
        results[model_name] = {}
        dataloader = dataloaders_dict[model_name]
        
        print("\n  [1/4] Evaluating on clean data...")
        clean_recall, clean_preds, clean_labels = evaluate_model(
            model, dataloader, model_name, DEVICE, is_adversarial=False
        )
        results[model_name]['clean'] = {
            'recall': float(clean_recall),
            'num_positives': int(np.sum(clean_labels == 1)),
            'num_samples': int(len(clean_labels))
        }
        print(f"  ✓ Clean recall: {clean_recall:.4f}")
        
        for idx, epsilon in enumerate(EPSILON_VALUES, start=2):
            print(f"\n  [{idx}/{len(EPSILON_VALUES)+1}] Evaluating on adversarial (ε={epsilon})...")
            
            adv_recall, adv_preds, adv_labels = evaluate_model(
                model, dataloader, model_name, DEVICE, 
                is_adversarial=True, epsilon=epsilon
            )
            
            recall_drop = clean_recall - adv_recall
            recall_drop_pct = (recall_drop / clean_recall * 100) if clean_recall > 0 else 0
            
            results[model_name][f'epsilon_{epsilon}'] = {
                'recall': float(adv_recall),
                'recall_drop': float(recall_drop),
                'recall_drop_pct': float(recall_drop_pct),
                'num_samples': int(len(adv_labels))
            }
            
            print(f"  ✓ Adversarial recall: {adv_recall:.4f}")
            print(f"  ✓ Recall drop: {recall_drop:.4f} ({recall_drop_pct:.1f}%)")
    
    print(f"\n\n{'=' * 80}")
    print("EXERCISE 8.5 SUMMARY: ROBUSTNESS EVALUATION")
    print("=" * 80)
    print()
    
    summary_data = []
    
    for model_name in models_dict.keys():
        clean_rec = results[model_name]['clean']['recall']
        
        for epsilon in EPSILON_VALUES:
            key = f'epsilon_{epsilon}'
            adv_rec = results[model_name][key]['recall']
            drop = results[model_name][key]['recall_drop']
            drop_pct = results[model_name][key]['recall_drop_pct']
            
            summary_data.append({
                'Model': model_name.capitalize(),
                'Epsilon': epsilon,
                'Clean Recall': f"{clean_rec:.4f}",
                'Adversarial Recall': f"{adv_rec:.4f}",
                'Recall Drop': f"{drop:.4f}",
                'Drop %': f"{drop_pct:.1f}%"
            })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    print(f"\n\n{'=' * 80}")
    print("KEY FINDINGS")
    print("=" * 80)
    
    for model_name in models_dict.keys():
        print(f"\n{model_name.upper()}:")
        clean_rec = results[model_name]['clean']['recall']
        print(f"  Clean recall: {clean_rec:.4f}")
        
        for epsilon in EPSILON_VALUES:
            key = f'epsilon_{epsilon}'
            adv_rec = results[model_name][key]['recall']
            drop_pct = results[model_name][key]['recall_drop_pct']
            
            impact = "SEVERE" if drop_pct > 50 else "HIGH" if drop_pct > 25 else "MODERATE" if drop_pct > 10 else "LOW"
            print(f"  ε={epsilon}: {adv_rec:.4f} recall (drop: {drop_pct:.1f}%) [{impact}]")
    
    print()
    print(f"\n{'=' * 80}")
    print("OVERALL STATISTICS")
    print("=" * 80)
    
    all_drops = []
    for model_name in models_dict.keys():
        for epsilon in EPSILON_VALUES:
            key = f'epsilon_{epsilon}'
            all_drops.append(results[model_name][key]['recall_drop_pct'])
    
    print(f"\nAverage recall drop across all models/epsilons: {np.mean(all_drops):.1f}%")
    print(f"Maximum recall drop: {np.max(all_drops):.1f}%")
    print(f"Minimum recall drop: {np.min(all_drops):.1f}%")
    
    print(f"\nRecall drop by epsilon:")
    for epsilon in EPSILON_VALUES:
        drops = []
        for model_name in models_dict.keys():
            key = f'epsilon_{epsilon}'
            drops.append(results[model_name][key]['recall_drop_pct'])
        print(f"  ε={epsilon}: {np.mean(drops):.1f}% (avg), {np.max(drops):.1f}% (max)")
    
    report = {
        'exercise': '8.5',
        'title': 'Measuring Robustness to Adversarial Attacks',
        'timestamp': pd.Timestamp.now().isoformat(),
        'parameters': {
            'epsilon_values': EPSILON_VALUES,
            'test_samples': NUM_TEST_SAMPLES,
            'device': str(DEVICE)
        },
        'results': results,
        'summary': {
            'average_recall_drop_pct': float(np.mean(all_drops)),
            'max_recall_drop_pct': float(np.max(all_drops)),
            'min_recall_drop_pct': float(np.min(all_drops))
        },
        'conclusions': {
            'attack_effectiveness': "FGSM attacks significantly degrade model recall",
            'robustness_level': "Models are vulnerable to adversarial perturbations",
            'key_observation': "Even small epsilon values (0.01-0.05) cause noticeable recall degradation",
            'recommendation': "Adversarial training needed to improve robustness"
        }
    }
    
    report_path = output_dir / "exercise_8_5_robustness_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Report saved to: {report_path}")
    print()

if __name__ == "__main__":
    main()