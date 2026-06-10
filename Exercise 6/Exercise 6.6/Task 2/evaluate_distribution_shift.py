import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import json
import os
import matplotlib.pyplot as plt

BASE = Path("c:/Users/ABI/Desktop/Sub Docs/Semester 5/ITMLS/2026")
MODELS_PATH = BASE / "Best Models"
OUTPUT = BASE / "Exercise 6" / "Exercise 6.6" / "Task 2" / "output"
OUTPUT.mkdir(parents=True, exist_ok=True)
DATASETS = {"test-fog": BASE / "test-fog" / "test-fog",
            "test-night": BASE / "test-night" / "test-night"}


IMAGE_LIMIT = 200
SAMPLE_SAVE = 5

from torchvision import models as tv_models


class SimpleCAM:
    def __init__(self, model, target_layer_name="layer3"):
        self.model = model
        self.target = target_layer_name
        self.feature = None
        for name, module in self.model.named_modules():
            if self.target in name:
                module.register_forward_hook(lambda m, i, o: setattr(self, 'feature', o.detach().float()))
                break
    def generate(self, x):
        x = x.float()
        with torch.no_grad():
            _ = self.model(x)
        if self.feature is None:
            return np.zeros((x.shape[2], x.shape[3]))
        feat = self.feature[0]
        cam = feat.mean(dim=0)
        cam = F.relu(cam)
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.cpu().numpy()


def create_resnet18():
    m = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m


def load_labels(dataset_path):
    labels = {}
    csv_path = dataset_path / 'labels.csv'
    if not csv_path.exists():
        return labels
    import csv
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = row.get('frame') or row.get('image_name')
            if frame is None:
                continue
            img = f"{frame}.jpg" if not frame.endswith('.jpg') else frame
            labels[img] = {
                'has_pedestrian': 1 if row.get('has_pedestrian','False').lower()=='true' else 0,
                'has_traffic_light': 1 if row.get('has_traffic_light','False').lower()=='true' else 0,
                'has_vehicle': 1 if row.get('has_vehicle','False').lower()=='true' else 0
            }
    return labels


def preprocess_image(path, size=(224,224)):
    img = Image.open(path).convert('RGB')
    img = img.resize(size)
    arr = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485,0.456,0.406])
    std = np.array([0.229,0.224,0.225])
    arr = (arr - mean) / std
    tensor = torch.from_numpy(arr.transpose(2,0,1)).unsqueeze(0)
    return tensor, img


def compute_metrics(counts):
    TP = counts['TP']
    FP = counts['FP']
    TN = counts['TN']
    FN = counts['FN']
    acc = (TP+TN) / max(1, TP+TN+FP+FN)
    prec = TP / (TP+FP) if (TP+FP)>0 else 0.0
    rec = TP / (TP+FN) if (TP+FN)>0 else 0.0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}


models_info = {
    'pedestrian': 'model_has_pedestrian.pth',
    'traffic_light': 'model_has_traffic_light.pth',
    'vehicle': 'model_has_vehicle.pth'
}

results = {}

for ds_name, ds_path in DATASETS.items():
    print(f"\nEvaluating on dataset: {ds_name}")
    labels = load_labels(ds_path)
    rgb = ds_path / 'rgb-front'
    if not rgb.exists():
        print(f"  No rgb-front folder at {rgb}")
        continue
    image_list = sorted(list(rgb.glob('*.jpg')))[:IMAGE_LIMIT]
    print(f"  Images to evaluate: {len(image_list)} (limit {IMAGE_LIMIT})")

    ds_results = {}

    for model_name, model_file in models_info.items():
        print(f"  Model: {model_name}")
        model_path = MODELS_PATH / model_file
        if not model_path.exists():
            print(f"    Model not found: {model_path}")
            continue
        model = create_resnet18()
        sd = torch.load(model_path, map_location='cpu')
        model.load_state_dict(sd)
        model.eval()
        model = model.float()
        cam = SimpleCAM(model, 'layer3')

        counts = {'TP':0,'FP':0,'TN':0,'FN':0}
        samples_saved = 0
        out_dir = OUTPUT / ds_name / model_name
        (out_dir).mkdir(parents=True, exist_ok=True)

        for p in image_list:
            img_name = p.name
            if img_name not in labels:
                continue
            true = labels[img_name].get(f'has_{model_name}', 0)
            tensor, pil = preprocess_image(p)
            
            try:
                param_dtype = next(model.parameters()).dtype
                tensor = tensor.to(dtype=param_dtype)
            except StopIteration:
                pass
            with torch.no_grad():
                out = model(tensor)
                probs = torch.softmax(out, dim=1)
                pred = int(torch.argmax(probs, dim=1).item())
                conf = float(probs[0,pred].item())
            if pred==1 and true==1:
                counts['TP'] += 1
            elif pred==1 and true==0:
                counts['FP'] += 1
            elif pred==0 and true==0:
                counts['TN'] += 1
            elif pred==0 and true==1:
                counts['FN'] += 1

            
            save_flag = False
            tag = None
            if pred!=true and samples_saved < SAMPLE_SAVE:
                save_flag = True
                tag = 'mispred'
            elif pred==true and samples_saved < SAMPLE_SAVE//2:
                save_flag = True
                tag = 'correct'

            if save_flag:
                heat = cam.generate(tensor)
                arr = np.array(pil)
                hres = cv2.resize(heat, (arr.shape[1], arr.shape[0]))
                heatmap = cv2.applyColorMap((hres*255).astype(np.uint8), cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(arr, 0.6, heatmap, 0.4, 0)
                fname = out_dir / f"{tag}_{img_name}"
                plt.figure(figsize=(10,4))
                plt.subplot(1,3,1); plt.imshow(arr); plt.title('Original'); plt.axis('off')
                plt.subplot(1,3,2); plt.imshow(hres, cmap='jet'); plt.title('CAM'); plt.axis('off')
                plt.subplot(1,3,3); plt.imshow(overlay); plt.title(f'Pred={pred} True={true}'); plt.axis('off')
                plt.tight_layout()
                plt.savefig(str(fname), dpi=100, bbox_inches='tight')
                plt.close()
                samples_saved += 1

        metrics = compute_metrics(counts)
        ds_results[model_name] = {'counts':counts, 'metrics':metrics}
        print(f"    Metrics: acc={metrics['accuracy']:.3f} prec={metrics['precision']:.3f} rec={metrics['recall']:.3f} f1={metrics['f1']:.3f}")

    results[ds_name] = ds_results


with open(OUTPUT / 'distribution_shift_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('\nDone. Results saved to', OUTPUT)
