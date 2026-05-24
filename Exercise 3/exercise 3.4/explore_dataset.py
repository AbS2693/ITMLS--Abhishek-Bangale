import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

base_path = Path("..")
datasets = {
    "train": base_path / "train" / "train",
    "validation": base_path / "validation" / "validation",
    "test": base_path / "test" / "test",
    "test-fog": base_path / "test-fog" / "test-fog",
    "test-night": base_path / "test-night" / "test-night",
    "test-town-01": base_path / "test-town-01" / "test-town-01"
}

def count_images(dataset_path):
    rgb_front_path = dataset_path / "rgb-front"
    if rgb_front_path.exists():
        images = list(rgb_front_path.glob("*.jpg"))
        return len(images)
    return 0

def load_labels(dataset_path):
    labels_path = dataset_path / "labels.csv"
    if labels_path.exists():
        return pd.read_csv(labels_path)
    return None

def print_dataset_info():
    print("=" * 80)
    print("CARLA DATASET EXPLORATION - EXERCISE 3.4 QUESTION 1")
    print("=" * 80)
    print("\nImage Count per Dataset:\n")
    
    total_images = 0
    image_counts = {}
    
    for dataset_name, dataset_path in datasets.items():
        if dataset_path.exists():
            count = count_images(dataset_path)
            image_counts[dataset_name] = count
            total_images += count
            print(f"{dataset_name:20s}: {count:8d} images")
        else:
            print(f"{dataset_name:20s}: Dataset not found")
    
    print("-" * 80)
    print(f"{'TOTAL':20s}: {total_images:8d} images")
    print("=" * 80)
    
    return image_counts

def explore_labels():
    """Explore label distributions in all datasets."""
    print("\n" + "=" * 80)
    print("LABEL DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    label_columns = ['has_traffic_light', 'has_pedestrian', 'has_vehicle']
    label_names = ['Traffic Light', 'Pedestrian', 'Vehicle']
    
    for dataset_name, dataset_path in datasets.items():
        if dataset_path.exists():
            labels = load_labels(dataset_path)
            if labels is not None:
                print(f"\n{dataset_name.upper()}:")
                print("-" * 80)
                print(f"Total samples: {len(labels)}")
                
                for col, name in zip(label_columns, label_names):
                    if col in labels.columns:
                        positive = (labels[col] == True).sum()
                        negative = (labels[col] == False).sum()
                        pos_pct = (positive / len(labels)) * 100
                        neg_pct = (negative / len(labels)) * 100
                        
                        print(f"\n  {name}:")
                        print(f"    Present (True):  {positive:6d} ({pos_pct:5.1f}%)")
                        print(f"    Absent (False):  {negative:6d} ({neg_pct:5.1f}%)")
                        balanced = 'Yes' if abs(pos_pct - 50) < 10 else 'No'
                        print(f"    Balanced:        {balanced}")
            else:
                print(f"\n{dataset_name}: No labels.csv found")

def visualize_label_distribution():
    """Create visualizations of label distributions."""
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    label_columns = ['has_traffic_light', 'has_pedestrian', 'has_vehicle']
    label_names = ['Traffic Light', 'Pedestrian', 'Vehicle']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Label Distribution Across Datasets', fontsize=16, fontweight='bold')
    
    for idx, dataset_name in enumerate(datasets.keys()):
        dataset_path = datasets[dataset_name]
        if dataset_path.exists():
            labels = load_labels(dataset_path)
            if labels is not None:
                ax = axes[idx // 3, idx % 3]
                
                # Count label combinations
                counts_per_label = []
                for col in label_columns:
                    if col in labels.columns:
                        pos_count = (labels[col] == True).sum()
                        counts_per_label.append(pos_count)
                
                ax.bar(label_names, counts_per_label, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
                ax.set_title(f'{dataset_name}\n(n={len(labels)})')
                ax.set_ylabel('Count of Present (True) Labels')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(axis='y', alpha=0.3)
    
    # Remove empty subplots
    for idx in range(len(datasets), 6):
        fig.delaxes(axes[idx // 3, idx % 3])
    
    plt.tight_layout()
    plt.savefig('label_distribution.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: label_distribution.png")
    plt.close()

def display_sample_images():
    """Display sample images from each dataset."""
    print("\n" + "=" * 80)
    print("SAMPLE IMAGES")
    print("=" * 80)
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    fig.suptitle('Sample Images from Each Dataset', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    try:
        from PIL import Image
        
        for idx, (dataset_name, dataset_path) in enumerate(datasets.items()):
            if idx < len(axes) and dataset_path.exists():
                rgb_path = dataset_path / "rgb-front" / "000000.jpg"
                if rgb_path.exists():
                    img = Image.open(rgb_path)
                    axes[idx].imshow(img)
                    axes[idx].set_title(f'{dataset_name}\n(Sample: 000000.jpg)')
                    axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(base_path / 'sample_images.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: sample_images.png")
        plt.close()
    except ImportError:
        print("Note: PIL not installed. Skipping image visualization.")

def generate_summary_report():
    """Generate a comprehensive summary report."""
    print("\n" + "=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80)
    
    summary_data = []
    
    for dataset_name, dataset_path in datasets.items():
        if dataset_path.exists():
            image_count = count_images(dataset_path)
            labels = load_labels(dataset_path)
            
            if labels is not None:
                summary_data.append({
                    'Dataset': dataset_name,
                    'Images': image_count,
                    'Labels': len(labels),
                    'Match': 'Yes' if image_count == len(labels) else 'No'
                })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        print("\n" + summary_df.to_string(index=False))

if __name__ == "__main__":
    # Run all exploration functions
    image_counts = print_dataset_info()
    explore_labels()
    visualize_label_distribution()
    display_sample_images()
    generate_summary_report()
    
    print("\n" + "=" * 80)
    print("Dataset exploration complete!")
    print("=" * 80)
