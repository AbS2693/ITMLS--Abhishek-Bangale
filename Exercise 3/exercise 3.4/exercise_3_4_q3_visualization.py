import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

base_path = Path("..")
train_path = base_path / "train" / "train"

def load_labels():
    labels_path = train_path / "labels.csv"
    return pd.read_csv(labels_path)

def get_samples_for_combination(labels_df, traffic_light, pedestrian, vehicle):
    mask = (
        (labels_df['has_traffic_light'] == traffic_light) &
        (labels_df['has_pedestrian'] == pedestrian) &
        (labels_df['has_vehicle'] == vehicle)
    )
    return labels_df[mask]['frame'].values

def load_image(frame_number):
    rgb_path = train_path / "rgb-front" / f"{frame_number:06d}.jpg"
    if rgb_path.exists():
        return Image.open(rgb_path)
    return None

def create_horizontal_visualization():
    labels_df = load_labels()
    
    combinations = [
        (False, False, False, "No Traffic Light\nNo Pedestrian\nNo Vehicle"),
        (True, False, False, "Traffic Light\nNo Pedestrian\nNo Vehicle"),
        (False, True, False, "No Traffic Light\nPedestrian\nNo Vehicle"),
        (False, False, True, "No Traffic Light\nNo Pedestrian\nVehicle"),
        (True, True, False, "Traffic Light\nPedestrian\nNo Vehicle"),
        (True, False, True, "Traffic Light\nNo Pedestrian\nVehicle"),
        (False, True, True, "No Traffic Light\nPedestrian\nVehicle"),
        (True, True, True, "Traffic Light\nPedestrian\nVehicle"),
    ]
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('CARLA Dataset: Sample Images for Label Combinations\n(Training Split - Horizontal View)', 
                 fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, (traffic_light, pedestrian, vehicle, label_text) in enumerate(combinations):
        matching_frames = get_samples_for_combination(labels_df, traffic_light, pedestrian, vehicle)
        
        ax = axes[idx]
        
        if len(matching_frames) > 0:
            sample_idx = len(matching_frames) // 2
            frame_num = matching_frames[sample_idx]
            
            img = load_image(frame_num)
            if img is not None:
                ax.imshow(img)
                ax.set_title(label_text, fontsize=10, fontweight='bold')
                ax.text(0.5, -0.15, f'Frame: {frame_num:06d}\nCount: {len(matching_frames)}', 
                       ha='center', transform=ax.transAxes, fontsize=8)
            else:
                ax.text(0.5, 0.5, 'Image not found', ha='center', va='center')
        else:
            ax.text(0.5, 0.5, f'No samples\n(0 images)', ha='center', va='center', fontsize=10)
        
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('exercise_3_4_q3_label_combinations.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: exercise_3_4_q3_label_combinations.png")
    plt.close()

def create_detailed_pattern_analysis():
    
    print("\n" + "=" * 80)
    print("EXERCISE 3.4 QUESTION 3: PATTERN ANALYSIS")
    print("=" * 80)
    
    combinations = [
        (False, False, False, "All Absent (Empty Road)"),
        (True, False, False, "Only Traffic Light"),
        (False, True, False, "Only Pedestrian"),
        (False, False, True, "Only Vehicle"),
        (True, True, False, "Traffic Light + Pedestrian"),
        (True, False, True, "Traffic Light + Vehicle"),
        (False, True, True, "Pedestrian + Vehicle"),
        (True, True, True, "All Present"),
    ]
    
    print("\nLabel Combination Statistics:\n")
    total = len(labels_df)
    
    for traffic_light, pedestrian, vehicle, description in combinations:
        matching = get_samples_for_combination(labels_df, traffic_light, pedestrian, vehicle)
        count = len(matching)
        percentage = (count / total) * 100
        print(f"{description:30s}: {count:5d} images ({percentage:5.1f}%)")

def create_pattern_observations():
    """Print key pattern observations."""
    labels_df = load_labels()
    
    print("\n" + "=" * 80)
    print("KEY PATTERNS OBSERVED:")
    print("=" * 80)
    
    print("\n1. EMPTY ROAD SCENES (All False):")
    empty = get_samples_for_combination(labels_df, False, False, False)
    print(f"   - {len(empty)} images ({len(empty)/len(labels_df)*100:.1f}%)")
    print("   - Question: Is the road completely empty?")
    print("   - Observation: These represent highway segments or open road areas with")
    print("                 minimal traffic objects, useful for negative training samples.")
    
    print("\n2. TRAFFIC LIGHT SCENES (Traffic Light = True):")
    traffic = labels_df[labels_df['has_traffic_light'] == True]
    print(f"   - Total: {len(traffic)} images ({len(traffic)/len(labels_df)*100:.1f}%)")
    print("   - Question: Are they always at intersections?")
    print("   - Observation: Traffic lights typically appear at intersections/road junctions")
    print("                 where vehicle control is critical for safety.")
    
    print("\n3. PEDESTRIAN SCENES (Pedestrian = True):")
    pedestrian = labels_df[labels_df['has_pedestrian'] == True]
    print(f"   - Total: {len(pedestrian)} images ({len(pedestrian)/len(labels_df)*100:.1f}%)")
    print("   - Question: Are pedestrians on sidewalks or crossing?")
    print("   - Observation: Pedestrian detection is critical safety requirement.")
    print(f"                 MINORITY class - only {len(pedestrian)/len(labels_df)*100:.1f}% of data!")
    print("                 Models may struggle with this sparse category.")
    
    print("\n4. VEHICLE SCENES (Vehicle = True):")
    vehicle = labels_df[labels_df['has_vehicle'] == True]
    print(f"   - Total: {len(vehicle)} images ({len(vehicle)/len(labels_df)*100:.1f}%)")
    print("   - Question: What types of vehicles appear?")
    print("   - Observation: MAJORITY class - vehicles are frequent in urban driving.")
    print("                 High occurrence helps model generalization but may bias")
    print("                 predictions toward 'vehicle present'.")
    
    print("\n5. MULTI-OBJECT SCENES (Multiple = True):")
    multi = labels_df[(labels_df['has_traffic_light'] == True) & 
                      (labels_df['has_pedestrian'] == True) & 
                      (labels_df['has_vehicle'] == True)]
    print(f"   - Total: {len(multi)} images ({len(multi)/len(labels_df)*100:.1f}%)")
    print("   - Observation: Complex urban intersections with all elements present.")
    print("                 These are high-safety-risk scenarios requiring accurate")
    print("                 multi-class detection.")
    
    print("\n" + "=" * 80)
    print("SAFETY-CRITICAL INSIGHTS FOR EXERCISE 3.5/3.6:")
    print("=" * 80)
    print("""
1. CLASS IMBALANCE PROBLEM:
   - Pedestrian detection is SEVERELY imbalanced (~24% positive)
   - Simple baseline that predicts "no pedestrian" would get 76% accuracy!
   - Must use Recall and F1-score metrics (not just accuracy)

2. SAFETY IMPLICATIONS:
   - Pedestrians = High severity (injury risk)
   - Traffic Lights = Medium severity (collision risk)
   - Vehicles = Medium severity (collision risk)
   - False negatives are more dangerous than false positives

3. TRAINING STRATEGY:
   - Use class weights: give higher weight to minority class (pedestrian)
   - Use appropriate loss function: weighted cross-entropy
   - Consider stratified sampling or oversampling techniques
   - Validate with Precision/Recall/F1 rather than accuracy alone
    """)

if __name__ == "__main__":
    print("Creating Exercise 3.4 Question 3 Visualization...\n")
    
    # Create visualizations
    create_horizontal_visualization()
    create_detailed_pattern_analysis()
    create_pattern_observations()
    
    print("\n" + "=" * 80)
    print("Exercise 3.4 Question 3 complete!")
    print("=" * 80)


