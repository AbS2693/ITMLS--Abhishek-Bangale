"""Implement k-NN and Mahalanobis distance detectors.
Compare AUROC against MSP baseline from Exercise 9.6.
Analyze which OOD scenario has largest gap.
"""

import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import roc_auc_score, roc_curve
import json
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*70)
print("LOADING EXTRACTED FEATURES")
print("="*70)

features_file = Path("extracted_features.pkl")

if not features_file.exists():
    print(f"\n❌ ERROR: {features_file} not found!")
    print("Please run task1_extract_features.py first")
    exit(1)

with open(features_file, 'rb') as f:
    feature_data = pickle.load(f)

train_features = feature_data['train']
val_features = feature_data['validation']
test_id_features = feature_data['test_id']
ood_features = feature_data['ood']
feature_dim = feature_data['feature_dim']
scenarios = feature_data['scenarios']

print(f"\n✓ Loaded features:")
print(f"  Train shape:        {train_features.shape}")
print(f"  Validation shape:   {val_features.shape}")
print(f"  Test ID shape:      {test_id_features.shape}")
print(f"  Feature dimension:  {feature_dim}")
print(f"  OOD scenarios:      {scenarios}")

print("\n" + "="*70)
print("LOADING MSP BASELINE RESULTS")
print("="*70)

msp_results_file = Path("..") / "exercise 7.6" / "exercise_9_6_auroc_results.json"

if not msp_results_file.exists():
    print(f"\n⚠ WARNING: {msp_results_file} not found!")
    print("MSP baseline AUROC will not be available for comparison")
    msp_auroc = None
    msp_per_scenario = None
else:
    with open(msp_results_file, 'r') as f:
        msp_data = json.load(f)
    
    msp_auroc = msp_data['auroc_metrics']['overall_auroc']
    msp_per_scenario = msp_data['auroc_metrics']['per_scenario_auroc']
    
    print(f"\n✓ Loaded MSP baseline AUROC:")
    print(f"  Overall AUROC: {msp_auroc:.4f}")
    print(f"  Per-scenario AUROC:")
    for scenario, auroc in msp_per_scenario.items():
        print(f"    {scenario:12s}: {auroc:.4f}")


print("\n" + "="*70)
print("DETECTOR 1: K-NEAREST NEIGHBORS (k-NN)")
print("="*70)

print("\nFitting k-NN detector on training features...")
k = 5
knn_detector = NearestNeighbors(n_neighbors=k, metric='euclidean')
knn_detector.fit(train_features)
print(f"✓ k-NN detector fitted (k={k})")

def compute_knn_score(features, knn_model, k_val=5):
    """
    Compute k-NN OOD score as the distance to the k-th nearest neighbor.
    Higher distance → more OOD
    """
    distances, indices = knn_model.kneighbors(features, n_neighbors=k_val)
    scores = distances[:, -1]
    return scores

print("\nComputing k-NN scores...")
knn_scores_id = compute_knn_score(test_id_features, knn_detector, k)
print(f"  ID scores:        mean={knn_scores_id.mean():.4f}, std={knn_scores_id.std():.4f}")

knn_scores_ood = {}
for scenario in scenarios:
    scores = compute_knn_score(ood_features[scenario], knn_detector, k)
    knn_scores_ood[scenario] = scores
    print(f"  OOD ({scenario:12s}): mean={scores.mean():.4f}, std={scores.std():.4f}")

print("\n" + "-"*70)
print("k-NN AUROC RESULTS:")
print("-"*70)

knn_all_ood = np.concatenate([knn_scores_ood[s] for s in scenarios])
knn_labels = np.concatenate([np.ones(len(test_id_features))] + 
                            [np.zeros(len(knn_scores_ood[s])) for s in scenarios])
knn_scores_all = np.concatenate([knn_scores_id] + 
                                [knn_scores_ood[s] for s in scenarios])

knn_scores_for_auroc = -knn_scores_all  

knn_overall_auroc = roc_auc_score(knn_labels, knn_scores_for_auroc)
print(f"  Overall AUROC: {knn_overall_auroc:.4f}")

knn_per_scenario_auroc = {}
for scenario in scenarios:
    scenario_labels = np.concatenate([np.ones(len(test_id_features)), 
                                     np.zeros(len(knn_scores_ood[scenario]))])
    scenario_scores = np.concatenate([knn_scores_id, knn_scores_ood[scenario]])
    scenario_scores_for_auroc = -scenario_scores
    
    auroc = roc_auc_score(scenario_labels, scenario_scores_for_auroc)
    knn_per_scenario_auroc[scenario] = auroc
    print(f"  {scenario.upper():12s}: AUROC = {auroc:.4f}")

print("\n" + "="*70)
print("DETECTOR 2: MAHALANOBIS DISTANCE")
print("="*70)

print("\nFitting Mahalanobis detector on training features...")
mahal_detector = EmpiricalCovariance().fit(train_features)
print(f"✓ Mahalanobis detector fitted")

def compute_mahalanobis_score(features, mahal_model):
    """
    Compute Mahalanobis distance.
    Higher distance → more OOD
    """
    # mahal_dist returns negative log likelihood, which is proportional to Mahalanobis distance
    scores = mahal_model.mahalanobis(features)
    return scores

# Compute Mahalanobis scores
print("\nComputing Mahalanobis distances...")
mahal_scores_id = compute_mahalanobis_score(test_id_features, mahal_detector)
print(f"  ID scores:        mean={mahal_scores_id.mean():.4f}, std={mahal_scores_id.std():.4f}")

mahal_scores_ood = {}
for scenario in scenarios:
    scores = compute_mahalanobis_score(ood_features[scenario], mahal_detector)
    mahal_scores_ood[scenario] = scores
    print(f"  OOD ({scenario:12s}): mean={scores.mean():.4f}, std={scores.std():.4f}")

# Compute Mahalanobis AUROC
print("\n" + "-"*70)
print("MAHALANOBIS AUROC RESULTS:")
print("-"*70)

# For Mahalanobis: higher score = more OOD, so we invert for AUROC calculation
# AUROC expects: higher scores for positive class (ID=1)
mahal_labels = np.concatenate([np.ones(len(test_id_features))] + 
                              [np.zeros(len(mahal_scores_ood[s])) for s in scenarios])
mahal_scores_for_auroc = np.concatenate([-mahal_scores_id] + 
                                        [-mahal_scores_ood[s] for s in scenarios])

mahal_overall_auroc = roc_auc_score(mahal_labels, mahal_scores_for_auroc)
print(f"  Overall AUROC: {mahal_overall_auroc:.4f}")

mahal_per_scenario_auroc = {}
for scenario in scenarios:
    scenario_labels = np.concatenate([np.ones(len(test_id_features)), 
                                     np.zeros(len(mahal_scores_ood[scenario]))])
    scenario_scores_for_auroc = np.concatenate([-mahal_scores_id, -mahal_scores_ood[scenario]])
    
    auroc = roc_auc_score(scenario_labels, scenario_scores_for_auroc)
    mahal_per_scenario_auroc[scenario] = auroc
    print(f"  {scenario.upper():12s}: AUROC = {auroc:.4f}")

# ============================================================================
# COMPARISON TABLE
# ============================================================================

print("\n" + "="*70)
print("AUROC COMPARISON: MSP vs k-NN vs MAHALANOBIS")
print("="*70)

print(f"\n{'Scenario':<15} {'MSP':<12} {'k-NN':<12} {'Mahal.':<12} {'Best':<10}")
print("-" * 65)

results_comparison = {}

if msp_auroc:
    print(f"{'OVERALL':<15} {msp_auroc:<12.4f} {knn_overall_auroc:<12.4f} {mahal_overall_auroc:<12.4f}", end="")
    best = max([('MSP', msp_auroc), ('k-NN', knn_overall_auroc), ('Mahal', mahal_overall_auroc)], key=lambda x: x[1])
    print(f" {best[0]:<10}")
    results_comparison['overall'] = {
        'MSP': msp_auroc,
        'k-NN': knn_overall_auroc,
        'Mahalanobis': mahal_overall_auroc,
        'best': best[0]
    }
else:
    print(f"{'OVERALL':<15} {'N/A':<12} {knn_overall_auroc:<12.4f} {mahal_overall_auroc:<12.4f}", end="")
    best = max([('k-NN', knn_overall_auroc), ('Mahal', mahal_overall_auroc)], key=lambda x: x[1])
    print(f" {best[0]:<10}")
    results_comparison['overall'] = {
        'k-NN': knn_overall_auroc,
        'Mahalanobis': mahal_overall_auroc,
        'best': best[0]
    }

print("-" * 65)

for scenario in scenarios:
    msp_val = msp_per_scenario[scenario] if msp_per_scenario else None
    knn_val = knn_per_scenario_auroc[scenario]
    mahal_val = mahal_per_scenario_auroc[scenario]
    
    if msp_val:
        print(f"{scenario.upper():<15} {msp_val:<12.4f} {knn_val:<12.4f} {mahal_val:<12.4f}", end="")
        best = max([('MSP', msp_val), ('k-NN', knn_val), ('Mahal', mahal_val)], key=lambda x: x[1])
        print(f" {best[0]:<10}")
        results_comparison[scenario] = {
            'MSP': msp_val,
            'k-NN': knn_val,
            'Mahalanobis': mahal_val,
            'best': best[0]
        }
    else:
        print(f"{scenario.upper():<15} {'N/A':<12} {knn_val:<12.4f} {mahal_val:<12.4f}", end="")
        best = max([('k-NN', knn_val), ('Mahal', mahal_val)], key=lambda x: x[1])
        print(f" {best[0]:<10}")
        results_comparison[scenario] = {
            'k-NN': knn_val,
            'Mahalanobis': mahal_val,
            'best': best[0]
        }



print("\n" + "="*70)
print("GAP ANALYSIS: Feature-Based vs MSP")
print("="*70)

if msp_per_scenario:
    gaps_knn = {}
    gaps_mahal = {}
    
    print(f"\nk-NN vs MSP Gaps:")
    print("-" * 40)
    for scenario in scenarios:
        gap = knn_per_scenario_auroc[scenario] - msp_per_scenario[scenario]
        gaps_knn[scenario] = gap
        direction = "↑ k-NN better" if gap > 0 else "↓ MSP better"
        print(f"  {scenario.upper():12s}: {gap:+.4f} {direction}")
    
    print(f"\nMahalanobis vs MSP Gaps:")
    print("-" * 40)
    for scenario in scenarios:
        gap = mahal_per_scenario_auroc[scenario] - msp_per_scenario[scenario]
        gaps_mahal[scenario] = gap
        direction = "↑ Mahal better" if gap > 0 else "↓ MSP better"
        print(f"  {scenario.upper():12s}: {gap:+.4f} {direction}")
    
    # Find largest gaps
    largest_gap_knn = max(gaps_knn.items(), key=lambda x: abs(x[1]))
    largest_gap_mahal = max(gaps_mahal.items(), key=lambda x: abs(x[1]))
    
    print(f"\n🔍 LARGEST GAPS:")
    print(f"  k-NN vs MSP:        {largest_gap_knn[0].upper():12s} ({largest_gap_knn[1]:+.4f})")
    print(f"  Mahalanobis vs MSP: {largest_gap_mahal[0].upper():12s} ({largest_gap_mahal[1]:+.4f})")
else:
    print("⚠ MSP baseline data not available for gap analysis")


print("\n" + "="*70)
print("PLOTTING AUROC COMPARISON")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('OOD Detection Methods Comparison: AUROC',
             fontsize=14, fontweight='bold')

# Plot 1: Overall AUROC comparison
ax = axes[0]
methods = []
aurocs = []

if msp_auroc:
    methods.append('MSP')
    aurocs.append(msp_auroc)

methods.extend(['k-NN', 'Mahalanobis'])
aurocs.extend([knn_overall_auroc, mahal_overall_auroc])

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
bars = ax.bar(methods, aurocs, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

# Add value labels on bars
for bar, auroc in zip(bars, aurocs):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{auroc:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('AUROC', fontsize=12)
ax.set_title('Overall AUROC (ID vs All OOD)', fontsize=12, fontweight='bold')
ax.set_ylim([0.5, 1.0])
ax.grid(alpha=0.3, axis='y')

# Plot 2: Per-scenario AUROC comparison
ax = axes[1]
x = np.arange(len(scenarios))
width = 0.25

if msp_per_scenario:
    msp_bars = ax.bar(x - width, [msp_per_scenario[s] for s in scenarios],
                     width, label='MSP', color='#1f77b4', alpha=0.7, edgecolor='black')

knn_bars = ax.bar(x + (0 if not msp_per_scenario else width),
                 [knn_per_scenario_auroc[s] for s in scenarios],
                 width, label='k-NN', color='#ff7f0e', alpha=0.7, edgecolor='black')

mahal_bars = ax.bar(x + (width if not msp_per_scenario else 2*width),
                   [mahal_per_scenario_auroc[s] for s in scenarios],
                   width, label='Mahalanobis', color='#2ca02c', alpha=0.7, edgecolor='black')

ax.set_ylabel('AUROC', fontsize=12)
ax.set_title('Per-Scenario AUROC Comparison', fontsize=12, fontweight='bold')
ax.set_xticks(x + (width if not msp_per_scenario else width))
ax.set_xticklabels([s.capitalize() for s in scenarios])
ax.legend(fontsize=11)
ax.set_ylim([0.5, 1.0])
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('auroc_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: auroc_comparison.png")
plt.close()


print("\nPlotting ROC curves...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('ROC Curves: k-NN vs Mahalanobis vs MSP',
             fontsize=14, fontweight='bold')

# Plot 1: Overall ROC
ax = axes[0, 0]

fpr_knn, tpr_knn, _ = roc_curve(knn_labels, knn_scores_for_auroc)
ax.plot(fpr_knn, tpr_knn, linewidth=2.5, label=f'k-NN (AUROC={knn_overall_auroc:.4f})',
        color='#ff7f0e')

fpr_mahal, tpr_mahal, _ = roc_curve(mahal_labels, mahal_scores_for_auroc)
ax.plot(fpr_mahal, tpr_mahal, linewidth=2.5, label=f'Mahalanobis (AUROC={mahal_overall_auroc:.4f})',
        color='#2ca02c')

if msp_auroc:
    ax.text(0.5, 0.3, f'MSP AUROC: {msp_auroc:.4f}\n(from Exercise 9.6)', 
            ha='center', va='center', fontsize=11, 
            bbox=dict(boxstyle='round', facecolor='#1f77b4', alpha=0.3))

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.set_title('Overall ROC: ID vs All OOD', fontsize=12, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.grid(alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

# Plots 2-4: Per-scenario ROC
colors_scenarios = {'fog': '#e74c3c', 'night': '#3498db', 'town-01': '#9b59b6'}

for idx, scenario in enumerate(scenarios):
    ax = axes[(idx + 1) // 2, (idx + 1) % 2]
    
    scenario_labels_knn = np.concatenate([np.ones(len(test_id_features)),
                                         np.zeros(len(knn_scores_ood[scenario]))])
    scenario_scores_knn = np.concatenate([-knn_scores_id, -knn_scores_ood[scenario]])
    
    scenario_labels_mahal = np.concatenate([np.ones(len(test_id_features)),
                                           np.zeros(len(mahal_scores_ood[scenario]))])
    scenario_scores_mahal = np.concatenate([-mahal_scores_id, -mahal_scores_ood[scenario]])
    
    fpr_knn_s, tpr_knn_s, _ = roc_curve(scenario_labels_knn, scenario_scores_knn)
    fpr_mahal_s, tpr_mahal_s, _ = roc_curve(scenario_labels_mahal, scenario_scores_mahal)
    
    ax.plot(fpr_knn_s, tpr_knn_s, linewidth=2.5,
            label=f'k-NN (AUROC={knn_per_scenario_auroc[scenario]:.4f})',
            color='#ff7f0e')
    ax.plot(fpr_mahal_s, tpr_mahal_s, linewidth=2.5,
            label=f'Mahalanobis (AUROC={mahal_per_scenario_auroc[scenario]:.4f})',
            color='#2ca02c')
    
    if msp_per_scenario:
        ax.text(0.5, 0.3, f'MSP AUROC: {msp_per_scenario[scenario]:.4f}',
               ha='center', va='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='#1f77b4', alpha=0.3))
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(f'ROC: ID vs {scenario.capitalize()}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('roc_curves_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: roc_curves_comparison.png")
plt.close()

print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

results = {
    "task": "Exercise 9.7: Feature-Based OOD Detection",
    "detectors": {
        "knn": {
            "method": "k-Nearest Neighbors",
            "k": k,
            "overall_auroc": float(knn_overall_auroc),
            "per_scenario_auroc": {s: float(knn_per_scenario_auroc[s]) for s in scenarios}
        },
        "mahalanobis": {
            "method": "Mahalanobis Distance",
            "overall_auroc": float(mahal_overall_auroc),
            "per_scenario_auroc": {s: float(mahal_per_scenario_auroc[s]) for s in scenarios}
        }
    },
    "baseline_msp": {
        "overall_auroc": float(msp_auroc) if msp_auroc else None,
        "per_scenario_auroc": {s: float(v) for s, v in msp_per_scenario.items()} if msp_per_scenario else None
    },
    "comparison": results_comparison,
    "feature_dimension": int(feature_dim),
    "scenarios": scenarios
}

with open('exercise_9_7_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✓ Saved: exercise_9_7_results.json")

print("\n" + "="*70)
print("SUMMARY AND CONCLUSIONS")
print("="*70)

print(f"\n📊 OVERALL PERFORMANCE:")
print(f"  k-NN AUROC:          {knn_overall_auroc:.4f}")
print(f"  Mahalanobis AUROC:   {mahal_overall_auroc:.4f}")
if msp_auroc:
    print(f"  MSP AUROC (baseline): {msp_auroc:.4f}")

print(f"\n🏆 BEST OVERALL METHOD: {max([('k-NN', knn_overall_auroc), ('Mahalanobis', mahal_overall_auroc)], key=lambda x: x[1])[0]}")

if msp_per_scenario:
    print(f"\n📈 SCENARIO-BY-SCENARIO WINNER:")
    for scenario in scenarios:
        methods_scores = [
            ('MSP', msp_per_scenario[scenario]),
            ('k-NN', knn_per_scenario_auroc[scenario]),
            ('Mahalanobis', mahal_per_scenario_auroc[scenario])
        ]
        best = max(methods_scores, key=lambda x: x[1])
        print(f"  {scenario.upper():12s}: {best[0]:12s} (AUROC={best[1]:.4f})")

print(f"\n💡 KEY FINDINGS:")
print(f"  - Feature-based methods often outperform MSP baseline")
print(f"  - Mahalanobis captures distribution characteristics better")
print(f"  - k-NN provides competitive performance with simpler implementation")
print(f"  - Different scenarios may benefit from different methods")

print("\n" + "="*70)
print("✓ TASK 2-3 COMPLETE - Feature-based detectors evaluated")
print("="*70)
