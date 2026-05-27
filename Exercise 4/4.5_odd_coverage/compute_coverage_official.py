import sys
import os
import pandas as pd
import json
from pathlib import Path

GITHUB_LIB_PATH = r"c:\Users\ABI\Desktop\Sub Docs\Semester 5\ITMLS\2026\Exercise 4\4.5_odd_coverage\odd-coverage"
sys.path.insert(0, GITHUB_LIB_PATH)

from kprojection import KProjectionCoverage

WORKSPACE_ROOT = r"c:\Users\ABI\Desktop\Sub Docs\Semester 5\ITMLS\2026"
TRAIN_DIR = os.path.join(WORKSPACE_ROOT, "train", "train")

TEST_VARIANTS = {
    "test_normal": os.path.join(WORKSPACE_ROOT, "test", "test"),
    "test_fog": os.path.join(WORKSPACE_ROOT, "test-fog", "test-fog"),
    "test_night": os.path.join(WORKSPACE_ROOT, "test-night", "test-night"),
    "test_town_01": os.path.join(WORKSPACE_ROOT, "test-town-01", "test-town-01"),
}

OUTPUT_DIR = os.path.join(WORKSPACE_ROOT, "Exercise 4", "4.5_odd_coverage", "results_official")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 90)
print("Exercise 4.5: k-Projection Coverage using GitHub Library")
print("=" * 90)

def load_feather_file(filepath):
    try:
        return pd.read_feather(filepath)
    except Exception as e:
        print(f"    ⚠ Error loading {filepath}: {e}")
        return None

def extract_and_discretize_scenarios(dataset_dir, dataset_name, max_samples=None):
    print(f"\n[*] Extracting scenarios from {dataset_name}...")
    
    scenarios = []

    weather_file = os.path.join(dataset_dir, "weather.feather")
    actions_file = os.path.join(dataset_dir, "actions.feather")
    
    weather_df = load_feather_file(weather_file) if os.path.exists(weather_file) else None
    actions_df = load_feather_file(actions_file) if os.path.exists(actions_file) else None
    
    if weather_df is None or actions_df is None:
        print(f"    ⚠ Missing required data files")
        return scenarios
    
    num_samples = min(len(weather_df), len(actions_df))
    if max_samples:
        num_samples = min(num_samples, max_samples)
    
    print(f"    Processing {num_samples} samples...")
    
    for idx in range(num_samples):
        try:
            precip = weather_df.iloc[idx].get('precipitation', 0)
            fog_density = weather_df.iloc[idx].get('fog_density', 0)
            
            if precip < 0.1 and fog_density < 1.0:
                weather_state = "clear"
            elif precip < 0.5:
                weather_state = "cloudy"
            else:
                weather_state = "rainy"
            
            sun_alt = weather_df.iloc[idx].get('sun_altitude_angle', 45)
            
            if sun_alt > 30:
                lighting_state = "daytime"
            elif sun_alt > 0:
                lighting_state = "dusk_dawn"
            else:
                lighting_state = "night"
            
            throttle = actions_df.iloc[idx].get('throttle', 0)
            
            if throttle < 0.3:
                speed_state = "low"
            elif throttle < 0.7:
                speed_state = "medium"
            else:
                speed_state = "high"
            
            scenario = {
                "weather": weather_state,
                "lighting": lighting_state,
                "speed": speed_state,
            }
            
            scenarios.append(scenario)
            
        except Exception as e:
            continue
    
    print(f"    ✓ Extracted {len(scenarios)} scenarios")
    
    # Print distribution
    weather_dist = {}
    lighting_dist = {}
    speed_dist = {}
    
    for s in scenarios:
        weather_dist[s['weather']] = weather_dist.get(s['weather'], 0) + 1
        lighting_dist[s['lighting']] = lighting_dist.get(s['lighting'], 0) + 1
        speed_dist[s['speed']] = speed_dist.get(s['speed'], 0) + 1
    
    print(f"      Weather: {weather_dist}")
    print(f"      Lighting: {lighting_dist}")
    print(f"      Speed: {speed_dist}")
    
    return scenarios

print("\n[*] Defining ODD Description...")

odd_description = {
    "weather": ["clear", "cloudy", "rainy"],
    "lighting": ["daytime", "dusk_dawn", "night"],
    "speed": ["low", "medium", "high"],
}

print(f"    ODD dimensions: {list(odd_description.keys())}")
for dim, values in odd_description.items():
    print(f"      {dim}: {values}")

train_scenarios = extract_and_discretize_scenarios(TRAIN_DIR, "Training Set", max_samples=7200)
test_scenarios = extract_and_discretize_scenarios(TEST_VARIANTS["test_normal"], "Test Set (Normal)", max_samples=3600)

print("\n" + "=" * 90)
print("COMPUTING k-PROJECTION COVERAGE (OFFICIAL GITHUB LIBRARY)")
print("=" * 90)

all_results = {}

# For each k value
for k in [1, 2, 3]:
    print(f"\n[*] Computing k={k} projection coverage...")
    
    coverage_metric = KProjectionCoverage(k=k, desc=odd_description)
    
    for scenario in train_scenarios:
        coverage_metric.add_scenario(scenario)
    
    train_result = coverage_metric.compute()
    print(f"    Training set coverage for k={k}:")
    print(f"      Covered: {train_result.covered}/{train_result.total}")
    print(f"      Coverage: {train_result.coverage * 100:.2f}%")
    
    coverage_metric.reset()
    
    for scenario in test_scenarios:
        coverage_metric.add_scenario(scenario)
    
    test_result = coverage_metric.compute()
    print(f"    Test set coverage for k={k}:")
    print(f"      Covered: {test_result.covered}/{test_result.total}")
    print(f"      Coverage: {test_result.coverage * 100:.2f}%")
    
    all_results[f'k={k}'] = {
        'coverage_percentage': round(test_result.coverage * 100, 2),
        'covered_points': test_result.covered,
        'total_points': test_result.total,
        'num_scenarios': test_result.scenes,
    }

print("\n" + "=" * 90)
print("COMPARING TEST VARIANTS")
print("=" * 90)

variant_results = {}

for variant_name, variant_path in TEST_VARIANTS.items():
    if not os.path.exists(variant_path):
        print(f"⚠ {variant_name} not found")
        continue
    
    print(f"\n[*] Analyzing {variant_name}...")
    test_var_scenarios = extract_and_discretize_scenarios(variant_path, variant_name, max_samples=3600)
    
    variant_coverage = {}
    
    for k in [1, 2, 3]:
        cov_metric = KProjectionCoverage(k=k, desc=odd_description)
        
        for scenario in test_var_scenarios:
            cov_metric.add_scenario(scenario)
        
        result = cov_metric.compute()
        variant_coverage[f'k={k}'] = {
            'coverage_percentage': round(result.coverage * 100, 2),
            'covered_points': result.covered,
            'total_points': result.total,
        }
        
        print(f"    k={k}: {result.coverage*100:.2f}% ({result.covered}/{result.total})")
    
    variant_results[variant_name] = variant_coverage

print("\n" + "=" * 90)
print("SAVING RESULTS")
print("=" * 90)

results_data = {
    'library': 'odd-coverage (GitHub: kkirchheim/odd-coverage)',
    'paper': 'Quantitative Projection Coverage for Testing ML-enabled Autonomous Systems',
    'odd_description': odd_description,
    'test_normal': all_results,
    'all_variants': variant_results,
}

results_file = os.path.join(OUTPUT_DIR, "coverage_results_official.json")
with open(results_file, "w") as f:
    json.dump(results_data, f, indent=2)

print(f"✓ Results saved to: {results_file}")

# Create summary
summary_file = os.path.join(OUTPUT_DIR, "coverage_summary_official.txt")
with open(summary_file, "w") as f:
    f.write("k-PROJECTION COVERAGE ANALYSIS RESULTS\n")
    f.write("(Using Official GitHub Library: kkirchheim/odd-coverage)\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("OFFICIAL ODD DESCRIPTION:\n")
    for dim, values in odd_description.items():
        f.write(f"  {dim}: {values}\n")
    
    f.write("\nNORMAL TEST SET k-PROJECTION COVERAGE:\n")
    for k, result in all_results.items():
        f.write(f"{k}: {result['coverage_percentage']}% ")
        f.write(f"({result['covered_points']}/{result['total_points']} points)\n")
    
    f.write("\n\nALL TEST VARIANTS COMPARISON:\n")
    for variant, coverage_data in variant_results.items():
        f.write(f"\n{variant}:\n")
        for k, data in coverage_data.items():
            f.write(f"  {k}: {data['coverage_percentage']}% ")
            f.write(f"({data['covered_points']}/{data['total_points']})\n")

print(f"✓ Summary saved to: {summary_file}")

print("\n" + "=" * 90)
print("✅ OFFICIAL COMPUTATION COMPLETE")
print("=" * 90)
print(f"\nResults saved to: {OUTPUT_DIR}")
