"""
Example: Model Benchmarking and Comparison

This example demonstrates how to train and compare multiple machine learning
models on microbiome data.
"""

import sys
sys.path.append('../notebooks')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from functions import (
    set_global_seeds,
    xgboost_benchmark,
    random_forest_benchmark,
    lightgbm_benchmark,
    gradient_boosting_benchmark
)

print("=" * 60)
print("Example: Model Benchmarking and Comparison")
print("=" * 60)

# Step 1: Set seeds for reproducibility
print("\n1. Setting random seeds...")
set_global_seeds(42)

# Step 2: Load data
print("\n2. Loading data...")
# Option A: Use selected features (faster)
try:
    X_train = pd.read_csv('../data/processed/X_train_selected.csv')
    X_test = pd.read_csv('../data/processed/X_test_selected.csv')
    print(f"   Loaded selected features: {X_train.shape[1]} features")
    
    # When using pre-split data, we need to load and split the target the same way
    # Load full data and perform the same split to get corresponding targets
    data = pd.read_csv('../data/processed/merged_samples.csv')
    y_full = data['age_group_at_sample']
    
    # Encode if needed
    if y_full.dtype == 'object':
        le = LabelEncoder()
        y_full_encoded = pd.Series(le.fit_transform(y_full))
    else:
        y_full_encoded = y_full
    
    # Recreate the same train/test split used in feature selection
    # (80/20 split with random_state=42, stratified)
    _, _, y_train, y_test = train_test_split(
        data.drop(columns=['age_group_at_sample']),  # X placeholder
        y_full_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_full_encoded
    )
    
except FileNotFoundError:
    # Option B: Use all features
    print("   Selected features not found, loading full dataset...")
    data = pd.read_csv('../data/processed/merged_samples.csv')
    metadata_cols = ['sample_id', 'age_group_at_sample', 'sex', 'family_id']
    feature_cols = [col for col in data.columns if col not in metadata_cols]
    X = data[feature_cols]
    y = data['age_group_at_sample']
    
    # Encode if needed
    if y.dtype == 'object':
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y))
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

print(f"   Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# Step 3: Benchmark models
print("\n3. Benchmarking models...")
print("   Note: This will take several minutes per model\n")

results = {}

# XGBoost
print("\n" + "-" * 60)
print("Training XGBoost...")
print("-" * 60)
results['XGBoost'] = xgboost_benchmark(
    X_train, X_test, y_train, y_test,
    label="XGBoost Benchmark"
)

# Random Forest
print("\n" + "-" * 60)
print("Training Random Forest...")
print("-" * 60)
results['Random Forest'] = random_forest_benchmark(
    X_train, X_test, y_train, y_test,
    label="Random Forest Benchmark"
)

# LightGBM
print("\n" + "-" * 60)
print("Training LightGBM...")
print("-" * 60)
results['LightGBM'] = lightgbm_benchmark(
    X_train, X_test, y_train, y_test,
    label="LightGBM Benchmark"
)

# Gradient Boosting
print("\n" + "-" * 60)
print("Training Gradient Boosting...")
print("-" * 60)
results['Gradient Boosting'] = gradient_boosting_benchmark(
    X_train, X_test, y_train, y_test,
    label="Gradient Boosting Benchmark"
)

# Step 4: Compare results
print("\n" + "=" * 60)
print("Model Comparison Results")
print("=" * 60)

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'RMSE': [r.rmse for r in results.values()],
    'R²': [r.r2 for r in results.values()],
    'Runtime (s)': [r.runtime for r in results.values()]
})

# Sort by R² (descending)
comparison_df = comparison_df.sort_values('R²', ascending=False)
print("\n", comparison_df.to_string(index=False))

# Step 5: Show best model details
print("\n" + "=" * 60)
best_model_name = comparison_df.iloc[0]['Model']
best_result = results[best_model_name]
print(f"Best Model: {best_model_name}")
print("=" * 60)
print(f"RMSE: {best_result.rmse:.4f}")
print(f"R²: {best_result.r2:.4f}")
print(f"Runtime: {best_result.runtime:.2f} seconds")
print(f"\nBest Hyperparameters:")
for param, value in best_result.best_params.items():
    print(f"  {param}: {value}")

print(f"\nTop 10 Most Important Features:")
print(best_result.top_features.head(10).to_string(index=False))

# Step 6: Save results
print("\n" + "=" * 60)
print("Saving results...")
comparison_df.to_csv('../data/processed/model_comparison.csv', index=False)
print("   Saved comparison to: ../data/processed/model_comparison.csv")

for model_name, result in results.items():
    filename = f"../data/processed/top_features_{model_name.lower().replace(' ', '_')}.csv"
    result.top_features.to_csv(filename, index=False)
    print(f"   Saved {model_name} top features to: {filename}")

print("\n" + "=" * 60)
print("Model benchmarking complete!")
print("=" * 60)

# Recommendations
print("\nRecommendations:")
print(f"  Best overall model: {best_model_name}")
print(f"  For interpretability: Check Random Forest or XGBoost feature importances")
print(f"  For speed: LightGBM typically fastest")
print(f"  For accuracy: {best_model_name} achieved highest R²")
