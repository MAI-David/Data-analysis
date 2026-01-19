"""
Example: Feature Selection with Neural Networks

This example demonstrates how to use neural network-based stability selection
to identify the most important features for prediction.
"""

import sys
sys.path.append('../notebooks')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from functions import nn_feature_search, set_global_seeds

print("=" * 60)
print("Example: Neural Network Feature Selection")
print("=" * 60)

# Set random seeds for reproducibility
print("\n1. Setting random seeds...")
set_global_seeds(42)

# Step 2: Load processed data
print("\n2. Loading processed data...")
data = pd.read_csv('../data/processed/merged_samples.csv')
print(f"   Loaded data: {data.shape}")

# Step 3: Prepare features and target
print("\n3. Preparing features and target...")
# Remove metadata columns
metadata_cols = ['sample_id', 'age_group_at_sample', 'sex', 'family_id', 
                 'year_of_birth', 'body_product']
feature_cols = [col for col in data.columns if col not in metadata_cols]

X = data[feature_cols]
y = data['age_group_at_sample']

# Encode target if needed
from sklearn.preprocessing import LabelEncoder
if y.dtype == 'object':
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y), name='age_group_encoded')
    print(f"   Target classes: {list(le.classes_)}")

print(f"   Features: {X.shape}")
print(f"   Target: {y.shape}")

# Step 4: Split data
print("\n4. Splitting data (80/20 train/test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train: {X_train.shape}, Test: {X_test.shape}")

# Step 5: Run feature selection
print("\n5. Running neural network feature selection...")
print("   This may take several minutes...")

result = nn_feature_search(
    X_train=X_train,
    X_test=X_test,
    Y_train=y_train,
    target_range=(50, 500),  # Target 50-500 features
    consensus_threshold=0.7,  # Features selected in >=70% of runs
    use_checkpointing=True
)

# Step 6: Analyze results
if result is not None:
    print("\n6. Feature selection results:")
    print(f"   Selected features: {result.n_features}")
    print(f"   Training RMSE: {result.rmse:.2f}")
    
    print("\n   Top 20 selected features:")
    for i, feature in enumerate(result.feature_names[:20], 1):
        # Truncate long feature names
        display_name = feature if len(feature) <= 60 else feature[:57] + "..."
        print(f"   {i:2d}. {display_name}")
    
    # Step 7: Save selected features
    print("\n7. Saving selected features...")
    selected_features_df = pd.DataFrame({
        'feature_name': result.feature_names,
        'rank': range(1, len(result.feature_names) + 1)
    })
    selected_features_df.to_csv('../data/processed/selected_features.csv', index=False)
    print("   Saved to: ../data/processed/selected_features.csv")
    
    # Save reduced datasets
    result.X_train_elite.to_csv('../data/processed/X_train_selected.csv', index=False)
    result.X_test_elite.to_csv('../data/processed/X_test_selected.csv', index=False)
    print("   Saved reduced train/test sets")
    
else:
    print("\n   Feature selection failed - no penalty met target range")
    print("   Try adjusting target_range or consensus_threshold")

print("\n" + "=" * 60)
print("Feature selection complete!")
print("=" * 60)

# Tips for tuning
print("\nTuning tips:")
print("  - Increase target_range max for more features")
print("  - Decrease consensus_threshold for more features")
print("  - Increase consensus_threshold for more stable features")
print("  - Check training RMSE to ensure model performance")
