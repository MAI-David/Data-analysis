"""
Example: Basic Data Loading and Preprocessing

This example demonstrates how to load MetaPhlAn abundance data and metadata,
merge them, and perform basic preprocessing steps.

Requirements:
- Place your data files in data/raw/ directory:
  * MAI3004_lucki_mpa411.csv
  * MAI3004_lucki_metadata_safe.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Set up paths
data_dir = Path('../data/raw')
abundance_file = data_dir / 'MAI3004_lucki_mpa411.csv'
metadata_file = data_dir / 'MAI3004_lucki_metadata_safe.csv'

print("=" * 60)
print("Example: Basic Data Loading and Preprocessing")
print("=" * 60)

# Step 1: Load abundance data
print("\n1. Loading abundance data...")
data = pd.read_csv(abundance_file)
print(f"   Loaded abundance data: {data.shape}")
print(f"   Columns: {list(data.columns[:5])}...")

# Step 2: Load metadata
print("\n2. Loading metadata...")
metadata = pd.read_csv(metadata_file)
print(f"   Loaded metadata: {metadata.shape}")
print(f"   Columns: {list(metadata.columns)}")

# Step 3: Identify sample columns
print("\n3. Identifying sample columns...")
sample_cols = [col for col in data.columns if col.startswith('mpa411_')]
print(f"   Found {len(sample_cols)} sample columns")

# Step 4: Create transposed abundance table
print("\n4. Creating sample-centric abundance table...")
sample_abundances = data[['clade_name'] + sample_cols].set_index('clade_name').T
sample_abundances.index.name = 'sample_id'
sample_abundances = sample_abundances.reset_index()
print(f"   Transposed abundance table: {sample_abundances.shape}")

# Step 5: Merge with metadata
print("\n5. Merging abundance and metadata...")
# Find common samples
metadata_common = metadata[metadata['sample_id'].isin(sample_abundances['sample_id'])]
print(f"   Common samples in metadata: {len(metadata_common)}")

# Merge
merged_samples = pd.merge(
    metadata_common,
    sample_abundances,
    on='sample_id',
    how='inner'
)
print(f"   Merged dataset: {merged_samples.shape}")

# Step 6: Basic data exploration
print("\n6. Basic data exploration...")
print(f"   Sample ID range: {merged_samples['sample_id'].min()} to {merged_samples['sample_id'].max()}")
print(f"   Age groups: {merged_samples['age_group_at_sample'].value_counts().to_dict()}")
print(f"   Sex distribution: {merged_samples['sex'].value_counts().to_dict()}")

# Step 7: Check for missing values
print("\n7. Missing value summary...")
missing = merged_samples.isnull().sum()
missing_pct = (missing / len(merged_samples) * 100).round(2)
missing_summary = pd.DataFrame({
    'Missing': missing[missing > 0],
    'Percentage': missing_pct[missing > 0]
})
if len(missing_summary) > 0:
    print(missing_summary)
else:
    print("   No missing values found!")

# Step 8: Save processed data (optional)
print("\n8. Saving processed data...")
output_dir = Path('../data/processed')
output_dir.mkdir(parents=True, exist_ok=True)
merged_samples.to_csv(output_dir / 'merged_samples.csv', index=False)
print(f"   Saved to: {output_dir / 'merged_samples.csv'}")

print("\n" + "=" * 60)
print("Data loading and preprocessing complete!")
print("=" * 60)

# Summary statistics
print("\nDataset Summary:")
print(f"  Total samples: {len(merged_samples)}")
print(f"  Total features: {merged_samples.shape[1] - 5}")  # Excluding metadata columns
print(f"  Metadata columns: {list(merged_samples.columns[:5])}")
print(f"  Feature columns: {merged_samples.shape[1] - 5} microbial taxa")
