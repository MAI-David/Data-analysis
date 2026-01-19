# Usage Examples

This directory contains practical examples demonstrating how to use the data analysis pipeline.

## Prerequisites

1. Install all dependencies:
   ```bash
   pip install -r ../notebooks/requirements.txt
   ```

2. Place your data in `../data/raw/`:
   - `MAI3004_lucki_mpa411.csv` (abundance data)
   - `MAI3004_lucki_metadata_safe.csv` (sample metadata)

## Examples

### 01_data_loading.py

**Purpose**: Basic data loading and preprocessing

**What it demonstrates**:
- Loading MetaPhlAn abundance data
- Loading sample metadata
- Merging datasets
- Basic data exploration
- Missing value analysis

**Run time**: < 1 minute

**Usage**:
```bash
cd examples
python 01_data_loading.py
```

**Output**:
- `../data/processed/merged_samples.csv` - Merged dataset ready for analysis

---

### 02_feature_selection.py

**Purpose**: Neural network-based feature selection

**What it demonstrates**:
- Setting random seeds for reproducibility
- Preparing features and target
- Running stability selection with neural networks
- Analyzing selected features
- Saving results

**Run time**: 10-30 minutes (depending on hardware)

**Usage**:
```bash
cd examples
python 02_feature_selection.py
```

**Output**:
- `../data/processed/selected_features.csv` - List of selected features
- `../data/processed/X_train_selected.csv` - Training set with selected features
- `../data/processed/X_test_selected.csv` - Test set with selected features

**Tuning parameters**:
- `target_range`: Adjust (min, max) features to select
- `consensus_threshold`: Higher = more stable features, lower = more features
- `use_checkpointing`: Set to True to save model checkpoints

---

### 03_model_benchmarking.py

**Purpose**: Train and compare multiple ML models

**What it demonstrates**:
- Training XGBoost, Random Forest, LightGBM, Gradient Boosting
- Hyperparameter tuning with RandomizedSearchCV
- Model comparison metrics (RMSE, R², runtime)
- Feature importance analysis

**Run time**: 20-60 minutes (depending on data size and hardware)

**Usage**:
```bash
cd examples
python 03_model_benchmarking.py
```

**Output**:
- `../data/processed/model_comparison.csv` - Comparison of all models
- `../data/processed/top_features_*.csv` - Top features for each model

**Notes**:
- Automatically uses selected features if available (faster)
- Falls back to full dataset if feature selection not run
- Uses cross-validation for robust performance estimation

---

## Running Order

For best results, run examples in order:

1. **First**: `01_data_loading.py` - Creates merged dataset
2. **Second** (optional): `02_feature_selection.py` - Reduces feature space
3. **Third**: `03_model_benchmarking.py` - Trains and compares models

## Tips and Tricks

### Memory Management

If you encounter memory issues:
- Use feature selection (`02_feature_selection.py`) before benchmarking
- Reduce `target_range` in feature selection
- Close other applications
- Consider using a system with more RAM (8GB+ recommended)

### Speed Optimization

To speed up execution:
- Use selected features instead of full dataset
- Reduce cross-validation folds
- Reduce `n_iter` in RandomizedSearchCV
- Use GPU for neural network feature selection (if available)

### Reproducibility

All examples use fixed random seeds:
- Python random: 42
- NumPy: 42
- TensorFlow: 42
- Cross-validation: 3004 (some models)

To change seeds, modify the `set_global_seeds()` calls.

### Custom Data

To use your own data:
1. Ensure your abundance data follows MetaPhlAn 4 format
2. Ensure metadata has required columns:
   - `sample_id`: Unique identifier
   - `age_group_at_sample`: Target variable
   - Other metadata columns as needed
3. Place files in `../data/raw/`
4. Update file names in example scripts if different

## Troubleshooting

### "File not found" errors

**Problem**: Data files not found

**Solution**: 
- Check that data files are in `../data/raw/`
- Verify file names match expected names
- Run `01_data_loading.py` first to create processed data

### "Module not found" errors

**Problem**: Missing dependencies

**Solution**:
```bash
pip install -r ../notebooks/requirements.txt
```

### GPU/CUDA errors

**Problem**: TensorFlow GPU errors in feature selection

**Solution**:
- The code will automatically fall back to CPU
- Or install appropriate CUDA/ROCm drivers for your GPU
- Or set `CUDA_VISIBLE_DEVICES=""` to force CPU mode

### Memory errors

**Problem**: Out of memory during training

**Solution**:
- Run feature selection first to reduce dimensionality
- Use smaller `n_iter` in hyperparameter search
- Close other applications
- Use a machine with more RAM

## Advanced Usage

### Modifying Examples

Examples are designed to be modified:

```python
# Adjust feature selection parameters
result = nn_feature_search(
    X_train=X_train,
    X_test=X_test,
    Y_train=y_train,
    target_range=(100, 300),  # More restrictive
    consensus_threshold=0.8,   # More stable features
    use_checkpointing=True
)

# Adjust model hyperparameters
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=10,
    cv=3,
    n_jobs=-1
)
```

### Creating Your Own Examples

Feel free to create additional examples:

1. Copy an existing example as a template
2. Modify for your specific use case
3. Add documentation at the top
4. Consider contributing back via pull request

## Additional Resources

- **Main notebook**: `../notebooks/data-pipeline.ipynb` - Comprehensive analysis
- **Functions module**: `../notebooks/functions.py` - All available functions
- **Documentation**: `../README.md` - Project overview
- **Data format**: `../data/raw/metaphlan411_data_description.md`

## Questions?

- Check the main README: `../README.md`
- Review function docstrings: `../notebooks/functions.py`
- Open an issue on GitHub

---

**Version**: 1.0.0  
**Last Updated**: 2026-01-19
