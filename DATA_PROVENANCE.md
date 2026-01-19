# Data Provenance and Metadata

## Dataset Information

### Primary Data Source

**Name**: LuCKi Living Lab Cohort Study - Gut Microbiome Profiling

**Citation**:
```
Lucki Cohort Study Group (2015). The LuCKi Living Lab - An Innovative 
Epidemiological Approach to Study the Gut Microbiome Over the Lifespan.
BMC Public Health, 15, 1-11.
DOI: 10.1186/s12889-015-2255-7
```

**Study Description**: The LuCKi (Lund University Microbiome in Childhood) Living Lab is a longitudinal cohort study investigating the gut microbiome composition across different age groups and its relationship with health outcomes.

**Data Collection Period**: As described in the original publication (2015)

**Sample Type**: Fecal samples from human subjects

### Data Files

#### 1. Microbial Abundance Data

**File**: `data/raw/MAI3004_lucki_mpa411.csv` or `MAI3004_lucki_mpa411.txt`

**Format**: Tab-separated values (TSV) / Comma-separated values (CSV)

**Tool**: MetaPhlAn 4.1.1 (Metagenomic Phylogenetic Analysis)

**Database Version**: MetaPhlAn 4.1.1 marker gene database

**Description**: 
- Taxonomic profiles of microbial communities
- Each row represents a taxonomic clade (from Kingdom to Species-level Genome Bin)
- Each column represents a sample (prefixed with `mpa411_`)
- Values are relative abundances (percentages)

**Dimensions**: 6,903 rows (taxonomic features) × 932 columns (930 samples + 2 metadata columns)

**Taxonomic Levels**:
- k__ = Kingdom
- p__ = Phylum
- c__ = Class
- o__ = Order
- f__ = Family
- g__ = Genus
- s__ = Species
- t__ = SGB (Species-Level Genome Bin)

**Processing Pipeline**:
1. Raw sequencing reads (likely 16S rRNA or shotgun metagenomics)
2. Quality control and filtering
3. Taxonomic profiling with MetaPhlAn 4.1.1
4. Normalization to relative abundances

#### 2. Sample Metadata

**File**: `data/raw/MAI3004_lucki_metadata_safe.csv`

**Format**: Comma-separated values (CSV)

**Description**: Clinical and demographic metadata for each sample

**Dimensions**: 930 rows (samples) × 6 columns (metadata variables)

**Columns**:
- `sample_id`: Unique identifier linking to abundance data (format: `mpa411_*`)
- `age_group_at_sample`: Age group classification at time of sampling (categorical)
- `sex`: Biological sex (categorical)
- `family_id`: Family identifier for related individuals (categorical)
- `year_of_birth`: Birth year (numeric) - removed in preprocessing
- `body_product`: Sample type/body site (categorical) - removed in preprocessing

**Target Variable**: `age_group_at_sample` - used as the prediction target in machine learning models

**Privacy**: Data has been de-identified ("safe" designation indicates personally identifiable information removed)

## Data Processing History

### Version Control

**Version**: 1.0.0 (Initial analysis)

**Last Modified**: 2026-01-19

### Preprocessing Steps Applied

1. **Data Loading**:
   - Abundance table and metadata loaded separately
   - Sample ID matching between datasets

2. **Data Merging**:
   - Inner join on sample IDs
   - Samples present in both datasets retained

3. **Column Selection**:
   - Removed `year_of_birth` and `body_product` columns
   - Extracted sample columns (prefixed with `mpa411_`)
   - Transposed abundance table for analysis

4. **Encoding**:
   - Label encoding applied to `sex` and `family_id`
   - Label encoding applied to `age_group_at_sample` (target variable)

5. **Missing Data Handling**:
   - Rows with missing `age_group_at_sample` removed
   - Missingness analysis performed on remaining data

6. **Quality Control**:
   - Outlier detection using IQR method
   - Normality testing (Shapiro-Wilk)
   - No outliers removed (documented only)

### Data Transformations

**Feature Engineering**:
- Prevalence-based feature selection (features present in >threshold of samples)
- Taxonomic level filtering (Kingdom to Species levels)
- Neural network-based feature selection (GatekeeperLayer)
- Top N features by importance/prevalence

**Normalization**:
- StandardScaler applied before machine learning models
- Z-score normalization for some visualizations

**Dimensionality Reduction**:
- PCA (Principal Component Analysis) for visualization
- Feature selection reduces from 6,903 to <1,250 features

## Data Quality Metrics

### Completeness

- Abundance Data: Complete for all taxonomic features × samples
- Metadata: ~100% complete after removing samples with missing age_group
- Missing values analyzed and documented in preprocessing stage

### Consistency

- Sample IDs consistent between abundance and metadata files
- Taxonomic paths follow standard MetaPhlAn 4 format
- Relative abundances sum to ~100% per sample (as expected)

### Accuracy

- Data sourced from peer-reviewed published study
- MetaPhlAn 4.1.1 is a validated tool for taxonomic profiling
- Quality control steps applied during sequencing and analysis

## Usage Restrictions

### License

Data is used in accordance with the original publication's data availability statement.

### Ethics

- Study approved by relevant ethics committees (see original publication)
- Data has been de-identified for privacy protection

### Acknowledgments

Users of this dataset should:
1. Cite the original LuCKi cohort study publication
2. Acknowledge the data providers
3. Follow ethical guidelines for secondary data analysis

## Reproducibility Information

### Software Versions

- Python: 3.8+
- pandas: >=2.0.0
- NumPy: >=1.24.0, <=2.3.0
- scikit-learn: >=1.3.0
- TensorFlow: >=2.13.0
- XGBoost: >=2.0.0
- LightGBM: >=4.0.0

See `notebooks/requirements.txt` for complete dependency list with version constraints.

### Random Seeds

- Python random: 42
- NumPy random: 42
- TensorFlow random: 42
- Cross-validation: 3004 (in some models)

### Hardware

Analysis can be run on:
- CPU-only systems (slower for neural networks)
- GPU-accelerated systems (CUDA or ROCm)
- Recommended: 8GB+ RAM

### Checksums

Users can verify data integrity by comparing file sizes:
- `MAI3004_lucki_mpa411.txt`: ~30 MB
- `MAI3004_lucki_mpa411.csv`: ~18 MB
- `MAI3004_lucki_metadata_safe.csv`: ~50 KB

## Data Access

### Availability

The specific dataset files are not included in the Git repository due to size and potential licensing restrictions. Users should:

1. Contact the original study authors for data access
2. Follow institutional data sharing policies
3. Place files in `data/raw/` directory after obtaining access

### Data Structure Expected

```
data/
└── raw/
    ├── MAI3004_lucki_mpa411.csv          # Abundance table
    ├── MAI3004_lucki_mpa411.txt          # Alternative format
    ├── MAI3004_lucki_metadata_safe.csv   # Sample metadata
    └── metaphlan411_data_description.md  # Format documentation
```

## Contact and Support

For questions about:
- **Data access**: Refer to original publication authors
- **Analysis pipeline**: Open an issue on GitHub
- **Technical problems**: Check documentation or open an issue

## References

1. Lucki Cohort Study Group. (2015). The LuCKi Living Lab - An Innovative Epidemiological Approach to Study the Gut Microbiome Over the Lifespan. *BMC Public Health*, 15, 1-11. https://doi.org/10.1186/s12889-015-2255-7

2. Blanco-Míguez, A., et al. (2023). Extending and improving metagenomic taxonomic profiling with uncharacterized species using MetaPhlAn 4. *Nature Biotechnology*, 41, 1633-1644. https://doi.org/10.1038/s41587-023-01688-w

---

**Document Version**: 1.0.0  
**Last Updated**: 2026-01-19  
**Maintained by**: MAI-David
