# Data Analysis - Microbiome Study

## Overview

This repository contains a data analysis pipeline for studying microbiome data from the LUCKI cohort. The analysis focuses on bacterial abundance patterns across different age groups and samples, using MetaPhlAn 4.1.1 taxonomic profiles.

### Key Features

- **Data Integration**: Merges abundance tables with sample metadata
- **Preprocessing Pipeline**: Handles missing data, outliers, and normalization
- **Exploratory Analysis**: Visualizes sample distributions and bacterial abundance patterns
- **Machine Learning**: Implements Random Forest models for predictive analysis
- **Feature Engineering**: Performs PCA and feature selection on taxonomic data

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook

### Installation

1. Clone the repository:
```bash
git clone https://github.com/MAI-David/Data-analysis.git
cd Data-analysis
```

2. Install required packages:
```bash
cd notebooks
pip install -r requirements.txt
```

### Dependencies

The project requires the following Python packages:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- click
- colorama
- supertree

## Usage

The main analysis is contained in the Jupyter notebook:

```bash
cd notebooks
jupyter notebook data-pipeline.ipynb
```

The notebook is organized into the following sections:
1. **Housekeeping**: Library imports and settings
2. **Data Preprocessing**: Merging, encoding, and cleaning data
3. **Exploratory Data Analysis**: Visualizations and statistical summaries
4. **Model Training**: Machine learning model development

## Data

The analysis uses two main data sources located in `data/raw/`:

- **MAI3004_lucki_mpa411.csv**: MetaPhlAn 4.1.1 abundance table (6903 × 932)
  - Contains taxonomic profiles with relative abundances for each sample
  - Each row represents a taxonomic clade
  - Columns are prefixed with `mpa411_` for sample identifiers

- **MAI3004_lucki_metadata_safe.csv**: Sample metadata (930 × 6)
  - Contains demographic and sample information
  - Includes family ID, sex, age group, and collection details

For more information about the MetaPhlAn data format, see `data/raw/metaphlan411_data_description.md`.

## Project Structure

```
Data-analysis/
├── README.md                 # This file
├── data/
│   └── raw/                  # Raw data files
│       ├── MAI3004_lucki_mpa411.csv
│       ├── MAI3004_lucki_metadata_safe.csv
│       └── metaphlan411_data_description.md
└── notebooks/
    ├── data-pipeline.ipynb   # Main analysis notebook
    └── requirements.txt      # Python dependencies
```

## Documentation

### Workflow Diagram

The following diagram illustrates the complete data processing and analysis workflow:

```mermaid
---
config:
  layout: elk
  look: neo
  theme: default
---
flowchart TB
    n1["Data"] --> n3["Merged DataFrame"]
    n2["Metadata"] --> n3
    n3 --> n4["Preprocessing"] & n6@{ label: "<span style=\"padding-left:\" data-darkreader-inline-color=\"\">Year of Birth<br>Body Product</span>" } & n7["Exploratory Data Analysis"]
    n6 --> n5["Drop unused columns"]
    n4 --> n8["LabelEncoding"]
    n9["Family ID<br>Sex<br>Age group"] --> n8
    n8 --> n10["Missingness check"]
    n11["Rows with NaN Age Group"] --> n12["Drop unknown samples"]
    n10 --> n11 & n13["Outlier check"]
    n13 --> n14["Summary"] & n15["Normalisation check"]
    n15 --> n16["Summary"]
    n7 --> n17["Shape measure"]
    n17 --> n18["Samples per child"]
    n18 --> n19["Samples per age group"]
    n19 --> n20["Bacterial abundance"]
    n20 --> n21["Feature analysis"]

    n1@{ shape: db}
    n2@{ shape: db}
    n6@{ shape: manual-input}
    n5@{ shape: event}
    n9@{ shape: manual-input}
    n11@{ shape: display}
    n14@{ shape: summary}
    n16@{ shape: summary}
     n1:::Aqua
     n2:::Aqua
    classDef Aqua stroke-width:1px, stroke-dasharray:none, stroke:#46EDC8, fill:#DEFFF8, color:#378E7A
    style n1 color:#000000
    style n2 color:#000000
```

### Key Variables and Objects

The following table describes important variables and data structures used throughout the analysis:

| Name                                         | Purpose                                                                                                                                                                                        |
|----------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `data`, `metadata`                           | Raw abundance table and sample metadata loaded from `../data/raw/MAI3004_lucki_mpa411.csv` and `../data/raw/MAI3004_lucki_metadata_safe.csv`; shapes asserted at `(6903, 932)` and `(930, 6)`. |
| `sample_cols`                                | List of abundance columns prefixed with `mpa411_`, used to isolate sample-level measurements.                                                                                                  |
| `sample_abundances`                          | Transposed abundance table keyed by `sample_id`, created from `sample_cols` and `clade_name`.                                                                                                  |
| `metadata_common`                            | Subset of metadata with sample IDs present in `sample_abundances`.                                                                                                                             |
| `merged_samples`                             | Inner merge of `metadata_common` and `sample_abundances`; drops `year_of_birth` and `body_product`.                                                                                            |
| `encoded_samples`                            | Copy of `merged_samples` with `sex` and `family_id` encoded and rows missing `age_group_at_sample` removed.                                                                                    |
| `age_encoder`, `age_groups`                  | `LabelEncoder` fitted on `age_group_at_sample`; `age_groups` maps age group labels to encoded integers.                                                                                        |
| `missing_table`                              | Summary of missing values per column in `encoded_samples`, including percentage of missing data.                                                                                               |
| `numeric_cols`, `outlier_table`              | Numeric column list and corresponding IQR-based outlier bounds/counts.                                                                                                                         |
| `normalized_samples`                         | Copy used for Shapiro-Wilk normality checks across `numeric_cols`.                                                                                                                             |
| `X`, `feature_cols`                          | Feature matrix derived from `merged_samples` after removing metadata columns; drives prevalence and PCA analysis.                                                                              |
| `top_features`, `X_sub`, `X_scaled`, `X_pca` | PCA prep artifacts: top 500 prevalent features, their subset matrix, scaled values, and resulting 2D projection.                                                                               |

## Analysis Workflow

The data pipeline follows these main steps:

1. **Data Loading**: Import raw abundance data and metadata
2. **Data Merging**: Combine abundance profiles with sample metadata
3. **Preprocessing**:
   - Label encoding for categorical variables (family_id, sex, age_group)
   - Handle missing values
   - Detect and analyze outliers using IQR method
   - Check normality assumptions
4. **Train-Test Split**: Separate data before further processing
5. **Normalization**: Apply log transformation to abundance data
6. **Exploratory Analysis**:
   - Analyze sample distributions
   - Examine bacterial abundance patterns
   - Perform PCA for dimensionality reduction
7. **Feature Selection**: Filter features at specific taxonomic levels (e.g., genus)
8. **Model Training**: Train Random Forest models with hyperparameter tuning

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is part of academic research. Please contact the repository owner for licensing information.

## Acknowledgments

- Data from the LUCKI cohort study
- MetaPhlAn 4.1.1 for taxonomic profiling
- Related publication: LUCKI cohort description (2015) - s12889-015-2255-7

## Contact

For questions or collaborations, please open an issue in this repository.
