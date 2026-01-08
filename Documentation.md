# Documentation

## UML activity diagram

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

## Important variables and objects

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
