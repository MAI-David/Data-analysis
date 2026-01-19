# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-01-19

### Added - FAIR Principles Implementation

#### Findability Improvements
- **CITATION.cff**: Machine-readable citation file with DOI references
- **DATA_PROVENANCE.md**: Comprehensive data source and processing documentation
- **Version identifiers**: Clear versioning (v1.0.0) throughout documentation
- **Keywords and metadata**: Enhanced README with searchable keywords
- **Repository structure**: Clear, documented directory organization
- **Changelog**: This file for version tracking

#### Accessibility Improvements
- **LICENSE**: MIT License for clear usage terms
- **Installation guide**: Step-by-step setup instructions in README
- **Requirements**: Pinned dependency versions with ranges
- **Documentation**: Comprehensive README with usage examples
- **Examples**: Three complete usage examples with explanations
- **Data access documentation**: Clear instructions for obtaining data
- **Hardware requirements**: Documented CPU/GPU requirements

#### Interoperability Improvements
- **Dependency versioning**: Semantic version ranges in requirements.txt
- **Type hints**: Added to critical functions (set_global_seeds, nn_feature_search, etc.)
- **Module docstring**: Comprehensive module-level documentation
- **Function docstrings**: NumPy-style docstrings with parameters, returns, examples
- **Data format documentation**: MetaPhlAn 4 format specifications
- **Standard formats**: CSV/TSV data, standard Python packages
- **Configuration file**: pyproject.toml for code quality tools

#### Reusability Improvements
- **Comprehensive docstrings**: Detailed documentation for all major functions
- **Usage examples**: Complete working examples in examples/ directory
  - `01_data_loading.py`: Data loading and preprocessing
  - `02_feature_selection.py`: Neural network feature selection
  - `03_model_benchmarking.py`: Model training and comparison
- **CONTRIBUTING.md**: Guidelines for contributors
- **Code standards**: Documented coding style and conventions
- **Reproducibility**: Documented random seeds and environment
- **Modular design**: Reusable functions in functions.py module
- **Example README**: Detailed guide for using examples

### Enhanced

#### Documentation Quality
- **README.md**: Complete rewrite with:
  - Project overview and key features
  - Installation instructions
  - Usage guide and pipeline structure
  - Repository structure documentation
  - Data format specifications
  - Function reference
  - Hardware considerations
  - Citation information
  - Contributing guidelines
  
- **Function documentation**: Added comprehensive docstrings to:
  - `set_global_seeds()`: Reproducibility helper
  - `GatekeeperLayer`: Custom Keras layer for feature selection
  - `nn_feature_search()`: Neural network stability selection
  - `xgboost_benchmark()`: XGBoost model training
  - `get_taxonomic_level()`: Taxonomic parsing utility
  - `explain_with_lime()`: LIME explainability

#### Code Quality
- **Type annotations**: Added type hints for better IDE support
- **Code formatting**: Configured Black, isort, pylint, mypy
- **Standards compliance**: PEP 8 and PEP 257 docstring conventions
- **Import organization**: Structured imports (stdlib, third-party, local)

### Fixed
- **Dependency conflicts**: Resolved numpy version constraint (<=2.3.0)
- **Documentation gaps**: Filled missing context and explanations

### Repository Structure

```
Data-analysis/
├── README.md                    # Main documentation
├── LICENSE                      # MIT License
├── CITATION.cff                # Citation metadata
├── CONTRIBUTING.md             # Contribution guidelines
├── DATA_PROVENANCE.md          # Data documentation
├── CHANGELOG.md                # This file
├── pyproject.toml              # Tool configuration
├── .gitignore                  # Git exclusions
├── data/
│   └── raw/                    # Raw data (not in repo)
├── notebooks/
│   ├── requirements.txt        # Python dependencies
│   ├── functions.py            # Reusable functions
│   └── data-pipeline.ipynb     # Main analysis notebook
└── examples/
    ├── README.md               # Examples guide
    ├── 01_data_loading.py      # Data loading example
    ├── 02_feature_selection.py # Feature selection example
    └── 03_model_benchmarking.py # Model comparison example
```

## [Unreleased]

### Planned Features
- Automated test suite with pytest
- Continuous integration (GitHub Actions)
- Additional visualization examples
- Docker containerization
- Conda environment specification
- API documentation with Sphinx
- Performance benchmarks
- Additional model types (e.g., deep learning)

### Known Limitations
- No automated tests (manual testing only)
- Examples require significant computation time
- Data files not included in repository
- GPU support requires manual CUDA/ROCm setup

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use this software, please cite:

```bibtex
@software{mai_david_data_analysis,
  author = {MAI-David},
  title = {Data Analysis Pipeline for Microbial Community Profiling},
  year = {2026},
  url = {https://github.com/MAI-David/Data-analysis},
  version = {1.0.0}
}
```

And cite the original data source:
```bibtex
@article{lucki2015,
  title={The LuCKi Living Lab - An Innovative Epidemiological Approach to Study the Gut Microbiome Over the Lifespan},
  journal={BMC Public Health},
  year={2015},
  doi={10.1186/s12889-015-2255-7}
}
```

## Links

- **Repository**: https://github.com/MAI-David/Data-analysis
- **Issues**: https://github.com/MAI-David/Data-analysis/issues
- **Original Study**: https://doi.org/10.1186/s12889-015-2255-7

---

[1.0.0]: https://github.com/MAI-David/Data-analysis/releases/tag/v1.0.0
