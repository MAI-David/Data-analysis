# Contributing to Data Analysis Pipeline

Thank you for your interest in contributing to this project! This document provides guidelines and information for contributors.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [How to Contribute](#how-to-contribute)
5. [Coding Standards](#coding-standards)
6. [Testing Guidelines](#testing-guidelines)
7. [Documentation](#documentation)
8. [Pull Request Process](#pull-request-process)
9. [Questions and Support](#questions-and-support)

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of background or experience level.

### Expected Behavior

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Accept constructive criticism gracefully
- Focus on what is best for the community and project
- Show empathy towards other community members

### Unacceptable Behavior

- Harassment, discriminatory comments, or personal attacks
- Trolling or insulting/derogatory comments
- Publishing others' private information without permission
- Any conduct inappropriate in a professional setting

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git version control
- Basic understanding of microbiome data analysis (helpful but not required)
- Familiarity with machine learning concepts (for ML contributions)

### Ways to Contribute

- **Bug Reports**: Found a bug? Report it!
- **Feature Requests**: Have an idea? Share it!
- **Code Contributions**: Fix bugs or implement features
- **Documentation**: Improve or add documentation
- **Examples**: Add usage examples or tutorials
- **Testing**: Add or improve tests

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR-USERNAME/Data-analysis.git
cd Data-analysis
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r notebooks/requirements.txt
```

### 4. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-number-description
```

## How to Contribute

### Reporting Bugs

**Before submitting a bug report:**
- Check existing issues to avoid duplicates
- Verify the bug with the latest version

**When submitting a bug report, include:**
- Python version (`python --version`)
- Operating system and version
- Package versions (`pip list`)
- Minimal code to reproduce the issue
- Expected vs. actual behavior
- Error messages and stack traces
- Data characteristics (if relevant, without sharing private data)

**Bug report template:**
```markdown
**Description**: Brief description of the bug

**To Reproduce**:
1. Step 1
2. Step 2
3. ...

**Expected Behavior**: What should happen

**Actual Behavior**: What actually happens

**Environment**:
- OS: [e.g., Ubuntu 22.04]
- Python: [e.g., 3.9.7]
- Key packages: [e.g., pandas 2.0.0, tensorflow 2.13.0]

**Additional Context**: Any other relevant information
```

### Requesting Features

**Feature request template:**
```markdown
**Feature Description**: Clear description of the proposed feature

**Use Case**: Why is this feature needed? Who benefits?

**Proposed Implementation**: (Optional) Ideas for implementation

**Alternatives Considered**: (Optional) Other approaches you've considered

**Additional Context**: Any other relevant information
```

## Coding Standards

### Python Style Guide

Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines:

- **Indentation**: 4 spaces (no tabs)
- **Line Length**: Maximum 88 characters (Black formatter default)
- **Naming Conventions**:
  - Functions/variables: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_SNAKE_CASE`
  - Private members: `_leading_underscore`

### Type Hints

Use type hints for function signatures:

```python
def process_data(
    data: pd.DataFrame,
    threshold: float = 0.5,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Process input data with given threshold."""
    # implementation
    return processed_data, metrics
```

### Docstrings

Use NumPy-style docstrings:

```python
def example_function(param1, param2):
    """
    Brief description of function.
    
    Longer description providing more details about what the
    function does and how it works.
    
    Parameters
    ----------
    param1 : type
        Description of param1
    param2 : type, optional
        Description of param2 (default: None)
        
    Returns
    -------
    type
        Description of return value
        
    Raises
    ------
    ValueError
        When param1 is invalid
        
    Examples
    --------
    >>> result = example_function(1, 2)
    >>> print(result)
    3
    
    Notes
    -----
    Additional notes, warnings, or important information.
    
    References
    ----------
    .. [1] Smith et al. (2020). Relevant Paper. Journal, 10(2), 123-145.
    """
    # implementation
```

### Code Organization

- **Functions**: One function per task, keep functions focused
- **Comments**: Explain *why*, not *what* (code should be self-explanatory)
- **Imports**: Organize as: standard library, third-party, local modules
- **Constants**: Define at module level
- **Magic Numbers**: Avoid; use named constants

### Example of Good Code

```python
"""Module for data preprocessing utilities."""

from typing import Optional
import pandas as pd
import numpy as np

# Constants
DEFAULT_THRESHOLD = 0.05
MIN_PREVALENCE = 0.1

def filter_low_abundance_features(
    data: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD,
    min_prevalence: float = MIN_PREVALENCE
) -> pd.DataFrame:
    """
    Remove features with low abundance across samples.
    
    Filters out features that appear in fewer than min_prevalence
    proportion of samples or have mean abundance below threshold.
    
    Parameters
    ----------
    data : pd.DataFrame
        Abundance data with features as columns
    threshold : float, optional
        Minimum mean abundance (default: 0.05)
    min_prevalence : float, optional
        Minimum proportion of samples (default: 0.1)
        
    Returns
    -------
    pd.DataFrame
        Filtered data with only abundant features
    """
    prevalence = (data > 0).sum() / len(data)
    mean_abundance = data.mean()
    
    mask = (prevalence >= min_prevalence) & (mean_abundance >= threshold)
    filtered_data = data.loc[:, mask]
    
    return filtered_data
```

## Testing Guidelines

### Test Structure

While this project currently doesn't have a formal test suite, contributors are encouraged to:

1. **Manual Testing**: Test your changes thoroughly
2. **Edge Cases**: Test with unusual inputs
3. **Documentation**: Document testing steps in PR description

### Future Test Framework

If you'd like to add automated tests, consider:

```python
# tests/test_preprocessing.py
import pytest
import pandas as pd
from notebooks.functions import filter_features_by_level

def test_filter_features_genus_level():
    """Test taxonomic filtering at genus level."""
    # Create sample data
    data = pd.DataFrame({
        'k__Bacteria|p__Firmicutes|c__Clostridia': [1.0, 2.0],
        'k__Bacteria|p__Firmicutes|c__Clostridia|o__Clostridiales|f__Ruminococcaceae|g__Faecalibacterium': [3.0, 4.0],
        'k__Bacteria|p__Firmicutes|c__Clostridia|o__Clostridiales|f__Ruminococcaceae|g__Faecalibacterium|s__prausnitzii': [5.0, 6.0]
    })
    
    # Filter to genus level
    result = filter_features_by_level(data, max_level='Genus')
    
    # Should keep class and genus, but not species
    assert 'k__Bacteria|p__Firmicutes|c__Clostridia' in result.columns
    assert 'k__Bacteria|p__Firmicutes|c__Clostridia|o__Clostridiales|f__Ruminococcaceae|g__Faecalibacterium' in result.columns
    assert 'k__Bacteria|p__Firmicutes|c__Clostridia|o__Clostridiales|f__Ruminococcaceae|g__Faecalibacterium|s__prausnitzii' not in result.columns
```

## Documentation

### Code Documentation

- **Functions**: Must have docstrings
- **Classes**: Must have class and method docstrings
- **Modules**: Should have module-level docstrings
- **Complex Logic**: Add inline comments

### Documentation Files

Update relevant documentation when making changes:

- `README.md`: For user-facing changes
- `DATA_PROVENANCE.md`: For data-related changes
- `CONTRIBUTING.md`: For contribution process changes
- Function docstrings: For API changes

### Examples and Tutorials

Consider adding examples for new features:

```python
# examples/feature_selection_example.py
"""
Example: Using Neural Network Feature Selection

This example demonstrates how to use the nn_feature_search function
to select important features from microbiome data.
"""

from notebooks.functions import nn_feature_search, set_global_seeds
import pandas as pd

# Set seeds for reproducibility
set_global_seeds(42)

# Load your data
X_train = pd.read_csv('data/X_train.csv')
X_test = pd.read_csv('data/X_test.csv')
y_train = pd.read_csv('data/y_train.csv').squeeze()

# Run feature selection
result = nn_feature_search(
    X_train=X_train,
    X_test=X_test,
    Y_train=y_train,
    target_range=(50, 250),
    consensus_threshold=0.7
)

print(f"Selected {result.n_features} features")
print(f"Top 10 features: {result.feature_names[:10]}")
```

## Pull Request Process

### Before Submitting

1. **Test**: Verify your changes work as expected
2. **Code Style**: Ensure code follows style guidelines
3. **Documentation**: Update relevant documentation
4. **Commits**: Use clear, descriptive commit messages
5. **Branch**: Ensure your branch is up to date with main

### Commit Messages

Follow conventional commit format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, no logic change)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(models): add LightGBM benchmark function

Implements LightGBM regressor with hyperparameter tuning
using RandomizedSearchCV. Includes feature importance ranking
and performance metrics.

Closes #42
```

### Pull Request Template

```markdown
## Description
Brief description of changes

## Motivation and Context
Why is this change needed? What problem does it solve?

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to change)
- [ ] Documentation update

## Testing
Describe how you tested your changes

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Tested thoroughly
- [ ] Commit messages are clear
```

### Review Process

1. **Automated Checks**: Ensure any CI checks pass
2. **Review**: Wait for maintainer review
3. **Feedback**: Address review comments
4. **Approval**: Once approved, changes will be merged

### After Merge

- Your contribution will be acknowledged in release notes
- Delete your feature branch if no longer needed
- Update your fork's main branch

## Questions and Support

### Getting Help

- **Questions**: Open a GitHub Discussion or issue
- **Bugs**: Open an issue with bug report template
- **Features**: Open an issue with feature request template

### Communication Channels

- **GitHub Issues**: Primary communication channel
- **Pull Request Comments**: For code-specific discussions
- **Commit Comments**: For specific implementation questions

## Additional Resources

### Microbiome Analysis

- [MetaPhlAn Documentation](https://github.com/biobakery/MetaPhlAn)
- [Microbiome Analysis in R/Python](https://doi.org/10.1038/s41592-019-0496-6)

### Machine Learning

- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

### Best Practices

- [FAIR Principles](https://www.go-fair.org/fair-principles/)
- [PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [NumPy Docstring Guide](https://numpydoc.readthedocs.io/)

## Recognition

Contributors will be recognized in:
- GitHub contributors list
- Release notes
- CITATION.cff file (for significant contributions)

Thank you for contributing to making microbiome data analysis more accessible and reproducible!

---

**Version**: 1.0.0  
**Last Updated**: 2026-01-19
