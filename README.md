# RABTECH Academy – Task 3: Feature Engineering & Preprocessing Pipeline

## Project Overview

This project is part of the RABTECH Academy Artificial Intelligence & Machine Learning program.

The objective of Task 3 is to build a robust and leak-free machine learning preprocessing pipeline using Scikit-Learn.

## Objectives

This project demonstrates:

- Loading a complex tabular dataset
- Separating features and target
- Splitting data into training and testing sets before preprocessing
- Numerical feature preprocessing
- Categorical feature preprocessing
- Using `ColumnTransformer`
- Building a reusable Scikit-Learn `Pipeline`
- Training a Random Forest classifier
- Evaluating model performance
- Performing tree-based feature importance analysis

## Dataset

The project uses the Adult Income tabular dataset.

Dataset characteristics:

- Records: 32,561
- Original columns: 15
- Features used for modelling: 14
- Target variable: `income`

The target is converted into binary values:

- `0` = `<=50K`
- `1` = `>50K`

The dataset contains both numerical and categorical features, making it suitable for demonstrating feature preprocessing.

## Preprocessing Pipeline

The project uses Scikit-Learn's `ColumnTransformer`.

### Numerical Features

Numerical features are processed using:

1. Median imputation
2. StandardScaler

### Categorical Features

Categorical features are processed using:

1. Most-frequent-value imputation
2. One-Hot Encoding

Unknown categories are handled using:

```python
OneHotEncoder(handle_unknown="ignore")
