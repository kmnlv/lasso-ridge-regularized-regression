# Lasso and Ridge Regression for Diamond Price Prediction

> Course: Programming for Descriptive and Predictive Analytics · Semester 1

## Overview

Theory and implementation of L1/L2 regularization. The solved notebook explains the geometric difference between Lasso and Ridge, then builds comparable scikit-learn pipelines on the Seaborn `diamonds` dataset. A second, instruction-only notebook contains the guided practice on the `insurance.csv` dataset.

## Repository structure

- `notebooks/lasso-ridge-diamonds.ipynb` - theory plus solved implementation with outputs.
- `exercises/lasso-ridge-practice.ipynb` - guided assignment handout (instructions only).
- `data/insurance.csv` - 1,338 records used in the practice assignment (`charges` as target).

## Methods

- `ColumnTransformer` combining `StandardScaler` for numeric predictors and `OneHotEncoder(drop='first')` for `cut`, `color` and `clarity`.
- `Lasso(alpha=0.1)` and `Ridge(alpha=0.1)` with identical preprocessing.
- Coefficient magnitude and sparsity comparison (number of non-zero coefficients).
- MSE, RMSE, MAE and R-squared; residual histogram and fitted-versus-observed plots.
- Discussion of heteroscedasticity in the residuals.

## Results

Both regularized models reach **R-squared ≈ 0.9198** on the diamond price task. The residuals show clear heteroscedasticity, so the notebook explicitly separates predictive accuracy from validity of the linear-model assumptions.

## Requirements

`numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn` and `jupyter`. The Seaborn dataset is downloaded by the library.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/lasso-ridge-diamonds.ipynb
```

> Note: `exercises/lasso-ridge-practice.ipynb` is the original assignment handout and is intentionally left unresolved.

---
Author: Karla Maria Monroy Alvarez
