# Flight Delay Prediction - Hybrid Models

Advanced hybrid machine learning pipeline for flight delay prediction using ensemble methods.

## Models Included
- LightGBM
- CatBoost
- TabNet (Deep Learning)
- XGBoost (Meta-learner)

## Features
- Classification ensemble for delay prediction (binary)
- Regression models for delay time prediction
- Stacking meta-models
- Trained models included for immediate use

## Files
- `train.py` - Main training script
- `test.py` - Testing and metrics utilities
- `flight_delay_hybrid_models.joblib` - Saved trained models
- `models/` - Individual model checkpoints

## Usage
```python
python train.py
```

## Requirements
- pandas
- numpy
- scikit-learn
- lightgbm
- catboost
- pytorch-tabnet
- xgboost
- optuna
- shap
- torch
