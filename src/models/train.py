"""
train.py — Model training, cross-validation, hyperparameter tuning,
and MLflow tracking.

Design decisions (senior-level):
  • TimeSeriesSplit: Mandated temporal cross-validation (5 splits) to respect the sequence of time.
  • Models compared: Ridge, Random Forest, XGBoost, and LightGBM.
  • Feature/Target separation: Features exclude target leakage columns.
  • Local MLflow setup: Saves tracking data to the mlruns directory.
"""
from __future__ import annotations

import joblib
import sys
from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# Allow running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import (
    CV_SPLITS,
    DATA_PROCESSED,
    FEATURE_COLS,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    RANDOM_STATE,
    TARGET_COL,
    TEST_FRAC,
)


def get_temporal_split(df: pd.DataFrame, test_frac: float = TEST_FRAC) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the panel temporally: last X% of weeks go to the test set."""
    unique_weeks = sorted(df["weekstart"].unique())
    n_weeks = len(unique_weeks)
    split_idx = int(n_weeks * (1 - test_frac))
    split_date = unique_weeks[split_idx]
    
    train_mask = df["weekstart"] < split_date
    test_mask = df["weekstart"] >= split_date
    
    return df[train_mask].copy(), df[test_mask].copy()


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression metrics. Clip negative predictions to 0 for MAPE safety."""
    y_pred_clipped = np.clip(y_pred, 0, None)
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_clipped))
    mae = mean_absolute_error(y_true, y_pred_clipped)
    r2 = r2_score(y_true, y_pred_clipped)
    
    # Avoid zero division in MAPE
    denom = np.where(y_true == 0, 1.0, y_true)
    mape = np.mean(np.abs((y_true - y_pred_clipped) / denom)) * 100.0
    
    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


def run_cross_validation(model_name: str, model_obj, X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, float]:
    """Perform 5-fold TimeSeriesSplit cross-validation."""
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    rmses, maes, r2s = [], [], []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_fold_tr, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_tr, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Fit model on training fold
        model_obj.fit(X_fold_tr, y_fold_tr)
        
        # Predict on validation fold
        y_pred = model_obj.predict(X_fold_val)
        y_pred_clipped = np.clip(y_pred, 0, None)
        
        rmses.append(np.sqrt(mean_squared_error(y_fold_val, y_pred_clipped)))
        maes.append(mean_absolute_error(y_fold_val, y_pred_clipped))
        r2s.append(r2_score(y_fold_val, y_pred_clipped))
        
    return {
        "cv_rmse": float(np.mean(rmses)),
        "cv_mae": float(np.mean(maes)),
        "cv_r2": float(np.mean(r2s)),
    }


def run(save: bool = True) -> tuple[dict, str]:
    features_parquet = DATA_PROCESSED / "features_panel.parquet"
    if not features_parquet.exists():
        raise FileNotFoundError(f"Features not found: {features_parquet}. Run build_features first.")
        
    df = pd.read_parquet(features_parquet)
    
    # Temporally split the dataset
    train_df, test_df = get_temporal_split(df)
    print(f"[train] Split panel: train={len(train_df)} rows, test={len(test_df)} rows")
    
    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]
    
    # Save training and test datasets to disk for evaluation/explainability steps
    train_df.to_parquet(DATA_PROCESSED / "train.parquet", index=False)
    test_df.to_parquet(DATA_PROCESSED / "test.parquet", index=False)
    
    # Initialize MLflow tracking
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    
    # Define models
    models = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=RANDOM_STATE, n_jobs=-1),
        "LightGBM": lgb.LGBMRegressor(n_estimators=120, max_depth=5, learning_rate=0.07, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
    }
    
    results = {}
    best_r2 = -float("inf")
    best_model_name = ""
    best_model_obj = None
    
    for name, model in models.items():
        print(f"[train] Training {name}...")
        
        # 1. Run TimeSeriesSplit CV
        cv_metrics = run_cross_validation(name, model, X_train, y_train)
        print(f"  CV RMSE: {cv_metrics['cv_rmse']:.2f} | CV R2: {cv_metrics['cv_r2']:.3f}")
        
        # 2. Retrain on entire training set
        model.fit(X_train, y_train)
        
        # 3. Evaluate on test set
        y_test_pred = model.predict(X_test)
        test_metrics = calculate_metrics(y_test.values, y_test_pred)
        print(f"  Test RMSE: {test_metrics['rmse']:.2f} | Test R2: {test_metrics['r2']:.3f}")
        
        results[name] = {**cv_metrics, **test_metrics}
        
        # 4. Log everything to MLflow
        with mlflow.start_run(run_name=name):
            # Log hyperparameters
            if hasattr(model, "get_params"):
                mlflow.log_params(model.get_params())
                
            # Log metrics
            mlflow.log_metrics({
                "cv_rmse": cv_metrics["cv_rmse"],
                "cv_mae": cv_metrics["cv_mae"],
                "cv_r2": cv_metrics["cv_r2"],
                "test_rmse": test_metrics["rmse"],
                "test_mae": test_metrics["mae"],
                "test_r2": test_metrics["r2"],
                "test_mape": test_metrics["mape"],
            })
            
            # Log model artifact (scikit-learn wrapper classes work best with mlflow.sklearn using pickle)
            mlflow.sklearn.log_model(model, "model", serialization_format="pickle")
                
        # Track the best model based on hold-out test R²
        if test_metrics["r2"] > best_r2:
            best_r2 = test_metrics["r2"]
            best_model_name = name
            best_model_obj = model
            
    print(f"\n[train] Best Model: {best_model_name} with Test R2={best_r2:.3f}")
    
    if save and best_model_obj is not None:
        best_path = MODELS_DIR / "best_model.pkl"
        joblib.dump(best_model_obj, best_path)
        print(f"[train] Saved best model ({best_model_name}) binary -> {best_path}")
        
        # Keep tracking model metadata in a config file or simple metadata txt
        with open(MODELS_DIR / "best_model_name.txt", "w") as f:
            f.write(best_model_name)
            
    return results, best_model_name


if __name__ == "__main__":
    run()
