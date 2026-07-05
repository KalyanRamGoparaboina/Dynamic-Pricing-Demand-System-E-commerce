"""
evaluate.py — Generates metric reports and diagnostic plots comparing actual
vs. predicted demand on the test set.

Saves diagnostic plots directly to the reports/figures/ folder.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Allow running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import FEATURE_COLS, FIGURES, TARGET_COL


def load_best_model():
    """Load the persisted model binary."""
    model_path = Path(__file__).resolve().parents[2] / "pricing-engine" / "models_artifacts" / "best_model.pkl"
    if not model_path.exists():
        # Try alternate path
        model_path = Path(__file__).resolve().parents[2] / "models_artifacts" / "best_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Best model not found at {model_path}. Run train first.")
    return joblib.load(model_path)


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, dates: pd.Series, sku: str) -> None:
    """Save actual vs. predicted timeline chart for a specific SKU."""
    df = pd.DataFrame({"Date": dates, "Actual": y_true, "Predicted": np.clip(y_pred, 0, None)})
    df = df.groupby("Date").sum().reset_index()  # Aggregate across weeks if needed
    
    plt.figure(figsize=(10, 5))
    plt.plot(df["Date"], df["Actual"], label="Actual Demand", color="#2b5c8f", marker="o", linewidth=2)
    plt.plot(df["Date"], df["Predicted"], label="Predicted Demand", color="#d95f02", marker="s", linestyle="--", linewidth=2)
    
    plt.title(f"Demand Forecast Verification — SKU: {sku}", fontsize=14, fontweight="bold")
    plt.xlabel("Week Start Date", fontsize=12)
    plt.ylabel("Quantity Demanded", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()
    
    out_path = FIGURES / f"forecast_verification_{sku}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[evaluate] Saved SKU prediction plot to {out_path}")


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Save residual distribution and actual vs predicted scatter plot."""
    y_pred_clipped = np.clip(y_pred, 0, None)
    residuals = y_true - y_pred_clipped
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Scatter Actual vs Predicted
    axes[0].scatter(y_true, y_pred_clipped, alpha=0.4, color="#7570b3", edgecolors="none")
    max_val = max(y_true.max(), y_pred_clipped.max())
    axes[0].plot([0, max_val], [0, max_val], color="red", linestyle="--", label="Perfect Forecast")
    axes[0].set_title("Actual vs. Predicted Demand", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Actual Demand")
    axes[0].set_ylabel("Predicted Demand")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()
    
    # 2. Residual Histogram
    axes[1].hist(residuals, bins=40, color="#1b9e77", edgecolor="white", alpha=0.8)
    axes[1].axvline(0, color="red", linestyle="--", label="Zero Residual")
    axes[1].set_title("Residual Error Distribution", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Residual (Actual − Predicted)")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()
    
    plt.suptitle("Model Diagnostic Analysis (Test Set)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    out_path = FIGURES / "residuals_diagnostic.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[evaluate] Saved model diagnostic plots to {out_path}")


def run() -> None:
    test_parquet = Path(__file__).resolve().parents[2] / "pricing-engine" / "data" / "processed" / "test.parquet"
    if not test_parquet.exists():
        test_parquet = Path(__file__).resolve().parents[2] / "data" / "processed" / "test.parquet"
    if not test_parquet.exists():
        raise FileNotFoundError(f"Test dataset not found at {test_parquet}. Run train first.")
        
    df_test = pd.read_parquet(test_parquet)
    model = load_best_model()
    
    X_test = df_test[FEATURE_COLS]
    y_test = df_test[TARGET_COL]
    
    # Run predictions
    y_pred = model.predict(X_test)
    
    # Generate general diagnostics
    plot_residuals(y_test.values, y_pred)
    
    # Generate per-SKU timelines for the top-performing SKU in test set to verify
    # Pick the SKU with the most transaction records in the test set
    top_test_sku = df_test["sku"].value_counts().index[0]
    df_sku = df_test[df_test["sku"] == top_test_sku].sort_values("weekstart")
    y_sku_true = df_sku[TARGET_COL].values
    y_sku_pred = model.predict(df_sku[FEATURE_COLS])
    
    plot_predictions(y_sku_true, y_sku_pred, df_sku["weekstart"], top_test_sku)


if __name__ == "__main__":
    run()
