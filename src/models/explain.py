"""
explain.py — Model interpretability using SHAP.

Computes global feature importances and local explanations, saving SHAP plots
to the reports/figures/ folder.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Segoe UI', 'DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'cm'
import numpy as np
import pandas as pd
import shap

# Allow running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import FEATURE_COLS, FIGURES
from src.models.evaluate import load_best_model


def run() -> None:
    train_parquet = Path(__file__).resolve().parents[2] / "pricing-engine" / "data" / "processed" / "train.parquet"
    if not train_parquet.exists():
        train_parquet = Path(__file__).resolve().parents[2] / "data" / "processed" / "train.parquet"
    if not train_parquet.exists():
        raise FileNotFoundError(f"Training dataset not found. Run train first.")
        
    df_train = pd.read_parquet(train_parquet)
    X_train = df_train[FEATURE_COLS]
    
    model = load_best_model()
    
    print("[explain] Initializing SHAP Explainer...")
    # Use TreeExplainer for GBDT models, Kernel/Linear Explainer for Ridge, otherwise generic
    model_name = type(model).__name__
    
    # We sample 500 records to speed up computation
    X_sample = X_train.sample(min(500, len(X_train)), random_state=42)
    
    if "XGB" in model_name or "LGBM" in model_name or "Forest" in model_name:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    else:
        explainer = shap.Explainer(model.predict, X_sample)
        shap_values = explainer(X_sample).values
        
    # Generate SHAP Summary Plot
    try:
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title(f"SHAP Global Feature Importance ({model_name})", fontsize=14, fontweight="bold")
        plt.tight_layout()
        out_path = FIGURES / "shap_summary_plot.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"[explain] Saved SHAP summary plot to {out_path}")
    except Exception as e:
        print(f"[explain WARNING] Failed to generate SHAP summary plot: {e}")
        plt.close()
        
    # Generate local explanation for a high-demand prediction
    try:
        # Pick a row
        idx = 0
        shap_val_row = shap_values[idx]
        if isinstance(shap_val_row, list):
            shap_val_row = shap_val_row[0]
            
        # Create a clean DataFrame of SHAP values for the features
        df_local = pd.DataFrame({
            "Feature": FEATURE_COLS,
            "SHAP Value": shap_val_row
        })
        df_local["Absolute Value"] = df_local["SHAP Value"].abs()
        df_local = df_local.sort_values("Absolute Value", ascending=True)
        
        # Color red for positive contributions to demand, blue for negative contributions
        colors = ["#f97316" if val >= 0 else "#0ea5e9" for val in df_local["SHAP Value"]]
        
        plt.figure(figsize=(10, 5))
        plt.barh(df_local["Feature"], df_local["SHAP Value"], color=colors, height=0.6)
        plt.axvline(0, color="white", linestyle="--", alpha=0.5)
        plt.title("SHAP Local Feature Attribution", fontsize=12, fontweight="bold")
        plt.xlabel("SHAP Value (Impact on predicted quantity)")
        plt.grid(True, linestyle=":", alpha=0.3)
        plt.tight_layout()
        
        local_path = FIGURES / "shap_local_waterfall.png"
        plt.savefig(local_path, dpi=150)
        plt.close()
        print(f"[explain] Saved local SHAP waterfall plot replacement to {local_path}")
    except Exception as e:
        print(f"[explain WARNING] Failed to generate local SHAP attribution plot: {e}")
        plt.close()


if __name__ == "__main__":
    run()
