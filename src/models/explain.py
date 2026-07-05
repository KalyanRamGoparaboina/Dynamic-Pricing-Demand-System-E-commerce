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
        plt.figure(figsize=(10, 4))
        # Pick a row
        idx = 0
        if hasattr(explainer, "expected_value"):
            exp_val = explainer.expected_value
            if isinstance(exp_val, (list, np.ndarray)) and len(exp_val) > 1:
                exp_val = exp_val[0]
                
            shap_val_row = shap_values[idx]
            if isinstance(shap_val_row, list):
                shap_val_row = shap_val_row[0]
                
            shap.plots._waterfall.waterfall_legacy(
                exp_val,
                shap_val_row,
                feature_names=FEATURE_COLS,
                max_display=10,
                show=False
            )
            plt.title("SHAP Local Explanation (Waterfall Plot)", fontsize=12, fontweight="bold")
            plt.tight_layout()
            
            local_path = FIGURES / "shap_local_waterfall.png"
            plt.savefig(local_path, dpi=150)
            plt.close()
            print(f"[explain] Saved local SHAP waterfall plot to {local_path}")
    except Exception as e:
        print(f"[explain WARNING] Failed to generate SHAP local waterfall plot: {e}")
        plt.close()


if __name__ == "__main__":
    run()
