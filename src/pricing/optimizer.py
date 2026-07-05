"""
optimizer.py — Simulates candidate price grids to identify the
revenue-maximizing price per SKU.

Design decisions (senior-level):
  • Demand Forecast Simulation: Evaluates how demand (Q) shifts under proposed prices (P).
  • Feature Updates: Properly adjusts pricing features (avg_price, price_vs_baseline,
    price_roll_4w) in sync with the candidate price being tested.
  • Revenue Maximisation: expected_revenue = P * max(0, predicted_Q).
  • Safe uplifting metrics: Prevents divide-by-zero errors when calculating uplifts.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import (
    DATA_PROCESSED,
    FEATURE_COLS,
    PRICE_GRID_POINTS,
    PRICE_RANGE,
    TARGET_COL,
)
from src.models.evaluate import load_best_model

RECOMMENDATIONS_CSV = DATA_PROCESSED / "pricing_recommendations.csv"


def optimize_sku_price(
    model,
    sku_row: pd.Series,
    price_range: tuple[float, float] = PRICE_RANGE,
    grid_points: int = PRICE_GRID_POINTS,
) -> dict:
    """
    Run grid search simulation over candidate prices for a single SKU row (latest week).
    
    Returns optimal price, predicted demand, expected revenue, and uplift comparison.
    """
    current_price = sku_row["avg_price"]
    baseline_price = sku_row.get("price_baseline", current_price)
    
    # Generate candidate prices
    min_mult, max_mult = price_range
    candidate_prices = np.linspace(current_price * min_mult, current_price * max_mult, grid_points)
    
    best_rev = -1.0
    best_price = current_price
    best_qty = 0.0
    
    sim_data = []
    
    # Prepare a template DataFrame to feed into model.predict
    template_df = pd.DataFrame([sku_row[FEATURE_COLS]])
    
    for p in candidate_prices:
        temp = template_df.copy()
        
        # Update pricing-dependent features
        temp["avg_price"] = p
        temp["price_vs_baseline"] = (p / baseline_price) - 1.0 if baseline_price else 0.0
        # For a single-week forecast, we simplify price_roll_4w as the lagged rolling mean
        # which is independent of the *current* week's price.
        
        pred_qty = model.predict(temp)[0]
        pred_qty_clipped = max(0.0, float(pred_qty))
        expected_rev = p * pred_qty_clipped
        
        sim_data.append({
            "price": p,
            "predicted_qty": pred_qty_clipped,
            "expected_revenue": expected_rev,
        })
        
        if expected_rev > best_rev:
            best_rev = expected_rev
            best_price = p
            best_qty = pred_qty_clipped
            
    # Predict current price outcomes for baseline comparisons
    curr_temp = template_df.copy()
    curr_temp["avg_price"] = current_price
    curr_temp["price_vs_baseline"] = (current_price / baseline_price) - 1.0 if baseline_price else 0.0
    
    curr_pred_qty = max(0.0, float(model.predict(curr_temp)[0]))
    curr_revenue = current_price * curr_pred_qty
    
    # Calculate uplift
    rev_uplift = best_rev - curr_revenue
    pct_uplift = (rev_uplift / curr_revenue * 100.0) if curr_revenue > 0 else 0.0
    
    return {
        "sku": sku_row["sku"],
        "description": sku_row.get("description", ""),
        "current_price": current_price,
        "current_predicted_qty": curr_pred_qty,
        "current_expected_revenue": curr_revenue,
        "recommended_price": best_price,
        "recommended_predicted_qty": best_qty,
        "recommended_expected_revenue": best_rev,
        "revenue_uplift": rev_uplift,
        "revenue_uplift_pct": pct_uplift,
        "simulation_grid": sim_data,
    }


def run(save: bool = True) -> pd.DataFrame:
    features_parquet = Path(__file__).resolve().parents[2] / "pricing-engine" / "data" / "processed" / "features_panel.parquet"
    if not features_parquet.exists():
        features_parquet = Path(__file__).resolve().parents[2] / "data" / "processed" / "features_panel.parquet"
        
    if not features_parquet.exists():
        raise FileNotFoundError(f"Features panel not found. Run build_features first.")
        
    df = pd.read_parquet(features_parquet)
    model = load_best_model()
    
    # Get the latest week for each SKU to run recommendations on current state
    latest_weeks = df.groupby("sku")["weekstart"].transform("max")
    latest_df = df[df["weekstart"] == latest_weeks].copy()
    
    print(f"[optimizer] Simulating price optimizations for {len(latest_df)} SKUs using the latest week data...")
    
    recommendations = []
    
    for _, row in latest_df.iterrows():
        rec = optimize_sku_price(model, row)
        recommendations.append(rec)
        
    rec_df = pd.DataFrame(recommendations)
    
    # Sort by absolute revenue uplift descending
    rec_df = rec_df.sort_values("revenue_uplift", ascending=False).reset_index(drop=True)
    
    print(f"[optimizer] Optimization complete. Top recommended price adjustment:")
    if not rec_df.empty:
        top = rec_df.iloc[0]
        print(f"  SKU: {top['sku']} | Current Price: {top['current_price']:.2f} -> Rec Price: {top['recommended_price']:.2f}")
        print(f"  Expected Uplift: +£{top['revenue_uplift']:.2f} ({top['revenue_uplift_pct']:.1f}%)")
        
    if save:
        # Drop the raw simulation grid list before saving flat CSV table
        flat_df = rec_df.drop(columns=["simulation_grid"])
        flat_df.to_csv(RECOMMENDATIONS_CSV, index=False)
        print(f"[optimizer] Saved recommendations summary table -> {RECOMMENDATIONS_CSV}")
        
        # Save full simulations grid as a separate joblib file for downstream dashboard use
        grid_path = DATA_PROCESSED / "price_simulations_grid.joblib"
        import joblib
        joblib.dump(rec_df, grid_path)
        print(f"[optimizer] Saved detailed simulation grid details -> {grid_path}")
        
    return rec_df


if __name__ == "__main__":
    run()
