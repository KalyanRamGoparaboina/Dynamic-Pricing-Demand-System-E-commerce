"""
build_features.py — Weekly aggregation, temporal reindexing, calendar features,
and lag-safe rolling/lagged statistics.

Design decisions (senior-level):
  • Complete Calendar Reindexing: Reindexes each SKU to cover all calendar weeks
    from the dataset start to end. Missing weeks are filled with 0 quantity,
    representing periods of zero demand (or stockouts).
  • Forward-Filled Prices: For missing weeks, prices are forward-filled (and backward-filled
    if necessary) to represent the last active price set by the retailer.
  • Leakage-Free Design: All lag and rolling features are shifted by at least 1 week
    relative to the target week. Contemporaneous features (such as same-week customer counts)
    are explicitly omitted.
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
    LAG_WEEKS,
    PRICE_BASELINE_WINDOW,
    ROLL_WINDOWS,
    TARGET_COL,
    TOP_N_SKUS,
    UK_HOLIDAYS,
)

FEATURES_PARQUET = DATA_PROCESSED / "features_panel.parquet"


def aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Group transactions to weekly SKU-level panel."""
    # Group by week and SKU
    weekly = (
        df.groupby(["weekstart", "sku"])
        .agg(
            total_qty=("quantity", "sum"),
            revenue=("revenue", "sum"),
            n_transactions=("invoiceno", "nunique"),
            total_price_vol=("revenue", "sum"),  # will divide by quantity to get weighted average price
        )
        .reset_index()
    )
    weekly["avg_price"] = weekly["revenue"] / weekly["total_qty"]
    return weekly


def select_top_skus(df: pd.DataFrame, top_n: int = TOP_N_SKUS) -> list[str]:
    """Identify the top-N SKUs by total historical revenue."""
    top_skus = (
        df.groupby("sku")["revenue"]
        .sum()
        .nlargest(top_n)
        .index.tolist()
    )
    return top_skus


def reindex_calendar(df: pd.DataFrame, top_skus: list[str]) -> pd.DataFrame:
    """Ensure complete Mon-Sun calendar panel for top SKUs."""
    # Filter to top SKUs
    df = df[df["sku"].isin(top_skus)].copy()
    
    # Generate all weeks in dataset range
    all_weeks = pd.date_range(start=df["weekstart"].min(), end=df["weekstart"].max(), freq="W-MON")
    
    # Build MultiIndex
    mux = pd.MultiIndex.from_product([all_weeks, top_skus], names=["weekstart", "sku"])
    
    # Reindex
    panel = df.set_index(["weekstart", "sku"]).reindex(mux).reset_index()
    
    # Fill target variables: missing weeks mean 0 demand and 0 transactions
    panel["total_qty"] = panel["total_qty"].fillna(0.0)
    panel["revenue"] = panel["revenue"].fillna(0.0)
    panel["n_transactions"] = panel["n_transactions"].fillna(0.0)
    
    # Carry forward prices: if no transactions, price is forward-filled (then backward-filled for early weeks)
    panel["avg_price"] = panel.groupby("sku")["avg_price"].ffill().bfill()
    
    return panel


def add_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Build lag-safe demand, pricing signal, and calendar features."""
    panel = panel.sort_values(["sku", "weekstart"]).copy()
    
    # 1. Price vs Baseline (52-week rolling median of price)
    # Shifted by 1 to make it lag-safe
    panel["price_baseline"] = (
        panel.groupby("sku")["avg_price"]
        .transform(lambda x: x.shift(1).rolling(window=PRICE_BASELINE_WINDOW, min_periods=1).median())
    )
    panel["price_vs_baseline"] = (panel["avg_price"] / panel["price_baseline"]) - 1.0
    panel["price_vs_baseline"] = panel["price_vs_baseline"].fillna(0.0)
    
    # 2. Price Rolling Mean
    panel["price_roll_4w"] = (
        panel.groupby("sku")["avg_price"]
        .transform(lambda x: x.shift(1).rolling(window=4, min_periods=1).mean())
    )
    
    # 3. Demand Lags (Shifted from target)
    for lag in LAG_WEEKS:
        panel[f"qty_lag_{lag}w"] = (
            panel.groupby("sku")["total_qty"]
            .transform(lambda x: x.shift(lag))
        )
        
    # 4. Demand Rolling Means (Shifted from target)
    for window in ROLL_WINDOWS:
        panel[f"qty_roll_{window}w"] = (
            panel.groupby("sku")["total_qty"]
            .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
        )
        
    # Fill remaining NaNs in lags/rolling features with 0.0 or group mean
    lag_cols = [c for c in panel.columns if "lag_" in c or "roll_" in c]
    for col in lag_cols:
        panel[col] = panel[col].fillna(0.0)
        
    # 5. Calendar Features
    panel["week_of_year"] = panel["weekstart"].dt.isocalendar().week.astype(float)
    panel["month"] = panel["weekstart"].dt.month.astype(float)
    
    holiday_dates = pd.to_datetime(UK_HOLIDAYS)
    panel["is_holiday_week"] = panel["weekstart"].isin(holiday_dates).astype(float)
    
    return panel


def run(save: bool = True) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    clean_parquet = DATA_PROCESSED / "transactions_clean.parquet"
    if not clean_parquet.exists():
        raise FileNotFoundError(f"Clean transactions not found: {clean_parquet}. Run preprocess first.")
        
    print(f"[build_features] Loading cleaned transactions from {clean_parquet}")
    df = pd.read_parquet(clean_parquet)
    
    print("[build_features] Aggregating weekly panel...")
    weekly = aggregate_weekly(df)
    
    print("[build_features] Selecting top SKUs...")
    top_skus = select_top_skus(weekly, TOP_N_SKUS)
    print(f"[build_features] Selected {len(top_skus)} SKUs by revenue.")
    
    print("[build_features] Reindexing calendar...")
    panel = reindex_calendar(weekly, top_skus)
    
    print("[build_features] Building lag-safe feature set...")
    features_df = add_features(panel)
    
    # Drop rows without sufficient history (e.g. earliest week has lag NaNs filled, but let's keep all and validate)
    assert not features_df[FEATURE_COLS + [TARGET_COL]].isna().any().any(), "NaNs in features or target!"
    
    if save:
        features_df.to_parquet(FEATURES_PARQUET, index=False)
        print(f"[build_features] Saved panel feature store -> {FEATURES_PARQUET} ({len(features_df):,} rows)")
        
    return features_df


if __name__ == "__main__":
    run()
