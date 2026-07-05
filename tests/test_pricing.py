"""
test_pricing.py — Verification of pricing optimizer simulation logic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLS
from src.pricing.optimizer import optimize_sku_price


class MockModel:
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        # Simple linear demand function: Q = 100 - 10 * Price
        # Returns linear response based on avg_price
        prices = df["avg_price"].values
        return 100.0 - 10.0 * prices


def test_optimize_sku_price():
    # Construct a sample row representing the latest week's context
    row_data = {
        "sku": "85123A",
        "description": "Mock Item",
        "avg_price": 4.0,
        "price_baseline": 4.0,
        "price_vs_baseline": 0.0,
        "price_roll_4w": 4.0,
        "qty_lag_1w": 50.0,
        "qty_lag_2w": 50.0,
        "qty_lag_4w": 50.0,
        "qty_roll_4w": 50.0,
        "qty_roll_12w": 50.0,
        "week_of_year": 12.0,
        "month": 3.0,
        "is_holiday_week": 0.0,
    }
    sku_row = pd.Series(row_data)
    
    mock_model = MockModel()
    
    # Run optimizer
    res = optimize_sku_price(mock_model, sku_row)
    
    assert res["sku"] == "85123A"
    assert res["current_price"] == 4.0
    assert "recommended_price" in res
    assert "revenue_uplift" in res
    # Recommended price should be non-empty and greater than 0
    assert res["recommended_price"] > 0
    # Expected demand is non-negative
    assert res["recommended_predicted_qty"] >= 0
