"""
test_features.py — Tests to verify lag safety and calendar feature creation.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.config import FEATURE_COLS
from src.features.build_features import add_features


def test_no_leakage_and_shape():
    # Construct small mock panel
    weeks = pd.date_range("2011-01-01", periods=10, freq="W-MON")
    data = []
    for w in weeks:
        data.append({
            "weekstart": w,
            "sku": "85123A",
            "total_qty": 10.0,
            "avg_price": 2.50,
            "revenue": 25.0,
            "n_transactions": 2,
        })
    df_panel = pd.DataFrame(data)
    
    res = add_features(df_panel)
    
    # 1. Assert all expected feature columns are present
    for col in FEATURE_COLS:
        assert col in res.columns
        
    # 2. Verify rolling/lags have no NaNs
    assert not res[FEATURE_COLS].isna().any().any()
    
    # 3. Check lag correctness (e.g. lag_1w should be the shifted target)
    # Row 0 has lag filled with 0.0, row 1 has lag_1w = row 0's total_qty (10.0)
    assert res.loc[1, "qty_lag_1w"] == 10.0
