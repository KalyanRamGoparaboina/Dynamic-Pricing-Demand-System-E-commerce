"""
test_preprocess.py — Tests for the data cleaning / preprocessing routines.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.data.preprocess import clean


@pytest.fixture
def sample_raw_data():
    return pd.DataFrame({
        "InvoiceNo": ["536365", "C536366", "536367", "536368"],
        "StockCode": ["85123A", "85123A", "85123A", "POST"],
        "Description": ["A", "A", "A", "Postage"],
        "Quantity": [6.0, 6.0, 10.0, 1.0],
        "UnitPrice": [2.55, 2.55, 2.55, 15.00],
        "CustomerID": ["17850", "17850", "17850", "17850"],
        "Country": ["United Kingdom", "United Kingdom", "United Kingdom", "United Kingdom"],
        "InvoiceDate": pd.to_datetime(["2010-12-01 08:26:00"] * 4),
    })


def test_clean_pipeline(sample_raw_data):
    df_clean, report = clean(sample_raw_data)
    
    # 1. Cancellations starting with C must be dropped
    assert not any(df_clean["invoiceno"].str.startswith("C"))
    
    # 2. Postage/Service StockCode 'POST' (which does not start with a digit) must be dropped
    assert not any(df_clean["sku"] == "POST")
    
    # 3. Final columns should be lowercased
    assert "sku" in df_clean.columns
    assert "quantity" in df_clean.columns
    assert "unit_price" in df_clean.columns
    assert "revenue" in df_clean.columns
