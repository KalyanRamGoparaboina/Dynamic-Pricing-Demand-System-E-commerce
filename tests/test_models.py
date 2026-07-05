"""
test_models.py — Basic sanity checks on model prediction shapes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.models.train import calculate_metrics


def test_calculate_metrics():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([11.0, 19.0, 31.0])
    
    metrics = calculate_metrics(y_true, y_pred)
    
    assert "rmse" in metrics
    assert "r2" in metrics
    assert metrics["r2"] > 0.0
