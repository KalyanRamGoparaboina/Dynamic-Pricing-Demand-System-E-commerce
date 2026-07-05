"""
config.py — Central configuration for the Dynamic Pricing Engine.

All paths, constants, and hyperparameter defaults live here so that
nothing is hard-coded across modules.
"""
from __future__ import annotations

import os
from pathlib import Path

# ── Project root (two levels up from this file: src/config.py → root) ──────
ROOT = Path(__file__).resolve().parent.parent

# ── Directory layout ─────────────────────────────────────────────────────────
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
MODELS_DIR = ROOT / "models_artifacts"  # saved model objects
MLRUNS_DIR = ROOT / "mlruns"

# Create dirs at import time so nothing breaks on first run
for _d in [DATA_RAW, DATA_PROCESSED, FIGURES, MODELS_DIR, MLRUNS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Raw data ─────────────────────────────────────────────────────────────────
RAW_CSV = DATA_RAW / "online_retail.csv"

# Dataset URL (Databricks Spark Guide → UCI Online Retail)
DATASET_URL = (
    "https://raw.githubusercontent.com/databricks/Spark-The-Definitive-Guide"
    "/master/data/retail-data/all/online-retail-dataset.csv"
)

# ── Preprocessing constants ───────────────────────────────────────────────────
TARGET_COUNTRY = "United Kingdom"
WINSOR_LOW = 0.01   # 1st percentile per SKU
WINSOR_HIGH = 0.99  # 99th percentile per SKU

# ── Feature engineering ───────────────────────────────────────────────────────
TOP_N_SKUS = 200          # Keep top-N SKUs by total revenue
LAG_WEEKS = [1, 2, 4]     # Demand lag features (weeks)
ROLL_WINDOWS = [4, 12]    # Rolling mean windows (weeks)
PRICE_BASELINE_WINDOW = 52  # Weeks for price baseline rolling median

# UK public holidays relevant to the dataset period (2010-12-01 → 2011-12-09)
UK_HOLIDAYS = [
    "2010-12-25", "2010-12-26", "2010-12-27",
    "2011-01-01", "2011-01-03",
    "2011-04-22", "2011-04-25",
    "2011-05-02", "2011-05-30",
    "2011-08-29",
    "2011-12-25", "2011-12-26",
]

# ── Modelling ─────────────────────────────────────────────────────────────────
TARGET_COL = "total_qty"
FEATURE_COLS: list[str] = [
    "avg_price",
    "price_vs_baseline",
    "price_roll_4w",
    "qty_lag_1w",
    "qty_lag_2w",
    "qty_lag_4w",
    "qty_roll_4w",
    "qty_roll_12w",
    "week_of_year",
    "month",
    "is_holiday_week",
]

TEST_FRAC = 0.20       # hold-out test fraction (most-recent weeks)
CV_SPLITS = 5          # TimeSeriesSplit folds
RANDOM_STATE = 42

# ── Pricing optimizer ─────────────────────────────────────────────────────────
PRICE_GRID_POINTS = 21           # candidate prices per SKU
PRICE_RANGE = (0.70, 1.30)       # ±30 % of current price

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_EXPERIMENT = "dynamic_pricing"
MLFLOW_TRACKING_URI = "sqlite:///" + str(DATA_PROCESSED / "mlflow.db").replace("\\", "/")

# ── API ───────────────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
