"""
preprocess.py — Cleans the raw UCI Online Retail CSV into a validated,
analysis-ready Parquet file.

Design decisions (senior-level):
  • Revenue is computed AFTER winsorizing so stored revenue is consistent
    with the stored (capped) quantity — avoids silent accounting bugs.
  • Cancellations (InvoiceNo starting with 'C') are dropped; they are
    credit memos, not demand signals.
  • Service / postage items (zero-digit StockCodes) are excluded.
  • Winsorize at SKU level, not globally, to preserve inter-SKU differences.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as a script: `python -m src.data.preprocess`
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import (
    DATA_PROCESSED,
    RAW_CSV,
    TARGET_COUNTRY,
    WINSOR_HIGH,
    WINSOR_LOW,
)


# ── Output path ───────────────────────────────────────────────────────────────
CLEAN_PARQUET = DATA_PROCESSED / "transactions_clean.parquet"


def load_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    """Load raw CSV with explicit dtypes to avoid mixed-type warnings."""
    df = pd.read_csv(
        path,
        dtype={
            "InvoiceNo": str,
            "StockCode": str,
            "Description": str,
            "Quantity": float,
            "UnitPrice": float,
            "CustomerID": str,
            "Country": str,
        },
        parse_dates=["InvoiceDate"],
        dayfirst=False,
    )
    return df


def _winsorize_col(series: pd.Series, low: float, high: float) -> pd.Series:
    """Clip a numeric series to [low, high] quantiles."""
    lo_val = series.quantile(low)
    hi_val = series.quantile(high)
    return series.clip(lower=lo_val, upper=hi_val)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Apply the full cleaning pipeline.

    Returns
    -------
    clean_df : pd.DataFrame
        Cleaned transaction table.
    report : dict
        Step-by-step row-count audit trail.
    """
    report: dict[str, int] = {"raw": len(df)}

    # 1. Drop rows with missing mandatory fields
    df = df.dropna(subset=["Quantity", "UnitPrice", "InvoiceDate", "StockCode"])
    report["after_drop_na"] = len(df)

    # 2. Remove cancellations (credit memos start with 'C')
    df = df[~df["InvoiceNo"].str.startswith("C", na=False)]
    report["after_drop_cancellations"] = len(df)

    # 3. Remove non-positive quantities and prices (data quality)
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    report["after_positive_filter"] = len(df)

    # 4. Filter to target country for a coherent market
    df = df[df["Country"] == TARGET_COUNTRY]
    report["after_country_filter"] = len(df)

    # 5. Remove service / postage / adjustment items (non-product StockCodes)
    #    These are codes that are entirely non-numeric (e.g. 'POST', 'DOT', 'BANK')
    df = df[df["StockCode"].str.match(r"^[0-9]", na=False)]
    report["after_service_item_filter"] = len(df)

    # 6. Winsorize Quantity per SKU (not globally) — BEFORE computing revenue
    df = df.copy()
    df["Quantity"] = (
        df.groupby("StockCode")["Quantity"]
        .transform(lambda s: _winsorize_col(s, WINSOR_LOW, WINSOR_HIGH))
    )
    report["after_winsorize"] = len(df)  # count unchanged; values are capped

    # 7. Compute revenue AFTER winsorizing (critical: consistent accounting)
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    # 8. Add temporal columns for downstream aggregation
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["WeekStart"] = df["InvoiceDate"].dt.to_period("W").apply(lambda p: p.start_time)
    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month

    # 9. Clean up and standardise column names
    df = df.rename(columns={
        "StockCode": "sku",
        "Description": "description",
        "Quantity": "quantity",
        "UnitPrice": "unit_price",
        "CustomerID": "customer_id",
        "InvoiceDate": "invoice_date",
        "Revenue": "revenue",
    })
    df.columns = [c.lower() for c in df.columns]

    report["final"] = len(df)
    return df.reset_index(drop=True), report


def print_report(report: dict) -> None:
    print("\n================== Cleaning Report ==================")
    prev = None
    for step, n in report.items():
        if prev is not None:
            dropped = prev - n
            tag = f"  (-{dropped:,})" if dropped else ""
        else:
            tag = ""
        print(f"  {step:<35s} {n:>8,}{tag}")
        prev = n
    pct_kept = report["final"] / report["raw"] * 100
    print(f"\n  Kept {pct_kept:.1f}% of raw rows")
    print("=====================================================\n")


def run(save: bool = True) -> pd.DataFrame:
    """Full preprocessing pipeline. Returns clean DataFrame."""
    print(f"[preprocess] Loading raw data from {RAW_CSV}")
    df_raw = load_raw()
    print(f"[preprocess] Raw shape: {df_raw.shape}")

    df_clean, report = clean(df_raw)
    print_report(report)

    # Basic schema validation
    assert df_clean["quantity"].min() > 0, "Non-positive quantities remain"
    assert df_clean["unit_price"].min() > 0, "Non-positive prices remain"
    assert not df_clean["revenue"].isna().any(), "NaN revenues"
    assert (df_clean["revenue"] == df_clean["quantity"] * df_clean["unit_price"]).all(), \
        "Revenue ≠ quantity × price — check winsorize order"

    if save:
        df_clean.to_parquet(CLEAN_PARQUET, index=False)
        print(f"[preprocess] Saved -> {CLEAN_PARQUET}  ({len(df_clean):,} rows)")

    return df_clean


if __name__ == "__main__":
    run()
