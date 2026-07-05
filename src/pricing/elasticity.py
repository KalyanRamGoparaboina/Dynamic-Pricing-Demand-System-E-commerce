"""
elasticity.py — Econometric price elasticity of demand estimation.

Design decisions (senior-level):
  • Fixed-Effects Panel Regression: Instead of running noisy, under-identified
    OLS per individual SKU (which suffers from small sample size and time trends),
    we pool the data using fixed-effects via SKU-level demeaning.
  • Genuine Price Responses: Only rows with total_qty > 0 are used. Zero-demand weeks
    are typically stockouts or store closures rather than voluntary consumer decisions at
    the posted price.
  • Semi-Log/Log-Log Specification: Log-Log model is fit: log(qty) ~ log(price) + is_holiday_week.
    The coefficient on log(price) represents the price elasticity directly.
  • SKU-Level Robustness: We also calculate individual SKU-level elasticities where
    there is sufficient price variation and observation count, applying a p-value filter.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Allow running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import DATA_PROCESSED

ELASTICITIES_CSV = DATA_PROCESSED / "elasticities.csv"


def estimate_pooled_elasticity(df: pd.DataFrame) -> dict:
    """
    Estimate pooled price elasticity using SKU-level demeaning (Fixed-Effects).
    
    Model: ln(Q_it) = α_i + β * ln(P_it) + γ * Holiday_it + ε_it
    We demean ln(Q), ln(P), and Holiday within each SKU to remove α_i,
    then run pooled OLS without intercept.
    """
    # 1. Filter out zero-demand weeks
    sub = df[df["total_qty"] > 0].copy()
    
    # 2. Add log-transformed variables
    sub["ln_qty"] = np.log(sub["total_qty"])
    sub["ln_price"] = np.log(sub["avg_price"])
    
    # 3. Calculate SKU-level means
    sku_means = sub.groupby("sku")[["ln_qty", "ln_price", "is_holiday_week"]].transform("mean")
    
    # 4. Demean variables
    sub["ln_qty_demean"] = sub["ln_qty"] - sku_means["ln_qty"]
    sub["ln_price_demean"] = sub["ln_price"] - sku_means["ln_price"]
    sub["holiday_demean"] = sub["is_holiday_week"] - sku_means["is_holiday_week"]
    
    # 5. OLS Regression without intercept: ln_qty_demean = β * ln_price_demean + γ * holiday_demean
    X = sub[["ln_price_demean", "holiday_demean"]].values
    y = sub["ln_qty_demean"].values
    
    # Solve (X^T X)^-1 X^T y
    coefs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    
    beta_price = coefs[0]
    beta_holiday = coefs[1]
    
    # Calculate standard errors and p-values
    n, k = X.shape
    dof = n - k
    mse = (residuals[0] if len(residuals) > 0 else np.sum((y - X @ coefs) ** 2)) / dof
    var_beta = mse * np.linalg.inv(X.T @ X)
    se_price = np.sqrt(var_beta[0, 0])
    se_holiday = np.sqrt(var_beta[1, 1])
    
    t_price = beta_price / se_price
    p_price = 2 * (1 - stats.t.cdf(abs(t_price), df=dof))
    
    t_holiday = beta_holiday / se_holiday
    p_holiday = 2 * (1 - stats.t.cdf(abs(t_holiday), df=dof))
    
    return {
        "pooled_elasticity": beta_price,
        "price_se": se_price,
        "price_p_value": p_price,
        "holiday_coef": beta_holiday,
        "holiday_p_value": p_holiday,
        "n_observations": n,
    }


def estimate_sku_elasticities(df: pd.DataFrame, min_obs: int = 15, min_price_var: float = 0.05) -> pd.DataFrame:
    """
    Estimate individual elasticity per SKU using log-log regression.
    Filters out noisy SKUs with low price variation or insufficient observations.
    """
    sub = df[df["total_qty"] > 0].copy()
    sub["ln_qty"] = np.log(sub["total_qty"])
    sub["ln_price"] = np.log(sub["avg_price"])
    
    results = []
    
    for sku, group in sub.groupby("sku"):
        if len(group) < min_obs:
            continue
            
        # Check price variation (coefficient of variation of price)
        price_std = group["avg_price"].std()
        price_mean = group["avg_price"].mean()
        if price_mean == 0 or (price_std / price_mean) < min_price_var:
            continue
            
        # Fit regression: ln_qty = alpha + beta * ln_price + gamma * holiday
        # Design matrix with intercept
        X = np.column_stack([np.ones(len(group)), group["ln_price"], group["is_holiday_week"]])
        y = group["ln_qty"].values
        
        try:
            coefs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
            beta_price = coefs[1]
            
            # calculate SEs
            n, k = X.shape
            dof = n - k
            if dof <= 0:
                continue
            mse = (residuals[0] if len(residuals) > 0 else np.sum((y - X @ coefs) ** 2)) / dof
            var_beta = mse * np.linalg.inv(X.T @ X)
            se_price = np.sqrt(var_beta[1, 1])
            
            t_val = beta_price / se_price
            p_val = 2 * (1 - stats.t.cdf(abs(t_val), df=dof))
            
            # Get description
            desc = group["description"].dropna().iloc[0] if "description" in group.columns else ""
            
            results.append({
                "sku": sku,
                "description": desc,
                "elasticity": beta_price,
                "se": se_price,
                "p_value": p_val,
                "n_weeks": len(group),
                "avg_weekly_qty": group["total_qty"].mean(),
                "avg_price": group["avg_price"].mean(),
            })
        except np.linalg.LinAlgError:
            continue
            
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        # Sort by p_value (most statistically significant first)
        res_df = res_df.sort_values("p_value").reset_index(drop=True)
    return res_df


def run(save: bool = True) -> tuple[dict, pd.DataFrame]:
    features_parquet = DATA_PROCESSED / "features_panel.parquet"
    if not features_parquet.exists():
        raise FileNotFoundError(f"Features not found: {features_parquet}. Run build_features first.")
        
    print(f"[elasticity] Loading features panel from {features_parquet}")
    df = pd.read_parquet(features_parquet)
    
    print("[elasticity] Estimating pooled fixed-effects elasticity...")
    pooled_res = estimate_pooled_elasticity(df)
    print(f"  Pooled Elasticity: {pooled_res['pooled_elasticity']:.3f} (p-value: {pooled_res['price_p_value']:.4e})")
    print(f"  Holiday Coefficient: {pooled_res['holiday_coef']:.3f} (p-value: {pooled_res['holiday_p_value']:.4e})")
    print(f"  Total observations: {pooled_res['n_observations']:,}")
    
    print("[elasticity] Estimating SKU-level elasticities...")
    sku_df = estimate_sku_elasticities(df)
    print(f"  Estimated elasticity for {len(sku_df)} eligible SKUs.")
    
    # Significant SKUs report
    sig_skus = sku_df[sku_df["p_value"] < 0.05]
    print(f"  Statistically significant (p < 0.05) SKUs: {len(sig_skus)}")
    if not sig_skus.empty:
        print("\nTop 5 statistically significant elastic SKUs:")
        for _, row in sig_skus.head(5).iterrows():
            print(f"  {row['sku']} ({row['description'][:30]}): Elast={row['elasticity']:.2f}, p={row['p_value']:.4f}")
            
    if save:
        sku_df.to_csv(ELASTICITIES_CSV, index=False)
        print(f"[elasticity] Saved SKU elasticities table -> {ELASTICITIES_CSV}")
        
    return pooled_res, sku_df


if __name__ == "__main__":
    run()
