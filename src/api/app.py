"""
app.py — FastAPI service delivering model predictions and pricing recommendations.

Includes schema-validated POST payloads using Pydantic, health endpoints,
and SKU catalog listing.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Allow running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import DATA_PROCESSED, FEATURE_COLS
from src.models.evaluate import load_best_model
from src.pricing.optimizer import optimize_sku_price

app = FastAPI(
    title="Dynamic Pricing & Demand Forecasting API",
    description="Production-grade ML engine to forecast SKU demand and recommend revenue-maximizing price points.",
    version="1.0.0",
)

# ── Load resources once at startup ───────────────────────────────────────────
try:
    model = load_best_model()
    features_df = pd.read_parquet(DATA_PROCESSED / "features_panel.parquet")
    # Load SKU elasticities table
    elast_path = DATA_PROCESSED / "elasticities.csv"
    elasticities_df = pd.read_csv(elast_path) if elast_path.exists() else pd.DataFrame()
except Exception as e:
    print(f"[API ERROR] Failed to load startup models/data: {e}")
    model = None
    features_df = pd.DataFrame()
    elasticities_df = pd.DataFrame()


# ── Pydantic Request/Response Schemas ────────────────────────────────────────
class ForecastRequest(BaseModel):
    sku: str = Field(..., example="85123A", description="Unique product SKU identifier")
    price: float = Field(..., gt=0, example=2.95, description="Proposed unit price")
    is_holiday_week: bool = Field(default=False, description="Whether the target week falls on a UK bank holiday")


class ForecastResponse(BaseModel):
    sku: str
    proposed_price: float
    predicted_demand: float
    predicted_revenue: float


class RecommendationResponse(BaseModel):
    sku: str
    current_price: float
    recommended_price: float
    current_expected_revenue: float
    recommended_expected_revenue: float
    revenue_uplift: float
    revenue_uplift_pct: float


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    """Liveness & readiness probe."""
    if model is None or features_df.empty:
        raise HTTPException(status_code=503, detail="Models or features panel not loaded properly")
    return {"status": "healthy", "model_loaded": True, "features_loaded": len(features_df)}


@app.get("/skus", response_model=List[str])
def list_skus():
    """List all available SKU codes tracked in the system."""
    if features_df.empty:
        return []
    return sorted(features_df["sku"].unique().tolist())


@app.post("/predict", response_model=ForecastResponse)
def predict_demand(payload: ForecastRequest):
    """Forecast demand and revenue for a given SKU at a proposed price point."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model binary unavailable")
        
    sku_data = features_df[features_df["sku"] == payload.sku]
    if sku_data.empty:
        raise HTTPException(status_code=404, detail=f"SKU {payload.sku} not found in historical record")
        
    # Grab the latest week available for this SKU to retrieve current baseline/lag values
    latest_week = sku_data.loc[sku_data["weekstart"].idxmax()].copy()
    
    # Update current inputs with payload arguments
    latest_week["avg_price"] = payload.price
    latest_week["is_holiday_week"] = float(payload.is_holiday_week)
    
    # Recalculate price relative indicators
    baseline_price = latest_week.get("price_baseline", payload.price)
    latest_week["price_vs_baseline"] = (payload.price / baseline_price) - 1.0 if baseline_price else 0.0
    
    # Cast to DataFrame row
    row_df = pd.DataFrame([latest_week[FEATURE_COLS]])
    
    pred_qty = max(0.0, float(model.predict(row_df)[0]))
    expected_rev = payload.price * pred_qty
    
    return {
        "sku": payload.sku,
        "proposed_price": payload.price,
        "predicted_demand": pred_qty,
        "predicted_revenue": expected_rev,
    }


@app.post("/recommend", response_model=RecommendationResponse)
def get_recommendation(sku: str):
    """Calculate the revenue-maximizing price point for a given SKU based on latest week features."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model binary unavailable")
        
    sku_data = features_df[features_df["sku"] == sku]
    if sku_data.empty:
        raise HTTPException(status_code=404, detail=f"SKU {sku} not found in historical record")
        
    latest_week = sku_data.loc[sku_data["weekstart"].idxmax()]
    res = optimize_sku_price(model, latest_week)
    
    return {
        "sku": sku,
        "current_price": res["current_price"],
        "recommended_price": res["recommended_price"],
        "current_expected_revenue": res["current_expected_revenue"],
        "recommended_expected_revenue": res["recommended_expected_revenue"],
        "revenue_uplift": res["revenue_uplift"],
        "revenue_uplift_pct": res["revenue_uplift_pct"],
    }


@app.get("/elasticity/{sku}")
def get_sku_elasticity(sku: str):
    """Retrieve elasticity value estimated from fixed-effects or individual SKU regressions."""
    if elasticities_df.empty:
        raise HTTPException(status_code=404, detail="Elasticity estimations not available. Run elasticity first.")
        
    match = elasticities_df[elasticities_df["sku"] == sku]
    if match.empty:
        # Check if the SKU is tracked at all
        if sku not in features_df["sku"].unique():
            raise HTTPException(status_code=404, detail=f"SKU {sku} is not tracked in the feature store")
        return {
            "sku": sku,
            "elasticity": "insufficient_data",
            "message": "SKU did not meet statistical eligibility requirements (price variance or count thresholds)"
        }
        
    row = match.iloc[0]
    return {
        "sku": sku,
        "description": row["description"],
        "elasticity": float(row["elasticity"]),
        "p_value": float(row["p_value"]),
        "n_weeks": int(row["n_weeks"]),
        "significance": "statistically_significant" if row["p_value"] < 0.05 else "not_significant",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=API_HOST, port=API_PORT, reload=True)
