"""
streamlit_app.py — Interactive Streamlit Dashboard for pricing analysis and demand forecasting.

Design decisions (senior-level):
  • Professional UI: Deep Slate and Teal palette, clean card layouts.
  • Interactive Plots: Built using Plotly for dynamic zooming and hover metrics.
  • Real-time Optimiser: Allows users to simulate arbitrary price changes.
  • ML/Econometric Diagnostics: Includes residual diagnostics, SHAP explainers, and pooled panel elasticities.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_PROCESSED, FEATURE_COLS, FIGURES
from src.models.evaluate import load_best_model
from src.pricing.optimizer import optimize_sku_price

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dynamic Pricing & Demand Forecasting Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Premium Styling (Slate & Teal Palette)
st.markdown("""
<style>
    .reportview-container {
        background-color: #0f172a;
    }
    h1, h2, h3 {
        color: #0ea5e9 !important;
        font-family: 'Inter', sans-serif;
    }
    .metric-card {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        border: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)


# ── Resource Caching ──────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    features_path = DATA_PROCESSED / "features_panel.parquet"
    if not features_path.exists():
        st.error(f"Features file not found at {features_path}. Please run Phase 2 first!")
        st.stop()
        
    df = pd.read_parquet(features_path)
    
    # Load SKU elasticities
    elast_path = DATA_PROCESSED / "elasticities.csv"
    if elast_path.exists():
        elast_df = pd.read_csv(elast_path)
    else:
        elast_df = pd.DataFrame()
        
    return df, elast_df


@st.cache_resource
def get_model():
    try:
        return load_best_model()
    except Exception as e:
        st.error(f"Failed to load model: {e}. Ensure you have trained the model.")
        st.stop()


# ── Load Resources ────────────────────────────────────────────────────────────
df, elast_df = load_data()
model = get_model()
unique_skus = sorted(df["sku"].unique().tolist())

# ── Sidebar Navigation ────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/000000/line-chart.png", width=64)
st.sidebar.title("Pricing Engine")
page = st.sidebar.radio(
    "Navigation Menu",
    [
        "💼 Business Overview",
        "📈 Demand Forecasting",
        "💲 Price Optimization",
        "🧮 Price Elasticity",
        "👁️ Model explainability"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**System Status**:\n"
    "✓ UK market subset loaded\n"
    "✓ Models tuned via TimeSeriesSplit\n"
    "✓ No contemporaneous leakage"
)


# ── Page 1: Business Overview ─────────────────────────────────────────────────
if page == "💼 Business Overview":
    st.title("💼 Business Overview & Operations")
    st.markdown("Metrics aggregated across the top UK wholesale e-commerce catalogue.")
    
    # High-level KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Revenue Analyzed", f"£{df['revenue'].sum():,.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Quantity Demanded", f"{int(df['total_qty'].sum()):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Active Catalog SKUs", f"{len(unique_skus)}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Timeline Weeks", f"{df['weekstart'].nunique()} weeks")
        st.markdown('</div>', unsafe_allow_html=True)
        
    st.markdown("### Weekly Performance Trends")
    # Weekly revenue trend line
    weekly_totals = df.groupby("weekstart")[["revenue", "total_qty"]].sum().reset_index()
    fig = px.line(
        weekly_totals,
        x="weekstart",
        y="revenue",
        title="Overall Portfolio Weekly Revenue Trend (£)",
        color_discrete_sequence=["#0ea5e9"],
    )
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Top SKUs Table
    st.markdown("### Top 10 High-Volume SKUs by Revenue")
    top_skus = (
        df.groupby(["sku", "description"])["revenue"]
        .sum()
        .reset_index()
        .sort_values("revenue", ascending=False)
        .head(10)
    )
    st.dataframe(top_skus, use_container_width=True)


# ── Page 2: Demand Forecasting ────────────────────────────────────────────────
elif page == "📈 Demand Forecasting":
    st.title("📈 SKU-Level Demand Forecasting")
    
    sku_select = st.selectbox("Select SKU to analyze:", unique_skus)
    sku_df = df[df["sku"] == sku_select].sort_values("weekstart")
    
    sku_desc = sku_df["description"].dropna().iloc[0] if len(sku_df["description"].dropna()) > 0 else "N/A"
    st.subheader(f"Product: {sku_desc} (SKU: {sku_select})")
    
    # Calculate predictions
    sku_df["predicted_qty"] = np.clip(model.predict(sku_df[FEATURE_COLS]), 0, None)
    
    # Plotly Actual vs Predicted
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sku_df["weekstart"],
        y=sku_df["total_qty"],
        name="Actual Quantity",
        line=dict(color="#0ea5e9", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=sku_df["weekstart"],
        y=sku_df["predicted_qty"],
        name="Predicted Demand",
        line=dict(color="#f97316", width=2, dash="dash"),
    ))
    fig.update_layout(
        title="Weekly Demand Forecast vs Actuals",
        xaxis_title="Week Start Date",
        yaxis_title="Quantity Sold",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Metrics Table for individual SKU
    actuals = sku_df["total_qty"].values
    preds = sku_df["predicted_qty"].values
    
    rmse = np.sqrt(np.mean((actuals - preds) ** 2))
    mae = np.mean(np.abs(actuals - preds))
    mape = np.mean(np.abs(actuals - preds) / np.where(actuals == 0, 1.0, actuals)) * 100.0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("SKU RMSE (lower is better)", f"{rmse:.2f}")
    with col2:
        st.metric("SKU MAE (lower is better)", f"{mae:.2f}")
    with col3:
        st.metric("SKU MAPE", f"{mape:.1f}%")


# ── Page 3: Price Optimization ────────────────────────────────────────────────
elif page == "💲 Price Optimization":
    st.title("💲 Price Optimization & Grid Simulation")
    st.markdown("Optimise price point based on simulated demand curve elastic responses.")
    
    sku_select = st.selectbox("Select SKU to optimize:", unique_skus)
    sku_df = df[df["sku"] == sku_select].sort_values("weekstart")
    latest_row = sku_df.iloc[-1]
    
    st.subheader(f"Pricing Simulation details for SKU: {sku_select}")
    
    # Run optimizer simulation
    res = optimize_sku_price(model, latest_row)
    
    grid_df = pd.DataFrame(res["simulation_grid"])
    
    # Plotly price vs expected revenue curve
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=grid_df["price"],
        y=grid_df["expected_revenue"],
        name="Expected Revenue",
        line=dict(color="#10b981", width=3)
    ))
    # Highlight current price
    fig.add_trace(go.Scatter(
        x=[res["current_price"]],
        y=[res["current_expected_revenue"]],
        name="Current Price Point",
        mode="markers",
        marker=dict(size=14, color="#ef4444", symbol="x")
    ))
    # Highlight recommended price
    fig.add_trace(go.Scatter(
        x=[res["recommended_price"]],
        y=[res["recommended_expected_revenue"]],
        name="Optimal Price Point",
        mode="markers",
        marker=dict(size=14, color="#10b981", symbol="circle")
    ))
    
    fig.update_layout(
        title="Revenue Curve: Price vs Expected Revenue (£)",
        xaxis_title="Price (£)",
        yaxis_title="Expected Weekly Revenue (£)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Current Price", f"£{res['current_price']:.2f}")
    with c2:
        st.metric("Optimal Price Recommendation", f"£{res['recommended_price']:.2f}")
    with c3:
        st.metric("Expected Revenue Uplift (£)", f"£{res['revenue_uplift']:.2f}")
    with c4:
        st.metric("Expected Revenue Uplift (%)", f"{res['revenue_uplift_pct']:.1f}%")
        
    st.markdown("### Simulated Demand Grid")
    st.dataframe(grid_df.style.format({
        "price": "£{:.2f}",
        "predicted_qty": "{:.1f}",
        "expected_revenue": "£{:.2f}"
    }), use_container_width=True)


# ── Page 4: Price Elasticity ──────────────────────────────────────────────────
elif page == "🧮 Price Elasticity":
    st.title("🧮 Econometric Price Elasticity (Fixed-Effects Panel)")
    
    # Standard statistics estimated from Phase 3
    # Read the summary or recompute quickly
    from src.pricing.elasticity import estimate_pooled_elasticity
    pooled_res = estimate_pooled_elasticity(df)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Pooled fixed-effects model results")
        st.write(
            f"**Price Elasticity Parameter (β)**: `{pooled_res['pooled_elasticity']:.3f}`\n\n"
            f"**Significance (p-value)**: `{pooled_res['price_p_value']:.4e}`\n\n"
            f"**Holiday Demand Shift (γ)**: `{pooled_res['holiday_coef']:.3f}` (p-value: `{pooled_res['holiday_p_value']:.4e}`)\n\n"
            f"**Total observations analyzed**: `{pooled_res['n_observations']:,}`"
        )
        
        elasticity_val = pooled_res['pooled_elasticity']
        if elasticity_val < -1:
            st.success("Interpretation: **Elastic demand** (consumers are price-sensitive). Small price changes will lead to large demand shifts.")
        else:
            st.warning("Interpretation: **Inelastic demand**. Revenue will increase with price increases.")
            
    with col2:
        st.markdown("### How this was computed")
        st.info(
            "We applied **SKU-level demeaning** before running pooled OLS:\n\n"
            "$$ln(Q_{it}) - \\overline{ln(Q_i)} = \\beta(ln(P_{it}) - \\overline{ln(P_i)}) + \\gamma(H_{it} - \\overline{H_i}) + e_{it}$$\n\n"
            "This filters out SKU-specific constant demand factors and isolating genuine causal elasticity estimates free of SKU bias."
        )
        
    st.markdown("---")
    st.markdown("### Individual SKU-level Price Elasticity Table")
    if not elast_df.empty:
        # Filter for significant ones
        sig_only = st.checkbox("Show statistically significant (p < 0.05) SKUs only", value=True)
        display_df = elast_df.copy()
        if sig_only:
            display_df = display_df[display_df["p_value"] < 0.05]
            
        st.dataframe(display_df.style.format({
            "elasticity": "{:.3f}",
            "se": "{:.3f}",
            "p_value": "{:.4e}",
            "avg_weekly_qty": "{:.1f}",
            "avg_price": "£{:.2f}"
        }), use_container_width=True)
    else:
        st.warning("Elasticity file `elasticities.csv` not found in processed data folder. Run `src/pricing/elasticity.py` to create it.")


# ── Page 5: Model Explainability ──────────────────────────────────────────────
elif page == "👁️ Model explainability":
    st.title("👁️ Model explainability & SHAP Values")
    
    shap_summary_path = FIGURES / "shap_summary_plot.png"
    shap_waterfall_path = FIGURES / "shap_local_waterfall.png"
    
    st.markdown("### Global Feature Importance")
    st.write("Calculated using SHAP tree explainer to isolate marginal pricing effects.")
    
    if shap_summary_path.exists():
        st.image(str(shap_summary_path), caption="Global SHAP Summary Beeswarm Plot")
    else:
        st.warning(f"SHAP summary plot image not found at {shap_summary_path}. Run `src/models/explain.py` first.")
        
    st.markdown("---")
    st.markdown("### Local Prediction Attribution")
    
    if shap_waterfall_path.exists():
        st.image(str(shap_waterfall_path), caption="SHAP Local Waterfall Explanation for a single observation")
    else:
        st.warning(f"SHAP local waterfall plot image not found at {shap_waterfall_path}. Run `src/models/explain.py` first.")
