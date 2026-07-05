# Dynamic Pricing & Demand Forecasting Engine

A production-grade machine learning system for **dynamic pricing optimization** and **demand forecasting** built on the **UCI Online Retail II dataset** (~540K real UK e-commerce transaction records, 2010–2011).

This system provides:
1. **Data Pipeline**: Outlier winsorization at the SKU level, credit-memo exclusion, UK market filtering, and validation ensuring consistent revenue accounting post-winsorizing.
2. **Lag-Safe Feature panel**: Reindexes products to cover all calendar weeks (Mon-Sun), forward-fills prices, and computes rolling demand features shifted by at least 1 week to eliminate target leakage.
3. **Econometric Elasticity Module**: Fixed-effects panel regression using within-SKU demeaning (Frisch-Waugh style) to estimate causal price elasticity without SKU biases.
4. **TimeSeriesSplit CV Machine Learning**: 5-fold temporal validation of Ridge, Random Forest, XGBoost, and LightGBM models tracked locally via SQLite-backed MLflow.
5. **Explainability & Optimizer**: SHAP global feature importances, local waterfalls, and candidate price grid search simulating revenue curves per SKU.
6. **Programmatic API (FastAPI)**: Schema-validated REST endpoints for programmatic pricing forecasts.
7. **Dashboard (Streamlit)**: 5-page interactive slate-themed corporate interface with Plotly charting.

---

## Folder Structure

```
pricing-engine/
├── data/
│   ├── raw/                    # Downloaded transaction CSV
│   └── processed/              # Cleaned Parquet panel and models
├── src/
│   ├── config.py               # Path constants, modeling parameters
│   ├── data/
│   │   ├── download.py         # Requests downloader with progress reporting
│   │   └── preprocess.py       # Cancellation drops, winsorizing, audit reports
│   ├── features/
│   │   └── build_features.py   # Calendar reindexing, lags (leakage-free)
│   ├── models/
│   │   ├── train.py            # Ridge, RF, XGB, LGBM CV + MLflow sqlite backend
│   │   ├── evaluate.py         # Residual distributions, actual vs pred timelines
│   │   └── explain.py          # SHAP global beeswarm and waterfall explanations
│   ├── pricing/
│   │   ├── elasticity.py       # Demeaned pooled OLS panel & per-SKU regressions
│   │   └── optimizer.py        # Revenue optimization grid simulations
│   └── api/
│       └── app.py              # FastAPI REST endpoints
├── app/
│   └── streamlit_app.py        # 5-page Streamlit Dashboard app
├── tests/                      # Pytest unit tests
├── reports/figures/            # Saved matplotlib metrics figures
├── Dockerfile                  # Production container definitions
├── docker-compose.yml          # Container multi-port orchestration
├── requirements.txt            # Python dependencies
├── Makefile                    # Modular execution commands
└── README.md                   # System documentation
```

---

## Getting Started

### 1. Installation
Install the project as an editable package:
```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Execution Pipeline
You can run the entire pipeline end-to-end using the `Makefile` commands or python commands:

```bash
# 1. Download the raw online retail dataset (~45MB)
python -m src.data.download

# 2. Preprocess & Clean transactions
python -m src.data.preprocess

# 3. Build weekly panel features (leakage-free)
python -m src.features.build_features

# 4. Estimate econometric price elasticities
python -m src.pricing.elasticity

# 5. Train & log models in MLflow
python -m src.models.train

# 6. Generate diagnostic evaluation figures
python -m src.models.evaluate

# 7. Compute SHAP explainability plots
python -m src.models.explain

# 8. Compute optimal pricing grid simulations
python -m src.pricing.optimizer
```

Using the `Makefile`:
```bash
make pipeline
```

---

## Production Services

### FastAPI Programmatic Endpoint
Start the REST API:
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```
Access the interactive documentation:
- Swagger Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Liveness Probe: `GET /health`
- Price Elasticity Lookup: `GET /elasticity/{sku}`
- Optimisation Recommendation: `POST /recommend?sku={sku}`

### Streamlit Business Dashboard
Launch the web interface:
```bash
streamlit run app/streamlit_app.py --server.port 8501
```
Access the dashboard at [http://localhost:8501](http://localhost:8501).

---

## Containerization
Launch both API and Streamlit app inside isolated Docker containers:
```bash
docker-compose up --build
```
- FastAPI: `http://localhost:8000`
- Streamlit: `http://localhost:8501`

---

## Testing
Run the complete unit test suite:
```bash
pytest tests/ -v
```
