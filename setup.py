from setuptools import find_packages, setup

setup(
    name="pricing_engine",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scipy",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "shap",
        "mlflow",
        "streamlit",
        "fastapi",
        "uvicorn",
        "plotly",
        "requests",
        "joblib",
        "pyarrow",
    ],
)
