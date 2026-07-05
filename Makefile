.PHONY: download preprocess features elasticity train evaluate explain optimize api streamlit test docker-build docker-up

download:
	python -m src.data.download

preprocess:
	python -m src.data.preprocess

features:
	python -m src.features.build_features

elasticity:
	python -m src.pricing.elasticity

train:
	python -m src.models.train

evaluate:
	python -m src.models.evaluate

explain:
	python -m src.models.explain

optimize:
	python -m src.pricing.optimizer

pipeline: download preprocess features elasticity train evaluate explain optimize

api:
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

streamlit:
	streamlit run app/streamlit_app.py --server.port 8501

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

docker-build:
	docker-compose build

docker-up:
	docker-compose up
