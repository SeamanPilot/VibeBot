PYTHONPATH=apps/api/src;services/executor/src;services/strategy/src;services/backtest/src;services/training/src;libs/common/src

install:
	python -m pip install -U pip
	python -m pip install -e .[dev]

format:
	python -m black .
	python -m ruff check --fix .

lint:
	python -m ruff check .

typecheck:
	python -m mypy apps services libs tests

test:
	python -m pytest

run-api:
	set PYTHONPATH=$(PYTHONPATH) && python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

run-dashboard:
	set PYTHONPATH=$(PYTHONPATH) && streamlit run apps/dashboard/app.py

generate-sample-data:
	set PYTHONPATH=$(PYTHONPATH) && python -m training.synthetic_data

train:
	set PYTHONPATH=$(PYTHONPATH) && python -m training.train train

backtest:
	set PYTHONPATH=$(PYTHONPATH) && python -m backtest.run

demo:
	$(MAKE) generate-sample-data
	$(MAKE) train
	$(MAKE) backtest
