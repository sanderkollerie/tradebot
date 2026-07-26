.PHONY: install test lint build local-invoke train-bootstrap deploy

install:
	python -m pip install -r requirements-dev.txt

test:
	PYTHONPATH=src pytest -q

lint:
	ruff check src tests

build:
	sam build

local-invoke:
	sam local invoke TradingFunction --event events/test.json

train-bootstrap:
	PYTHONPATH=src python -m trading_bot.train --live-bootstrap --pair XBTEUR --interval 15 --output model/btc-eur-15m.joblib

deploy:
	sam deploy --guided
