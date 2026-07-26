# Kraken Lambda Trader

Dit project is een long/flat BTC/EUR tradingbot voor AWS Lambda. Het model wordt **buiten Lambda** getraind en getest. Lambda doet alleen inference, risicobeheer, orderuitvoering en logging.

## Wat dit project wel doet

- Ensemble van logistic regression, gradient boosting en random forest.
- Probability calibration op een latere tijdsperiode.
- Chronologische train/calibration/test-splits zonder random shuffle, met een purged gap tegen look-ahead leakage.
- Backtest met taker fees, slippage, next-bar execution, stops, positiegrootte, cooldown en equity-circuit-breakers.
- Paper trading met fictief EUR-saldo.
- Live Kraken market orders met verwerking van echte fills.
- Na iedere live koop een exchange-side stop-loss order; hogere trailing stops worden atomair in-place geamendeerd.
- Bij een definitief mislukte stoporder volgt een noodverkoop; bij een onzekere API-response wordt dezelfde client-order-id teruggevonden voordat iets nieuws wordt verstuurd.
- Write-ahead order intents, deterministische client-order-id’s, DynamoDB-idempotentie, reserved concurrency 1 en aparte event/trade logging.
- Live mode accepteert standaard alleen een modelartifact dat de testcriteria heeft gehaald én expliciet met `--approve-live` is gemaakt.

Dit is serieuze infrastructuur, maar geen garantie op winst. Een model kan na deployment zijn statistische voordeel verliezen. De meegeleverde code is automatisch getest en de trainingspipeline is met synthetische data uitgevoerd; echte live-orders zijn niet tegen jouw Kraken-account uitgevoerd.

## Projectstructuur

```text
src/trading_bot/
  backtest.py     event-driven backtester
  config.py       strikte environmentconfiguratie
  execution.py    paper/live execution en stoporders
  features.py     technische features zonder toekomstige data
  handler.py      Lambda-handler
  kraken.py       officiële Kraken Spot REST-koppeling
  model.py        modelartifact en S3-loader
  repository.py   DynamoDB state, locks en logs
  risk.py         volatility-based position sizing
  strategy.py     entry/exitbeslissingen
  train.py        training, calibratie en out-of-sample test
```

## 1. Installeren en testen

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
make test
```

## 2. Snel model voor uitsluitend paper testing

Kraken's live OHLC-endpoint levert slechts de recente candles. Hiermee kun je de volledige techniek testen, maar het artifact wordt nooit voor live goedgekeurd.

```bash
mkdir -p model
PYTHONPATH=src python -m trading_bot.train \
  --live-bootstrap \
  --pair XBTEUR \
  --interval 15 \
  --output model/btc-eur-15m.joblib
```

## 3. Serieuze training met Kraken-historie

Download de officiële historische OHLCVT ZIP van Kraken en pak het 15-minutenbestand voor XBT/EUR uit. De trainer verwacht de Krakenkolommen:

```text
timestamp, open, high, low, close, volume, trades
```

Train daarna:

```bash
PYTHONPATH=src python -m trading_bot.train \
  --csv /pad/naar/XBTEUR_15.csv \
  --pair XBTEUR \
  --interval 15 \
  --output model/btc-eur-15m.joblib
```

De JSON-manifest toont de resultaten op de onaangeraakte testperiode. Voor een live-goedgekeurd artifact:

```bash
PYTHONPATH=src python -m trading_bot.train \
  --csv /pad/naar/XBTEUR_15.csv \
  --pair XBTEUR \
  --interval 15 \
  --output model/btc-eur-15m.joblib \
  --approve-live
```

`approved_for_live` wordt alleen waar wanneer ook de ingebouwde metriekdrempels worden gehaald.

## 4. Model naar S3

```bash
aws s3 mb s3://MIJN-UNIEKE-MODEL-BUCKET --region eu-west-1
aws s3 cp model/btc-eur-15m.joblib s3://MIJN-UNIEKE-MODEL-BUCKET/models/btc-eur-15m.joblib
aws s3 cp model/btc-eur-15m.joblib.json s3://MIJN-UNIEKE-MODEL-BUCKET/models/btc-eur-15m.joblib.json
```

## 5. Paper deployment

Installeer AWS SAM CLI en Docker. Bouw en deploy:

```bash
sam build
sam deploy --guided
```

Gebruik bij de eerste deployment:

```text
TradingMode: paper
ModelBucket: MIJN-UNIEKE-MODEL-BUCKET
ModelKey: models/btc-eur-15m.joblib
LiveTradingEnabled: false
```

In paper mode is geen Kraken-account, API-key of Kraken-saldo nodig. De bot start standaard met EUR 1.000 fictief saldo.

## 6. Kraken API-key voor live mode

Maak in Kraken Pro een API-key met uitsluitend:

- Query Funds
- Query Open Orders & Trades
- Query Closed Orders & Trades
- Create & Modify Orders
- Cancel & Close Orders

Geef de key **geen withdrawal permission**.

Maak daarna het secret:

```bash
aws secretsmanager create-secret \
  --name trading/kraken \
  --secret-string '{"api_key":"JOUW_KEY","api_secret":"JOUW_SECRET"}'
```

Voor live handelen moet op Kraken EUR-saldo staan. `LIVE_CAPITAL_EUR` bepaalt welk maximaal kapitaal de bot als eigen allocatie beschouwt; `MAX_ORDER_EUR`, `RISK_PER_TRADE` en `MAX_POSITION_FRACTION` beperken iedere positie verder. Bij 3% verlies binnen één UTC-dag pauzeert hij tot de volgende dag; bij 15% drawdown vanaf de equitypiek blijft hij gestopt totdat je de state bewust reset.

## 7. Live deployment

Zet pas na paper testing om:

```text
TradingMode: live
LiveTradingEnabled: true
LiveConfirmation: I_UNDERSTAND_LIVE_TRADING
```

Live mode weigert te starten wanneer het model niet `approved_for_live=true` bevat.

## 8. Wat je in de Lambda-response ziet

```json
{
  "status": "SUCCESS",
  "mode": "paper",
  "action": "HOLD",
  "reason": "probability_below_entry_threshold",
  "probability_up": 0.531,
  "model_version": "20260725T220000Z",
  "state": {
    "position": "FLAT",
    "paper_cash": "1000",
    "realized_pnl": "0"
  }
}
```

De `TradingBotEvents`-tabel bevat candle-events en transactielogs. De `TradingBotState`-tabel bevat de actuele positie en cumulatieve resultaten.

## 9. Reset van paper trading

Verwijder uitsluitend het item met jouw `bot_id` uit de state-tabel en verwijder eventueel de bijbehorende event-items. Daarna begint paper trading opnieuw met `STARTING_CASH_EUR`.

## Belangrijke technische beperkingen

- Lambda ontvangt maximaal de meest recente 720 Kraken-candles; het model moet daarom extern zijn getraind.
- Eén testperiode bewijst geen blijvend voordeel. Gebruik meerdere marktcycli en laat paper mode eerst voldoende transacties verzamelen.
- Deze strategie handelt alleen long/flat spot, dus geen leverage en geen shortposities.
- De backtest is pessimistisch wanneer stop en winstniveau binnen dezelfde candle geraakt zouden kunnen zijn.
- Een exchange-side stop blijft actief tussen Lambda-invocations. De bot verhoogt die stop via Kraken AmendOrder, dus zonder eerst te annuleren.
- De code gebruikt echte Kraken orderfills (`vol_exec`, `cost`, `fee`, `price`) om de positie bij te werken.

## Multimarket training (12 CSV files)

Place these files in `data/`: `XBTEUR_15.csv`, `XBTEUR_60.csv`, `XBTEUR_240.csv`, `XBTUSD_15.csv`, `XBTUSD_60.csv`, `XBTUSD_240.csv`, `ETHEUR_15.csv`, `ETHEUR_60.csv`, `ETHEUR_240.csv`, `ETHUSD_15.csv`, `ETHUSD_60.csv`, and `ETHUSD_240.csv`.

Train with:

```powershell
$env:PYTHONPATH="src"
python -m trading_bot.train --data-dir data --pair XBTEUR --interval 15 --output model/btc-eur-multimarket.joblib
```

The model artifact records the context markets it needs. During Lambda execution the same 11 auxiliary series are fetched automatically from Kraken and aligned backward to the current BTC/EUR 15-minute candle.
