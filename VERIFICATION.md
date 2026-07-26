# Verification

Uitgevoerd op 25 juli 2026:

- `python -m compileall`: geslaagd.
- `PYTHONPATH=src pytest -q`: 14 tests geslaagd.
- Trainingspipeline + modelserialisatie + manifest + out-of-sample backtest: geslaagd op 3.500 synthetische candles.
- Het synthetische artifact bleef terecht `approved_for_live=false`.
- `template.yaml`: YAML-structuur succesvol geparsed; `ScheduleV2` aanwezig.

Niet uitgevoerd:

- Geen echte Kraken-order geplaatst of geannuleerd, omdat daarvoor de API-key en rekening van de gebruiker nodig zijn.
- Geen `sam build` of Docker-build uitgevoerd, omdat SAM CLI en Docker niet in de uitvoeringsomgeving aanwezig waren.

Begin daarom met `TradingMode=paper`. Live mode heeft daarnaast drie expliciete vergrendelingen: `TradingMode=live`, `LiveTradingEnabled=true`, `LiveConfirmation=I_UNDERSTAND_LIVE_TRADING`, plus een modelartifact met `approved_for_live=true`.
