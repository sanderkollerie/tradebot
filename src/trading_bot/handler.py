from __future__ import annotations

import json
import logging
import time
from decimal import Decimal
from typing import Any

import pandas as pd

from .config import Settings
from .execution import ExecutionEngine
from .features import build_features, build_multimarket_features
from .kraken import KrakenClient
from .model import ModelStore
from .repository import Repository
from .risk import apply_equity_limits, trailing_stop_price
from .strategy import decide


LOGGER = logging.getLogger(__name__)
_SETTINGS: Settings | None = None
_REPOSITORY: Repository | None = None
_PUBLIC_KRAKEN: KrakenClient | None = None
_PRIVATE_KRAKEN: KrakenClient | None = None
_MODEL_STORE: ModelStore | None = None


def _json_default(value: Any) -> str:
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)}")


def _asset_balance(balances: dict[str, Decimal], candidates: list[str]) -> Decimal:
    for key in candidates:
        if key in balances:
            return balances[key]
    return Decimal("0")


def _allocated_live_equity(settings: Settings, state: Any, bid: Decimal) -> Decimal:
    unrealized = Decimal("0")
    if state.position == "LONG":
        unrealized = state.volume * bid - state.entry_cost
    return max(settings.live_capital_eur + state.realized_pnl + unrealized, Decimal("0"))


def _bootstrap() -> tuple[Settings, Repository, KrakenClient, KrakenClient | None, ModelStore]:
    global _SETTINGS, _REPOSITORY, _PUBLIC_KRAKEN, _PRIVATE_KRAKEN, _MODEL_STORE
    if _SETTINGS is None:
        _SETTINGS = Settings.from_env()
        logging.getLogger().setLevel(_SETTINGS.log_level)
        _REPOSITORY = Repository(_SETTINGS.state_table, _SETTINGS.events_table)
        _PUBLIC_KRAKEN = KrakenClient()
        _PRIVATE_KRAKEN = (
            KrakenClient(_SETTINGS.secret_name, authenticated=True)
            if _SETTINGS.mode == "live"
            else None
        )
        _MODEL_STORE = ModelStore(_SETTINGS.model_bucket, _SETTINGS.model_key)
    assert _REPOSITORY and _PUBLIC_KRAKEN and _MODEL_STORE
    return _SETTINGS, _REPOSITORY, _PUBLIC_KRAKEN, _PRIVATE_KRAKEN, _MODEL_STORE


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    settings, repository, public_kraken, private_kraken, model_store = _bootstrap()
    model = model_store.load()

    if model.pair != settings.pair or model.interval_minutes != settings.interval_minutes:
        raise RuntimeError(
            f"Model expects {model.pair}/{model.interval_minutes}m, "
            f"Lambda is configured for {settings.pair}/{settings.interval_minutes}m"
        )
    if settings.mode == "live" and settings.require_approved_model and not model.approved_for_live:
        raise RuntimeError("Model artifact is not approved_for_live")

    candles = public_kraken.get_ohlc(settings.pair, settings.interval_minutes)
    if len(candles) < 200:
        raise RuntimeError("Kraken returned too few closed candles")
    frame = pd.DataFrame(candles)
    if model.context_markets:
        context_frames: dict[str, pd.DataFrame] = {}
        for spec in model.context_markets:
            name = str(spec["name"])
            pair = str(spec["pair"])
            interval = int(spec["interval"])
            rows = public_kraken.get_ohlc(pair, interval)
            if len(rows) < 200:
                raise RuntimeError(f"Kraken returned too few candles for {name}")
            context_frames[name] = pd.DataFrame(rows)
        feature_frame = build_multimarket_features(frame, context_frames)
    else:
        feature_frame = build_features(frame)
    latest_index = len(frame) - 1
    latest = candles[-1]
    candle_time = int(latest["timestamp"])
    latest_features = feature_frame.iloc[[latest_index]]
    if latest_features[model.feature_names].isna().any(axis=None):
        raise RuntimeError("Latest feature vector contains NaN values")

    state = repository.load_state(settings.bot_id, settings.mode, settings.starting_cash_eur)
    pair_rules = public_kraken.get_pair_rules(settings.pair)
    ticker = public_kraken.get_ticker(settings.pair)

    execution_client = private_kraken if private_kraken is not None else public_kraken
    engine = ExecutionEngine(settings, execution_client, pair_rules)

    reconciliation: dict[str, Any] | None = None
    if settings.mode == "live":
        assert private_kraken is not None
        reconciliation = engine.reconcile_live_orders(state, candle_time)
        if state.position == "LONG" and not state.stop_order_txid:
            recovered_stop = engine.ensure_live_stop(state, candle_time)
            if recovered_stop:
                reconciliation = {**(reconciliation or {}), **recovered_stop}
        if reconciliation:
            repository.save_state(state)

    if state.pending_order_txid or state.pending_client_order_id:
        result = {
            "status": "PENDING_ORDER",
            "mode": settings.mode,
            "candle_time": candle_time,
            "pending_txid": state.pending_order_txid,
            "pending_client_order_id": state.pending_client_order_id,
            "pending_candle_time": state.pending_candle_time,
            "reconciliation": reconciliation,
        }
        LOGGER.info(json.dumps(result, default=_json_default))
        return result

    # In paper mode, replay every newly observed candle so a missed invocation
    # cannot silently skip a protective stop.
    if settings.mode == "paper" and state.position == "LONG":
        for index, historical in enumerate(candles):
            historical_time = int(historical["timestamp"])
            if historical_time <= state.last_market_candle:
                continue
            if Decimal(str(historical["low"])) <= state.stop_price:
                trade = engine.paper_sell(
                    state,
                    ticker["bid"],
                    historical_time,
                    reason="protective_stop",
                    forced_price=min(
                        Decimal(str(historical["open"])), state.stop_price
                    ),
                )
                state.last_market_candle = historical_time
                repository.save_state(state)
                repository.log_trade(settings.bot_id, historical_time, trade)
                result = {
                    "status": "SUCCESS",
                    "mode": settings.mode,
                    "action": "SELL",
                    "reason": "protective_stop",
                    "trade": trade,
                    "state": state.as_dict(),
                }
                LOGGER.info(json.dumps(result, default=_json_default))
                return result
            state.peak_price = max(state.peak_price, Decimal(str(historical["high"])))
            atr_fraction = feature_frame.iloc[index].get("atr_pct_14")
            if pd.notna(atr_fraction):
                atr_value = Decimal(str(historical["close"])) * Decimal(str(atr_fraction))
                state.stop_price = trailing_stop_price(
                    state.stop_price,
                    state.peak_price,
                    atr_value,
                    settings.trailing_atr_multiple,
                )
            state.last_market_candle = historical_time

    state.last_market_candle = max(state.last_market_candle, candle_time)

    if candle_time <= state.last_processed_candle:
        equity = (
            state.paper_cash + state.volume * ticker["bid"]
            if settings.mode == "paper"
            else _allocated_live_equity(settings, state, ticker["bid"])
        )
        result = {
            "status": "NO_NEW_CANDLE",
            "mode": settings.mode,
            "candle_time": candle_time,
            "position": state.position,
            "equity_eur": equity,
            "realized_pnl": state.realized_pnl,
            "reconciliation": reconciliation,
        }
        LOGGER.info(json.dumps(result, default=_json_default))
        return result

    if not repository.acquire_candle(settings.bot_id, candle_time):
        return {"status": "ALREADY_PROCESSING", "candle_time": candle_time}

    try:
        probability = float(model.predict_probability(latest_features)[0])

        if settings.mode == "paper":
            available_cash = state.paper_cash
            equity = state.paper_cash + state.volume * ticker["bid"]
        else:
            assert private_kraken is not None
            balances = private_kraken.get_balances()
            available_cash = _asset_balance(balances, [pair_rules.quote_asset, "ZEUR", "EUR"])
            bot_position_value = state.volume * ticker["bid"]
            allocated_equity = _allocated_live_equity(settings, state, ticker["bid"])
            equity = min(allocated_equity, available_cash + bot_position_value)

        risk_halt = apply_equity_limits(
            state,
            equity,
            candle_time,
            settings.max_drawdown_fraction,
            settings.max_daily_loss_fraction,
        )

        decision = decide(
            settings=settings,
            model=model,
            feature_row=feature_frame.iloc[latest_index],
            candle=latest,
            state=state,
            available_cash=available_cash,
            equity=equity,
            probability_up=probability,
        )
        decision.diagnostics["equity_eur"] = equity
        decision.diagnostics["equity_peak_eur"] = state.equity_peak
        decision.diagnostics["risk_halt"] = risk_halt
        if risk_halt:
            decision.action = "SELL" if state.position == "LONG" else "HOLD"
            decision.reason = f"risk_halt_{risk_halt}"

        if ticker["spread_bps"] > settings.max_spread_bps and decision.action == "BUY":
            decision.action = "HOLD"
            decision.reason = "spread_too_wide"
            decision.diagnostics["spread_bps"] = ticker["spread_bps"]

        trade: dict[str, Any] | None = None
        stop_update: dict[str, Any] | None = None

        if decision.action == "BUY":
            if settings.mode == "paper":
                trade = engine.paper_buy(
                    state,
                    decision.notional_eur,
                    ticker["ask"],
                    decision.stop_price,
                    candle_time,
                )
            else:
                client_order_id = engine.client_order_id(
                    candle_time, "buy", "entry"
                )
                state.pending_side = "buy"
                state.pending_client_order_id = client_order_id
                state.pending_created_at = int(time.time())
                state.pending_candle_time = candle_time
                state.stop_price = decision.stop_price
                repository.save_state(state)
                trade = engine.live_buy(
                    state,
                    decision.notional_eur,
                    ticker["ask"],
                    decision.stop_price,
                    candle_time,
                    client_order_id,
                )
        elif decision.action == "SELL":
            if settings.mode == "paper":
                trade = engine.paper_sell(
                    state, ticker["bid"], candle_time, decision.reason
                )
            else:
                client_order_id = engine.client_order_id(
                    candle_time, "sell", "model-exit"
                )
                state.pending_side = "sell"
                state.pending_client_order_id = client_order_id
                state.pending_created_at = int(time.time())
                state.pending_candle_time = candle_time
                repository.save_state(state)
                trade = engine.live_sell(
                    state, candle_time, decision.reason, client_order_id
                )
        elif state.position == "LONG" and decision.stop_price > state.stop_price:
            if settings.mode == "paper":
                state.stop_price = decision.stop_price
                stop_update = {"mode": "paper", "new_stop": state.stop_price}
            else:
                stop_update = engine.replace_live_stop(
                    state, decision.stop_price, candle_time
                )

        state.last_processed_candle = candle_time
        state.model_version = model.version
        repository.save_state(state)

        payload = {
            "status": "SUCCESS",
            "mode": settings.mode,
            "action": decision.action,
            "reason": decision.reason,
            "probability_up": probability,
            "price": ticker["last"],
            "spread_bps": ticker["spread_bps"],
            "candle_time": candle_time,
            "model_version": model.version,
            "model_approved_for_live": model.approved_for_live,
            "trade": trade,
            "stop_update": stop_update,
            "diagnostics": decision.diagnostics,
            "state": state.as_dict(),
        }
        if trade:
            repository.log_trade(settings.bot_id, candle_time, trade)
        repository.finish_candle(settings.bot_id, candle_time, payload)
        LOGGER.info(json.dumps(payload, default=_json_default))
        return payload
    except Exception as exc:
        repository.fail_candle(settings.bot_id, candle_time, str(exc))
        LOGGER.exception("Trading invocation failed")
        raise
