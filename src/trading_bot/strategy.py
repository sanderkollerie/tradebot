from __future__ import annotations

from decimal import Decimal
from typing import Any

import pandas as pd

from .config import Settings
from .domain import BotState, Decision
from .features import regime_allows_entry
from .model import ProbabilityModel
from .risk import calculate_position_notional, initial_stop_price, trailing_stop_price


def decide(
    settings: Settings,
    model: ProbabilityModel,
    feature_row: pd.Series,
    candle: dict[str, Any],
    state: BotState,
    available_cash: Decimal,
    equity: Decimal,
    probability_up: float,
) -> Decision:
    close = Decimal(str(candle["close"]))
    atr_fraction = Decimal(str(feature_row["atr_pct_14"]))
    atr_value = close * atr_fraction

    diagnostics = {
        "buy_threshold": model.buy_threshold,
        "sell_threshold": model.sell_threshold,
        "atr_fraction": atr_fraction,
        "adx": float(feature_row["adx_14"]),
        "volatility_ratio": float(feature_row["volatility_ratio"]),
        "ema_21_55": float(feature_row["ema_21_55"]),
    }

    if state.position == "LONG":
        bars_held = 0
        if state.entry_candle:
            bars_held = max(
                0,
                (int(candle["timestamp"]) - state.entry_candle)
                // (settings.interval_minutes * 60),
            )
        diagnostics["bars_held"] = bars_held

        if probability_up <= model.sell_threshold:
            return Decision(
                action="SELL",
                reason="model_exit",
                probability_up=probability_up,
                model_version=model.version,
                diagnostics=diagnostics,
            )
        if bars_held >= settings.max_holding_bars:
            return Decision(
                action="SELL",
                reason="max_holding_period",
                probability_up=probability_up,
                model_version=model.version,
                diagnostics=diagnostics,
            )

        peak = max(state.peak_price, Decimal(str(candle["high"])))
        new_stop = trailing_stop_price(
            state.stop_price, peak, atr_value, settings.trailing_atr_multiple
        )
        state.peak_price = peak
        return Decision(
            action="HOLD",
            reason="position_open",
            probability_up=probability_up,
            stop_price=new_stop,
            model_version=model.version,
            diagnostics=diagnostics,
        )

    if int(candle["timestamp"]) < state.cooldown_until_candle:
        return Decision(
            action="HOLD",
            reason="cooldown_after_loss",
            probability_up=probability_up,
            model_version=model.version,
            diagnostics=diagnostics,
        )

    regime_ok, regime_reason = regime_allows_entry(feature_row)
    diagnostics["regime"] = regime_reason
    if not regime_ok:
        return Decision(
            action="HOLD",
            reason=regime_reason,
            probability_up=probability_up,
            model_version=model.version,
            diagnostics=diagnostics,
        )

    if probability_up < model.buy_threshold:
        return Decision(
            action="HOLD",
            reason="probability_below_entry_threshold",
            probability_up=probability_up,
            model_version=model.version,
            diagnostics=diagnostics,
        )

    notional = calculate_position_notional(
        equity_eur=equity,
        available_cash_eur=available_cash,
        atr_fraction=atr_fraction,
        risk_per_trade=settings.risk_per_trade,
        max_position_fraction=settings.max_position_fraction,
        max_order_eur=settings.max_order_eur,
        cash_reserve_eur=settings.min_cash_reserve_eur,
        stop_atr_multiple=settings.stop_atr_multiple,
        minimum_stop_fraction=settings.minimum_stop_fraction,
    )
    if notional <= 0:
        return Decision(
            action="HOLD",
            reason="position_size_zero",
            probability_up=probability_up,
            model_version=model.version,
            diagnostics=diagnostics,
        )

    stop = initial_stop_price(
        close, atr_value, settings.stop_atr_multiple, settings.minimum_stop_fraction
    )
    return Decision(
        action="BUY",
        reason="model_entry",
        probability_up=probability_up,
        notional_eur=notional,
        stop_price=stop,
        model_version=model.version,
        diagnostics=diagnostics,
    )
