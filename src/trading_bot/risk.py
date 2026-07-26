from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN

from .domain import BotState


def calculate_stop_fraction(
    atr_fraction: Decimal,
    stop_atr_multiple: Decimal,
    minimum_stop_fraction: Decimal,
) -> Decimal:
    return max(atr_fraction * stop_atr_multiple, minimum_stop_fraction)


def calculate_position_notional(
    equity_eur: Decimal,
    available_cash_eur: Decimal,
    atr_fraction: Decimal,
    risk_per_trade: Decimal,
    max_position_fraction: Decimal,
    max_order_eur: Decimal,
    cash_reserve_eur: Decimal,
    stop_atr_multiple: Decimal,
    minimum_stop_fraction: Decimal,
) -> Decimal:
    usable_cash = max(available_cash_eur - cash_reserve_eur, Decimal("0"))
    if usable_cash <= 0 or equity_eur <= 0:
        return Decimal("0")

    stop_fraction = calculate_stop_fraction(
        atr_fraction, stop_atr_multiple, minimum_stop_fraction
    )
    risk_budget = equity_eur * risk_per_trade
    risk_sized_notional = risk_budget / stop_fraction
    exposure_cap = equity_eur * max_position_fraction
    notional = min(usable_cash, risk_sized_notional, exposure_cap, max_order_eur)
    return max(notional.quantize(Decimal("0.01"), rounding=ROUND_DOWN), Decimal("0"))


def initial_stop_price(
    entry_price: Decimal,
    atr_value: Decimal,
    stop_atr_multiple: Decimal,
    minimum_stop_fraction: Decimal,
) -> Decimal:
    atr_stop = entry_price - atr_value * stop_atr_multiple
    percentage_stop = entry_price * (Decimal("1") - minimum_stop_fraction)
    return min(atr_stop, percentage_stop)


def trailing_stop_price(
    current_stop: Decimal,
    peak_price: Decimal,
    atr_value: Decimal,
    trailing_atr_multiple: Decimal,
) -> Decimal:
    candidate = peak_price - atr_value * trailing_atr_multiple
    return max(current_stop, candidate)


def apply_equity_limits(
    state: BotState,
    equity: Decimal,
    candle_time: int,
    max_drawdown_fraction: Decimal,
    max_daily_loss_fraction: Decimal,
) -> str:
    """Update persistent daily/total equity circuit breakers."""
    day = datetime.fromtimestamp(candle_time, tz=timezone.utc).date().isoformat()

    if state.equity_peak <= 0:
        state.equity_peak = equity
    else:
        state.equity_peak = max(state.equity_peak, equity)

    if state.risk_day != day:
        state.risk_day = day
        state.day_start_equity = equity
        if state.halted_reason == "daily_loss":
            state.halted_reason = ""
    elif state.day_start_equity <= 0:
        state.day_start_equity = equity

    if (
        state.equity_peak > 0
        and equity <= state.equity_peak * (Decimal("1") - max_drawdown_fraction)
    ):
        state.halted_reason = "max_drawdown"
    elif (
        state.halted_reason != "max_drawdown"
        and state.day_start_equity > 0
        and equity
        <= state.day_start_equity * (Decimal("1") - max_daily_loss_fraction)
    ):
        state.halted_reason = "daily_loss"

    return state.halted_reason
