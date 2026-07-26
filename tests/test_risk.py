from decimal import Decimal

from trading_bot.risk import calculate_position_notional, initial_stop_price


def test_position_size_is_capped_by_risk():
    notional = calculate_position_notional(
        equity_eur=Decimal("10000"),
        available_cash_eur=Decimal("10000"),
        atr_fraction=Decimal("0.02"),
        risk_per_trade=Decimal("0.005"),
        max_position_fraction=Decimal("0.25"),
        max_order_eur=Decimal("10000"),
        cash_reserve_eur=Decimal("0"),
        stop_atr_multiple=Decimal("2"),
        minimum_stop_fraction=Decimal("0.01"),
    )
    assert notional == Decimal("1250.00")


def test_initial_stop_uses_wider_of_atr_and_minimum():
    stop = initial_stop_price(
        Decimal("100"), Decimal("1"), Decimal("2"), Decimal("0.03")
    )
    assert stop == Decimal("97.00")

from trading_bot.domain import BotState
from trading_bot.risk import apply_equity_limits


def test_daily_loss_halt_resets_on_next_utc_day():
    state = BotState(bot_id="x", mode="paper", paper_cash=Decimal("1000"))
    day_one = 1_700_000_000
    apply_equity_limits(state, Decimal("1000"), day_one, Decimal("0.15"), Decimal("0.03"))
    reason = apply_equity_limits(
        state, Decimal("969"), day_one + 900, Decimal("0.15"), Decimal("0.03")
    )
    assert reason == "daily_loss"

    reason = apply_equity_limits(
        state, Decimal("969"), day_one + 24 * 60 * 60, Decimal("0.15"), Decimal("0.03")
    )
    assert reason == ""


def test_max_drawdown_halt_is_persistent():
    state = BotState(bot_id="x", mode="paper", paper_cash=Decimal("1000"))
    timestamp = 1_700_000_000
    apply_equity_limits(state, Decimal("1100"), timestamp, Decimal("0.15"), Decimal("0.03"))
    reason = apply_equity_limits(
        state, Decimal("930"), timestamp + 900, Decimal("0.15"), Decimal("0.50")
    )
    assert reason == "max_drawdown"
    reason = apply_equity_limits(
        state, Decimal("1000"), timestamp + 24 * 60 * 60, Decimal("0.15"), Decimal("0.50")
    )
    assert reason == "max_drawdown"
