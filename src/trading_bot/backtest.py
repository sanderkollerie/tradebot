from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    interval_minutes: int = 15
    starting_cash: float = 10_000.0
    fee_rate: float = 0.0026
    slippage_bps: float = 8.0
    risk_per_trade: float = 0.005
    max_position_fraction: float = 0.25
    max_order_value: float = 2_500.0
    stop_atr_multiple: float = 2.2
    minimum_stop_fraction: float = 0.012
    trailing_atr_multiple: float = 2.8
    max_holding_bars: int = 192
    cooldown_bars_after_loss: int = 8
    cash_reserve: float = 25.0
    max_drawdown_fraction: float = 0.15
    max_daily_loss_fraction: float = 0.03


@dataclass
class BacktestResult:
    net_return: float
    buy_hold_return: float
    max_drawdown: float
    annualized_sharpe: float
    trade_count: int
    win_rate: float
    profit_factor: float
    exposure: float
    ending_equity: float
    trades: list[dict[str, Any]]

    def summary(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("trades", None)
        return data


def run_backtest(
    candles: pd.DataFrame,
    features: pd.DataFrame,
    probabilities: np.ndarray,
    buy_threshold: float,
    sell_threshold: float,
    config: BacktestConfig,
) -> BacktestResult:
    if len(candles) != len(probabilities):
        raise ValueError("candles and probabilities must have equal length")

    cash = config.starting_cash
    volume = 0.0
    entry_cost = 0.0
    entry_price = 0.0
    entry_index = 0
    stop_price = 0.0
    peak_price = 0.0
    pending_action: tuple[str, str, float] | None = None
    trades: list[dict[str, Any]] = []
    equity_curve: list[float] = []
    position_bars = 0
    cooldown_until_index = 0
    equity_peak = config.starting_cash
    risk_day = ""
    day_start_equity = config.starting_cash
    permanent_halt = False
    daily_halt = False
    slippage = config.slippage_bps / 10_000

    first_price = float(candles.iloc[0]["open"])

    for i in range(1, len(candles)):
        row = candles.iloc[i]
        feature = features.iloc[i]
        open_price = float(row["open"])
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])

        if pending_action is not None:
            side, reason, notional = pending_action
            pending_action = None
            if side == "buy" and volume == 0:
                execution_price = open_price * (1 + slippage)
                fee = notional * config.fee_rate
                volume = max((notional - fee) / execution_price, 0.0)
                cash -= notional
                entry_cost = notional
                entry_price = execution_price
                entry_index = i
                peak_price = execution_price
                atr_fraction = float(features.iloc[i - 1]["atr_pct_14"])
                stop_fraction = max(
                    atr_fraction * config.stop_atr_multiple,
                    config.minimum_stop_fraction,
                )
                stop_price = execution_price * (1 - stop_fraction)
            elif side == "sell" and volume > 0:
                execution_price = open_price * (1 - slippage)
                gross = volume * execution_price
                fee = gross * config.fee_rate
                proceeds = gross - fee
                pnl = proceeds - entry_cost
                cash += proceeds
                trades.append(
                    {
                        "entry_index": entry_index,
                        "exit_index": i,
                        "entry_price": entry_price,
                        "exit_price": execution_price,
                        "pnl": pnl,
                        "reason": reason,
                    }
                )
                if pnl < 0:
                    cooldown_until_index = i + config.cooldown_bars_after_loss
                volume = 0.0
                entry_cost = 0.0
                entry_price = 0.0
                stop_price = 0.0
                peak_price = 0.0

        if volume > 0:
            position_bars += 1
            if low <= stop_price:
                execution_price = min(open_price, stop_price) * (1 - slippage)
                gross = volume * execution_price
                fee = gross * config.fee_rate
                proceeds = gross - fee
                pnl = proceeds - entry_cost
                cash += proceeds
                trades.append(
                    {
                        "entry_index": entry_index,
                        "exit_index": i,
                        "entry_price": entry_price,
                        "exit_price": execution_price,
                        "pnl": pnl,
                        "reason": "stop",
                    }
                )
                if pnl < 0:
                    cooldown_until_index = i + config.cooldown_bars_after_loss
                volume = 0.0
                entry_cost = 0.0
                entry_price = 0.0
                stop_price = 0.0
                peak_price = 0.0
            else:
                peak_price = max(peak_price, high)
                atr = float(feature["atr_pct_14"]) * close
                stop_price = max(stop_price, peak_price - atr * config.trailing_atr_multiple)

        current_equity = cash + volume * close
        timestamp = pd.to_datetime(row["timestamp"], utc=True)
        current_day = timestamp.date().isoformat()
        if current_day != risk_day:
            risk_day = current_day
            day_start_equity = current_equity
            daily_halt = False
        equity_peak = max(equity_peak, current_equity)
        if current_equity <= equity_peak * (1 - config.max_drawdown_fraction):
            permanent_halt = True
        if current_equity <= day_start_equity * (1 - config.max_daily_loss_fraction):
            daily_halt = True
        risk_halted = permanent_halt or daily_halt

        probability = float(probabilities[i])
        if (
            volume == 0
            and not risk_halted
            and i >= cooldown_until_index
            and probability >= buy_threshold
        ):
            adx = float(feature.get("adx_14", np.nan))
            trend = float(feature.get("ema_21_55", np.nan))
            vol_ratio = float(feature.get("volatility_ratio", np.nan))
            if not np.isnan(adx) and not np.isnan(trend) and not np.isnan(vol_ratio):
                if vol_ratio <= 2.8 and (adx >= 0.12 or abs(trend) >= 0.001):
                    equity = cash
                    atr_fraction = max(float(feature["atr_pct_14"]), 1e-6)
                    stop_fraction = max(
                        atr_fraction * config.stop_atr_multiple,
                        config.minimum_stop_fraction,
                    )
                    usable_cash = max(cash - config.cash_reserve, 0.0)
                    notional = min(
                        usable_cash,
                        equity * config.max_position_fraction,
                        equity * config.risk_per_trade / stop_fraction,
                        config.max_order_value,
                    )
                    if notional > 10:
                        pending_action = ("buy", "model_entry", notional)
        elif volume > 0:
            held = i - entry_index
            if risk_halted:
                pending_action = (
                    "sell",
                    "max_drawdown" if permanent_halt else "daily_loss",
                    0.0,
                )
            elif probability <= sell_threshold:
                pending_action = ("sell", "model_exit", 0.0)
            elif held >= config.max_holding_bars:
                pending_action = ("sell", "max_holding", 0.0)

        equity_curve.append(cash + volume * close)

    if volume > 0:
        final_price = float(candles.iloc[-1]["close"]) * (1 - slippage)
        gross = volume * final_price
        fee = gross * config.fee_rate
        proceeds = gross - fee
        pnl = proceeds - entry_cost
        cash += proceeds
        trades.append(
            {
                "entry_index": entry_index,
                "exit_index": len(candles) - 1,
                "entry_price": entry_price,
                "exit_price": final_price,
                "pnl": pnl,
                "reason": "end_of_test",
            }
        )
        equity_curve[-1] = cash

    equity = pd.Series(equity_curve, dtype=float)
    returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    periods_per_year = 365 * 24 * 60 / config.interval_minutes
    sharpe = 0.0
    if returns.std(ddof=0) > 0:
        sharpe = returns.mean() / returns.std(ddof=0) * sqrt(periods_per_year)

    pnl_values = [float(trade["pnl"]) for trade in trades]
    wins = [value for value in pnl_values if value > 0]
    losses = [value for value in pnl_values if value < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

    ending_equity = float(equity.iloc[-1]) if not equity.empty else config.starting_cash
    buy_hold_return = float(candles.iloc[-1]["close"] / first_price - 1)
    return BacktestResult(
        net_return=ending_equity / config.starting_cash - 1,
        buy_hold_return=buy_hold_return,
        max_drawdown=float(drawdown.min()) if not drawdown.empty else 0.0,
        annualized_sharpe=float(sharpe),
        trade_count=len(trades),
        win_rate=len(wins) / len(trades) if trades else 0.0,
        profit_factor=float(profit_factor),
        exposure=position_bars / max(len(candles), 1),
        ending_equity=ending_equity,
        trades=trades,
    )
