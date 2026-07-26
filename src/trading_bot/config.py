from __future__ import annotations

import os
from dataclasses import dataclass
from decimal import Decimal


def _bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name, str(default)).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false, got {value!r}")


@dataclass(frozen=True)
class Settings:
    bot_id: str
    pair: str
    interval_minutes: int
    mode: str

    state_table: str
    events_table: str
    secret_name: str
    model_bucket: str
    model_key: str

    starting_cash_eur: Decimal
    live_capital_eur: Decimal
    max_order_eur: Decimal
    min_cash_reserve_eur: Decimal
    risk_per_trade: Decimal
    max_position_fraction: Decimal

    stop_atr_multiple: Decimal
    minimum_stop_fraction: Decimal
    trailing_atr_multiple: Decimal
    stop_raise_min_fraction: Decimal
    max_holding_bars: int
    cooldown_bars_after_loss: int
    max_spread_bps: Decimal
    paper_slippage_bps: Decimal
    paper_fee_rate: Decimal
    max_drawdown_fraction: Decimal
    max_daily_loss_fraction: Decimal

    live_enabled: bool
    live_confirmation: str
    require_approved_model: bool
    log_level: str

    @classmethod
    def from_env(cls) -> "Settings":
        mode = os.getenv("MODE", "paper").strip().lower()
        if mode not in {"paper", "live"}:
            raise ValueError("MODE must be 'paper' or 'live'")

        settings = cls(
            bot_id=os.getenv("BOT_ID", "BTC-EUR-15M"),
            pair=os.getenv("PAIR", "XBTEUR"),
            interval_minutes=int(os.getenv("INTERVAL_MINUTES", "15")),
            mode=mode,
            state_table=os.getenv("STATE_TABLE", "TradingBotState"),
            events_table=os.getenv("EVENTS_TABLE", "TradingBotEvents"),
            secret_name=os.getenv("KRAKEN_SECRET_NAME", "trading/kraken"),
            model_bucket=os.getenv("MODEL_BUCKET", ""),
            model_key=os.getenv("MODEL_KEY", "models/btc-eur-15m.joblib"),
            starting_cash_eur=Decimal(os.getenv("STARTING_CASH_EUR", "1000")),
            live_capital_eur=Decimal(os.getenv("LIVE_CAPITAL_EUR", "1000")),
            max_order_eur=Decimal(os.getenv("MAX_ORDER_EUR", "250")),
            min_cash_reserve_eur=Decimal(os.getenv("MIN_CASH_RESERVE_EUR", "25")),
            risk_per_trade=Decimal(os.getenv("RISK_PER_TRADE", "0.005")),
            max_position_fraction=Decimal(os.getenv("MAX_POSITION_FRACTION", "0.25")),
            stop_atr_multiple=Decimal(os.getenv("STOP_ATR_MULTIPLE", "2.2")),
            minimum_stop_fraction=Decimal(os.getenv("MINIMUM_STOP_FRACTION", "0.012")),
            trailing_atr_multiple=Decimal(os.getenv("TRAILING_ATR_MULTIPLE", "2.8")),
            stop_raise_min_fraction=Decimal(os.getenv("STOP_RAISE_MIN_FRACTION", "0.004")),
            max_holding_bars=int(os.getenv("MAX_HOLDING_BARS", "192")),
            cooldown_bars_after_loss=int(os.getenv("COOLDOWN_BARS_AFTER_LOSS", "8")),
            max_spread_bps=Decimal(os.getenv("MAX_SPREAD_BPS", "35")),
            paper_slippage_bps=Decimal(os.getenv("PAPER_SLIPPAGE_BPS", "8")),
            paper_fee_rate=Decimal(os.getenv("PAPER_FEE_RATE", "0.0026")),
            max_drawdown_fraction=Decimal(os.getenv("MAX_DRAWDOWN_FRACTION", "0.15")),
            max_daily_loss_fraction=Decimal(os.getenv("MAX_DAILY_LOSS_FRACTION", "0.03")),
            live_enabled=_bool("LIVE_TRADING_ENABLED", False),
            live_confirmation=os.getenv("LIVE_CONFIRMATION", ""),
            require_approved_model=_bool("REQUIRE_APPROVED_MODEL", True),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.interval_minutes not in {1, 5, 15, 30, 60, 240, 1440}:
            raise ValueError("INTERVAL_MINUTES is not supported by Kraken spot OHLC")
        if not (Decimal("0") < self.risk_per_trade <= Decimal("0.05")):
            raise ValueError("RISK_PER_TRADE must be between 0 and 0.05")
        if not (Decimal("0") < self.max_position_fraction <= Decimal("1")):
            raise ValueError("MAX_POSITION_FRACTION must be between 0 and 1")
        if not (Decimal("0") < self.max_drawdown_fraction < Decimal("1")):
            raise ValueError("MAX_DRAWDOWN_FRACTION must be between 0 and 1")
        if not (Decimal("0") < self.max_daily_loss_fraction < Decimal("1")):
            raise ValueError("MAX_DAILY_LOSS_FRACTION must be between 0 and 1")
        if self.mode == "live":
            if not self.live_enabled:
                raise RuntimeError("Live mode requires LIVE_TRADING_ENABLED=true")
            if self.live_confirmation != "I_UNDERSTAND_LIVE_TRADING":
                raise RuntimeError(
                    "Live mode requires LIVE_CONFIRMATION=I_UNDERSTAND_LIVE_TRADING"
                )
            if not self.model_bucket:
                raise RuntimeError("MODEL_BUCKET is required in live mode")
