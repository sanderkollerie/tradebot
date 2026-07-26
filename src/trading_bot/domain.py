from __future__ import annotations

from dataclasses import asdict, dataclass, field
from decimal import Decimal
from typing import Any


ZERO = Decimal("0")


@dataclass
class PairRules:
    pair_key: str
    altname: str
    base_asset: str
    quote_asset: str
    lot_decimals: int
    pair_decimals: int
    order_min: Decimal
    cost_min: Decimal
    taker_fee_rate: Decimal


@dataclass
class BotState:
    bot_id: str
    mode: str
    paper_cash: Decimal
    volume: Decimal = ZERO
    entry_price: Decimal = ZERO
    entry_cost: Decimal = ZERO
    entry_candle: int = 0
    peak_price: Decimal = ZERO
    stop_price: Decimal = ZERO
    stop_order_txid: str = ""
    pending_order_txid: str = ""
    pending_side: str = ""
    pending_client_order_id: str = ""
    pending_created_at: int = 0
    pending_candle_time: int = 0
    last_processed_candle: int = 0
    last_market_candle: int = 0
    cooldown_until_candle: int = 0
    realized_pnl: Decimal = ZERO
    total_fees: Decimal = ZERO
    trade_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    model_version: str = ""
    equity_peak: Decimal = ZERO
    risk_day: str = ""
    day_start_equity: Decimal = ZERO
    halted_reason: str = ""
    version: int = 0

    @property
    def position(self) -> str:
        return "LONG" if self.volume > ZERO else "FLAT"

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["position"] = self.position
        return result


@dataclass
class Fill:
    txid: str
    side: str
    status: str
    volume: Decimal
    price: Decimal
    cost: Decimal
    fee: Decimal
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class Decision:
    action: str
    reason: str
    probability_up: float
    notional_eur: Decimal = ZERO
    stop_price: Decimal = ZERO
    model_version: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)
