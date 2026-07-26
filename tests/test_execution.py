from decimal import Decimal
from types import SimpleNamespace

from trading_bot.domain import BotState, Fill, PairRules
from trading_bot.execution import ExecutionEngine


class FakeKraken:
    def __init__(self):
        self.amended = None
        self.existing = None

    @staticmethod
    def round_volume(value, decimals):
        return value.quantize(Decimal("1").scaleb(-decimals))

    @staticmethod
    def round_price(value, decimals):
        return value.quantize(Decimal("1").scaleb(-decimals))

    def query_order(self, txid):
        return Fill(
            txid=txid,
            side="sell",
            status="open",
            volume=Decimal("0"),
            price=Decimal("0"),
            cost=Decimal("0"),
            fee=Decimal("0"),
        )

    def amend_order(self, txid, trigger_price):
        self.amended = (txid, trigger_price)
        return {"amend_id": "AMEND-1"}

    def find_order_by_client_id(self, client_order_id):
        return self.existing


def settings():
    return SimpleNamespace(
        bot_id="BTC-EUR-15M",
        cooldown_bars_after_loss=8,
        interval_minutes=15,
        paper_slippage_bps=Decimal("8"),
        paper_fee_rate=Decimal("0.0026"),
        pair="XBTEUR",
        stop_raise_min_fraction=Decimal("0.004"),
    )


def rules():
    return PairRules(
        pair_key="XXBTZEUR",
        altname="XBTEUR",
        base_asset="XXBT",
        quote_asset="ZEUR",
        lot_decimals=8,
        pair_decimals=1,
        order_min=Decimal("0.0001"),
        cost_min=Decimal("5"),
        taker_fee_rate=Decimal("0.0026"),
    )


def state():
    return BotState(
        bot_id="BTC-EUR-15M",
        mode="live",
        paper_cash=Decimal("0"),
        volume=Decimal("0.01"),
        entry_price=Decimal("50000"),
        entry_cost=Decimal("500"),
        entry_candle=1_700_000_000,
        peak_price=Decimal("52000"),
        stop_price=Decimal("49000"),
        stop_order_txid="STOP-1",
    )


def test_client_order_id_is_deterministic_short_uuid():
    engine = ExecutionEngine(settings(), FakeKraken(), rules())
    first = engine.client_order_id(1_700_000_000, "buy", "entry")
    second = engine.client_order_id(1_700_000_000, "buy", "entry")
    assert first == second
    assert len(first) == 32
    assert all(character in "0123456789abcdef" for character in first)


def test_replace_live_stop_amends_in_place():
    fake = FakeKraken()
    engine = ExecutionEngine(settings(), fake, rules())
    bot_state = state()

    result = engine.replace_live_stop(bot_state, Decimal("50000"), 1_700_000_900)

    assert fake.amended == ("STOP-1", Decimal("50000.0"))
    assert bot_state.stop_order_txid == "STOP-1"
    assert bot_state.stop_price == Decimal("50000.0")
    assert result["amend_id"] == "AMEND-1"


def test_ensure_live_stop_recovers_existing_order():
    fake = FakeKraken()
    fake.existing = Fill(
        txid="RECOVERED-STOP",
        side="sell",
        status="open",
        volume=Decimal("0"),
        price=Decimal("0"),
        cost=Decimal("0"),
        fee=Decimal("0"),
    )
    engine = ExecutionEngine(settings(), fake, rules())
    bot_state = state()
    bot_state.stop_order_txid = ""

    result = engine.ensure_live_stop(bot_state, 1_700_000_900)

    assert bot_state.stop_order_txid == "RECOVERED-STOP"
    assert result["protective_stop_recovered"] == "RECOVERED-STOP"


def test_partial_sell_preserves_remaining_cost_basis():
    engine = ExecutionEngine(settings(), FakeKraken(), rules())
    bot_state = state()
    bot_state.stop_order_txid = ""
    fill = Fill(
        txid="PARTIAL",
        side="sell",
        status="canceled",
        volume=Decimal("0.004"),
        price=Decimal("51000"),
        cost=Decimal("204"),
        fee=Decimal("0.53"),
    )

    pnl = engine._apply_sell_fill(bot_state, fill, 1_700_000_900)

    assert bot_state.volume == Decimal("0.006")
    assert bot_state.entry_cost == Decimal("300.0")
    assert pnl == Decimal("3.47")
    assert bot_state.position == "LONG"

from trading_bot.execution import ProtectiveStopUncertain


class RecoverAfterSubmitKraken(FakeKraken):
    def __init__(self):
        super().__init__()
        self.find_calls = 0
        self.add_calls = 0

    def find_order_by_client_id(self, client_order_id):
        self.find_calls += 1
        if self.find_calls == 1:
            return None
        return Fill(
            txid="STOP-AFTER-TIMEOUT",
            side="sell",
            status="open",
            volume=Decimal("0"),
            price=Decimal("0"),
            cost=Decimal("0"),
            fee=Decimal("0"),
        )

    def add_order(self, *args, **kwargs):
        self.add_calls += 1
        return {}


def test_stop_submission_without_txid_is_recovered_by_client_id():
    fake = RecoverAfterSubmitKraken()
    engine = ExecutionEngine(settings(), fake, rules())
    bot_state = state()
    bot_state.stop_order_txid = ""

    result = engine.ensure_live_stop(bot_state, 1_700_000_900)

    assert fake.add_calls == 1
    assert bot_state.stop_order_txid == "STOP-AFTER-TIMEOUT"
    assert result["protective_stop_recovered"] == "STOP-AFTER-TIMEOUT"


class QueryFailureKraken(FakeKraken):
    def find_order_by_client_id(self, client_order_id):
        raise RuntimeError("temporary query failure")


def test_stop_verification_failure_is_uncertain_not_emergency_exit():
    engine = ExecutionEngine(settings(), QueryFailureKraken(), rules())
    bot_state = state()
    bot_state.stop_order_txid = ""

    try:
        engine.ensure_live_stop(bot_state, 1_700_000_900)
    except ProtectiveStopUncertain:
        pass
    else:
        raise AssertionError("Expected ProtectiveStopUncertain")
