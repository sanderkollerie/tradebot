from __future__ import annotations

from decimal import Decimal
import hashlib
import time
from typing import Any

from .config import Settings
from .domain import BotState, Fill, PairRules
from .kraken import KrakenClient


class ProtectiveStopUncertain(RuntimeError):
    """The stop submission result is unknown; do not submit a conflicting exit."""


class ExecutionEngine:
    def __init__(self, settings: Settings, kraken: KrakenClient, pair_rules: PairRules):
        self.settings = settings
        self.kraken = kraken
        self.rules = pair_rules

    def client_order_id(self, candle_time: int, side: str, purpose: str) -> str:
        seed = f"{self.settings.bot_id}:{candle_time}:{side}:{purpose}".encode()
        # Kraken accepts a 32-character short UUID format.
        return hashlib.sha256(seed).hexdigest()[:32]

    @staticmethod
    def _clear_pending(state: BotState) -> None:
        state.pending_order_txid = ""
        state.pending_side = ""
        state.pending_client_order_id = ""
        state.pending_created_at = 0
        state.pending_candle_time = 0

    def _apply_buy_fill(self, state: BotState, fill: Fill, candle_time: int) -> None:
        if fill.volume <= 0:
            raise RuntimeError(f"Buy order {fill.txid} executed no volume")
        state.volume = fill.volume
        state.entry_price = fill.price if fill.price > 0 else fill.cost / fill.volume
        state.entry_cost = fill.cost + fill.fee
        state.entry_candle = candle_time
        state.peak_price = state.entry_price
        state.total_fees += fill.fee
        self._clear_pending(state)

    def _apply_sell_fill(self, state: BotState, fill: Fill, candle_time: int) -> Decimal:
        if fill.volume <= 0:
            raise RuntimeError(f"Sell order {fill.txid} executed no volume")
        if state.volume <= 0:
            raise RuntimeError(f"Sell order {fill.txid} filled without a tracked position")

        sold_volume = min(fill.volume, state.volume)
        fraction = sold_volume / state.volume
        cost_basis = state.entry_cost * fraction
        proceeds = fill.cost - fill.fee
        pnl = proceeds - cost_basis

        state.realized_pnl += pnl
        state.total_fees += fill.fee
        state.trade_count += 1
        if pnl >= 0:
            state.win_count += 1
        else:
            state.loss_count += 1
            state.cooldown_until_candle = candle_time + (
                self.settings.cooldown_bars_after_loss * self.settings.interval_minutes * 60
            )

        remaining_volume = state.volume - sold_volume
        remaining_cost = state.entry_cost - cost_basis
        volume_quantum = Decimal("1").scaleb(-self.rules.lot_decimals)

        if remaining_volume < max(self.rules.order_min, volume_quantum):
            state.volume = Decimal("0")
            state.entry_price = Decimal("0")
            state.entry_cost = Decimal("0")
            state.entry_candle = 0
            state.peak_price = Decimal("0")
            state.stop_price = Decimal("0")
            state.stop_order_txid = ""
        else:
            state.volume = remaining_volume
            state.entry_cost = max(remaining_cost, Decimal("0"))
            # The old protective stop cannot be trusted after any exit fill.
            state.stop_order_txid = ""

        self._clear_pending(state)
        return pnl

    def paper_buy(
        self,
        state: BotState,
        notional: Decimal,
        ask: Decimal,
        stop_price: Decimal,
        candle_time: int,
    ) -> dict[str, Any]:
        slippage = self.settings.paper_slippage_bps / Decimal("10000")
        price = ask * (Decimal("1") + slippage)
        fee = notional * self.settings.paper_fee_rate
        volume = self.kraken.round_volume((notional - fee) / price, self.rules.lot_decimals)
        if volume < self.rules.order_min or volume * price < self.rules.cost_min:
            raise RuntimeError("Paper order is below Kraken minimum")
        state.paper_cash -= notional
        state.volume = volume
        state.entry_price = price
        state.entry_cost = notional
        state.entry_candle = candle_time
        state.peak_price = price
        stop_fraction = max((ask - stop_price) / ask, Decimal("0"))
        state.stop_price = price * (Decimal("1") - stop_fraction)
        state.total_fees += fee
        return {
            "mode": "paper",
            "side": "buy",
            "price": price,
            "volume": volume,
            "cost": notional,
            "fee": fee,
            "stop_price": state.stop_price,
        }

    def paper_sell(
        self,
        state: BotState,
        bid: Decimal,
        candle_time: int,
        reason: str,
        forced_price: Decimal | None = None,
    ) -> dict[str, Any]:
        slippage = self.settings.paper_slippage_bps / Decimal("10000")
        reference = forced_price if forced_price is not None else bid
        price = reference * (Decimal("1") - slippage)
        gross = state.volume * price
        fee = gross * self.settings.paper_fee_rate
        proceeds = gross - fee
        pnl = proceeds - state.entry_cost
        volume = state.volume
        state.paper_cash += proceeds
        state.realized_pnl += pnl
        state.total_fees += fee
        state.trade_count += 1
        if pnl >= 0:
            state.win_count += 1
        else:
            state.loss_count += 1
            state.cooldown_until_candle = candle_time + (
                self.settings.cooldown_bars_after_loss * self.settings.interval_minutes * 60
            )
        state.volume = Decimal("0")
        state.entry_price = Decimal("0")
        state.entry_cost = Decimal("0")
        state.entry_candle = 0
        state.peak_price = Decimal("0")
        state.stop_price = Decimal("0")
        return {
            "mode": "paper",
            "side": "sell",
            "reason": reason,
            "price": price,
            "volume": volume,
            "gross": gross,
            "fee": fee,
            "proceeds": proceeds,
            "pnl": pnl,
        }

    def _emergency_exit(self, state: BotState, candle_time: int, purpose: str) -> dict[str, Any]:
        client_order_id = self.client_order_id(candle_time, "sell", purpose)
        existing = self.kraken.find_order_by_client_id(client_order_id)
        if existing is None:
            response = self.kraken.add_order(
                self.settings.pair,
                "sell",
                "market",
                self.kraken.round_volume(state.volume, self.rules.lot_decimals),
                client_order_id=client_order_id,
            )
            txids = response.get("txid", [])
            if not txids:
                raise RuntimeError(f"Emergency exit returned no txid: {response}")
            fill = self.kraken.wait_for_fill(txids[0])
        else:
            fill = existing
            if fill.status not in {"closed", "canceled", "expired"}:
                fill = self.kraken.wait_for_fill(fill.txid)

        if fill.volume > 0 and fill.status in {"closed", "canceled", "expired"}:
            pnl = self._apply_sell_fill(state, fill, candle_time)
            return {"txid": fill.txid, "status": fill.status, "pnl": pnl}
        return {"txid": fill.txid, "status": fill.status}

    def live_buy(
        self,
        state: BotState,
        notional: Decimal,
        ask: Decimal,
        stop_price: Decimal,
        candle_time: int,
        client_order_id: str,
    ) -> dict[str, Any]:
        volume = self.kraken.round_volume(
            notional / (ask * Decimal("1.005")), self.rules.lot_decimals
        )
        if volume < self.rules.order_min or volume * ask < self.rules.cost_min:
            raise RuntimeError("Live buy order is below Kraken minimum")

        state.stop_price = stop_price
        response = self.kraken.add_order(
            self.settings.pair,
            "buy",
            "market",
            volume,
            client_order_id=client_order_id,
        )
        txids = response.get("txid", [])
        if not txids:
            raise RuntimeError(f"Kraken returned no transaction id: {response}")
        txid = txids[0]
        state.pending_order_txid = txid
        state.pending_side = "buy"
        state.pending_client_order_id = client_order_id

        fill = self.kraken.wait_for_fill(txid)
        if fill.status not in {"closed", "canceled", "expired"}:
            return {"mode": "live", "side": "buy", "status": fill.status, "txid": txid}
        if fill.volume <= 0:
            self._clear_pending(state)
            return {"mode": "live", "side": "buy", "status": fill.status, "txid": txid}

        stop_fraction = max((ask - stop_price) / ask, Decimal("0"))
        self._apply_buy_fill(state, fill, candle_time)
        state.stop_price = state.entry_price * (Decimal("1") - stop_fraction)

        try:
            stop_result = self.ensure_live_stop(state, candle_time)
        except ProtectiveStopUncertain:
            # Do not send a second sell while Kraken's answer is uncertain.
            raise
        except Exception as exc:
            emergency = self._emergency_exit(state, candle_time, "stop-failed-emergency")
            raise RuntimeError(
                f"Protective stop definitively failed; emergency exit result: {emergency}: {exc}"
            ) from exc

        return {
            "mode": "live",
            "side": "buy",
            "txid": fill.txid,
            "price": fill.price,
            "volume": fill.volume,
            "cost": fill.cost,
            "fee": fill.fee,
            "stop": stop_result,
            "stop_txid": state.stop_order_txid,
            "stop_price": state.stop_price,
        }

    def live_sell(
        self,
        state: BotState,
        candle_time: int,
        reason: str,
        client_order_id: str,
    ) -> dict[str, Any]:
        if state.stop_order_txid:
            stop_fill = self.kraken.query_order(state.stop_order_txid)
            if stop_fill.status == "closed":
                pnl = self._apply_sell_fill(state, stop_fill, candle_time)
                return {
                    "mode": "live",
                    "side": "sell",
                    "reason": "protective_stop_already_filled",
                    "txid": stop_fill.txid,
                    "pnl": pnl,
                }
            if stop_fill.status in {"open", "pending"}:
                self.kraken.cancel_order(state.stop_order_txid)
                after_cancel = self.kraken.query_order(state.stop_order_txid)
                if after_cancel.status == "closed":
                    pnl = self._apply_sell_fill(state, after_cancel, candle_time)
                    return {
                        "mode": "live",
                        "side": "sell",
                        "reason": "protective_stop_filled_during_cancel",
                        "txid": after_cancel.txid,
                        "pnl": pnl,
                    }
                if after_cancel.status not in {"canceled", "expired"}:
                    raise RuntimeError("Could not safely cancel protective stop")
            state.stop_order_txid = ""

        volume = self.kraken.round_volume(state.volume, self.rules.lot_decimals)
        response = self.kraken.add_order(
            self.settings.pair,
            "sell",
            "market",
            volume,
            client_order_id=client_order_id,
        )
        txids = response.get("txid", [])
        if not txids:
            raise RuntimeError(f"Kraken returned no sell transaction id: {response}")
        txid = txids[0]
        state.pending_order_txid = txid
        state.pending_side = "sell"
        state.pending_client_order_id = client_order_id

        fill = self.kraken.wait_for_fill(txid)
        if fill.status not in {"closed", "canceled", "expired"}:
            return {"mode": "live", "side": "sell", "status": fill.status, "txid": txid}
        if fill.volume <= 0:
            self._clear_pending(state)
            self.ensure_live_stop(state, candle_time)
            return {"mode": "live", "side": "sell", "status": fill.status, "txid": txid}

        pnl = self._apply_sell_fill(state, fill, candle_time)
        remaining_stop = self.ensure_live_stop(state, candle_time) if state.position == "LONG" else None
        return {
            "mode": "live",
            "side": "sell",
            "reason": reason,
            "txid": fill.txid,
            "price": fill.price,
            "volume": fill.volume,
            "cost": fill.cost,
            "fee": fill.fee,
            "pnl": pnl,
            "remaining_stop": remaining_stop,
        }

    def _recover_client_order(
        self, client_order_id: str, attempts: int = 4, delay_seconds: float = 0.75
    ) -> tuple[Fill | None, bool]:
        had_query_error = False
        for attempt in range(attempts):
            try:
                recovered = self.kraken.find_order_by_client_id(client_order_id)
            except Exception:
                had_query_error = True
                recovered = None
            if recovered is not None:
                return recovered, had_query_error
            if attempt < attempts - 1:
                time.sleep(delay_seconds)
        return None, had_query_error

    def _accept_stop_fill(
        self, state: BotState, fill: Fill, candle_time: int, rounded_stop: Decimal
    ) -> dict[str, Any] | None:
        if fill.status == "closed":
            pnl = self._apply_sell_fill(state, fill, candle_time)
            return {"protective_stop_already_filled": fill.txid, "pnl": pnl}
        if fill.status in {"open", "pending"}:
            state.stop_order_txid = fill.txid
            state.stop_price = rounded_stop
            return {"protective_stop_recovered": fill.txid, "price": rounded_stop}
        if fill.status in {"canceled", "expired"} and fill.volume > 0:
            pnl = self._apply_sell_fill(state, fill, candle_time)
            if state.position == "FLAT":
                return {"protective_stop_partial_terminal": fill.txid, "pnl": pnl}
        return None

    def ensure_live_stop(self, state: BotState, candle_time: int) -> dict[str, Any] | None:
        if state.position != "LONG" or state.stop_order_txid:
            return None
        if state.stop_price <= 0:
            raise RuntimeError("Live position has no protective stop price")

        rounded_stop = self.kraken.round_price(state.stop_price, self.rules.pair_decimals)
        client_order_id = self.client_order_id(
            state.entry_candle or candle_time, "sell", "protective-stop"
        )
        try:
            existing = self.kraken.find_order_by_client_id(client_order_id)
        except Exception as query_exc:
            raise ProtectiveStopUncertain(
                f"Cannot verify whether a protective stop already exists: {query_exc}"
            ) from query_exc
        if existing is not None:
            accepted = self._accept_stop_fill(state, existing, candle_time, rounded_stop)
            if accepted is not None:
                return accepted

        submit_error: Exception | None = None
        try:
            response = self.kraken.add_order(
                self.settings.pair,
                "sell",
                "stop-loss",
                self.kraken.round_volume(state.volume, self.rules.lot_decimals),
                client_order_id=client_order_id,
                price=rounded_stop,
            )
        except Exception as exc:
            submit_error = exc
            response = {}

        txids = response.get("txid", [])
        if not txids:
            # A POST response can be lost after the exchange accepted the order.
            recovered, had_query_error = self._recover_client_order(client_order_id)
            if recovered is not None:
                accepted = self._accept_stop_fill(state, recovered, candle_time, rounded_stop)
                if accepted is not None:
                    return accepted
                raise RuntimeError(f"Recovered protective stop is terminal: {recovered.status}")
            if had_query_error:
                raise ProtectiveStopUncertain(
                    f"Protective stop result is uncertain after submission: {submit_error}"
                ) from submit_error
            raise RuntimeError("Protective stop was not accepted by Kraken") from submit_error
        state.stop_order_txid = txids[0]
        state.stop_price = rounded_stop
        return {"protective_stop_created": state.stop_order_txid, "price": rounded_stop}

    def reconcile_live_orders(self, state: BotState, candle_time: int) -> dict[str, Any] | None:
        if not state.pending_order_txid and state.pending_client_order_id:
            recovered = self.kraken.find_order_by_client_id(state.pending_client_order_id)
            if recovered is not None:
                state.pending_order_txid = recovered.txid
            elif state.pending_created_at and int(time.time()) - state.pending_created_at > 90:
                client_id = state.pending_client_order_id
                self._clear_pending(state)
                return {"intent_not_found": client_id, "status": "cleared"}
            else:
                return {
                    "pending_client_order_id": state.pending_client_order_id,
                    "status": "awaiting_recovery",
                }

        if state.pending_order_txid:
            fill = self.kraken.query_order(state.pending_order_txid)
            terminal = fill.status in {"closed", "canceled", "expired"}
            if terminal and fill.volume > 0:
                if state.pending_side == "buy":
                    entry_candle = state.pending_candle_time or candle_time
                    self._apply_buy_fill(state, fill, entry_candle)
                    stop = self.ensure_live_stop(state, candle_time)
                    return {
                        "reconciled": fill.txid,
                        "status": fill.status,
                        "side": fill.side,
                        "stop": stop,
                    }
                if state.pending_side == "sell":
                    pnl = self._apply_sell_fill(state, fill, candle_time)
                    stop = self.ensure_live_stop(state, candle_time) if state.position == "LONG" else None
                    return {
                        "reconciled": fill.txid,
                        "status": fill.status,
                        "side": fill.side,
                        "pnl": pnl,
                        "stop": stop,
                    }
            if terminal:
                self._clear_pending(state)
                return {"reconciled": fill.txid, "status": fill.status}
            return {"pending": fill.txid, "status": fill.status}

        if state.stop_order_txid:
            fill = self.kraken.query_order(state.stop_order_txid)
            if fill.status == "closed":
                pnl = self._apply_sell_fill(state, fill, candle_time)
                return {"protective_stop_filled": fill.txid, "pnl": pnl}
            if fill.status in {"canceled", "expired"}:
                state.stop_order_txid = ""
                if fill.volume > 0:
                    pnl = self._apply_sell_fill(state, fill, candle_time)
                    stop = self.ensure_live_stop(state, candle_time) if state.position == "LONG" else None
                    return {
                        "protective_stop_partial_terminal": fill.txid,
                        "pnl": pnl,
                        "stop": stop,
                    }
                return {"protective_stop_missing": fill.txid, "status": fill.status}
        return None

    def replace_live_stop(
        self, state: BotState, new_stop: Decimal, candle_time: int
    ) -> dict[str, Any] | None:
        if not state.stop_order_txid or new_stop <= state.stop_price:
            return None
        relative_raise = (new_stop - state.stop_price) / state.stop_price
        if relative_raise < self.settings.stop_raise_min_fraction:
            return None

        txid = state.stop_order_txid
        fill = self.kraken.query_order(txid)
        if fill.status == "closed":
            pnl = self._apply_sell_fill(state, fill, candle_time)
            return {"stop_filled": txid, "pnl": pnl}
        if fill.status in {"canceled", "expired"}:
            state.stop_order_txid = ""
            if fill.volume > 0:
                pnl = self._apply_sell_fill(state, fill, candle_time)
                if state.position == "FLAT":
                    return {"stop_partial_terminal": txid, "pnl": pnl}
            state.stop_price = new_stop
            return self.ensure_live_stop(state, candle_time)
        if fill.status not in {"open", "pending"}:
            raise RuntimeError(f"Cannot amend stop in status {fill.status}")

        rounded_stop = self.kraken.round_price(new_stop, self.rules.pair_decimals)
        response = self.kraken.amend_order(txid, rounded_stop)
        state.stop_price = rounded_stop
        return {
            "stop_txid": txid,
            "amend_id": response.get("amend_id", ""),
            "price": rounded_stop,
        }
