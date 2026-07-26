from __future__ import annotations

import base64
import hashlib
import hmac
import time
import urllib.parse
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN
from typing import Any

import boto3
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .domain import Fill, PairRules


class KrakenError(RuntimeError):
    pass


class KrakenClient:
    BASE_URL = "https://api.kraken.com"

    def __init__(self, secret_name: str = "", authenticated: bool = False):
        self.secret_name = secret_name
        self.authenticated = authenticated
        self._credentials: dict[str, str] | None = None
        self._session = requests.Session()
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
        )
        self._session.mount("https://", HTTPAdapter(max_retries=retry))

    def _get_credentials(self) -> dict[str, str]:
        if not self.authenticated:
            raise RuntimeError("Authenticated Kraken call attempted without credentials")
        if self._credentials is None:
            response = boto3.client("secretsmanager").get_secret_value(SecretId=self.secret_name)
            import json

            data = json.loads(response["SecretString"])
            self._credentials = {
                "api_key": data["api_key"],
                "api_secret": data["api_secret"],
            }
        return self._credentials

    @staticmethod
    def _nonce() -> int:
        return time.time_ns() // 1_000

    @staticmethod
    def _deadline(seconds: int = 10) -> str:
        value = datetime.now(timezone.utc) + timedelta(seconds=seconds)
        return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")

    @staticmethod
    def _signature(path: str, data: dict[str, Any], api_secret: str) -> str:
        encoded = urllib.parse.urlencode(data)
        message = str(data["nonce"]).encode() + encoded.encode()
        digest = hashlib.sha256(message).digest()
        mac = hmac.new(
            base64.b64decode(api_secret),
            path.encode() + digest,
            hashlib.sha512,
        )
        return base64.b64encode(mac.digest()).decode()

    @staticmethod
    def _unwrap(response: requests.Response) -> dict[str, Any]:
        response.raise_for_status()
        payload = response.json()
        errors = payload.get("error") or []
        if errors:
            raise KrakenError("; ".join(errors))
        return payload.get("result", {})

    def public_get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        response = self._session.get(
            f"{self.BASE_URL}{path}", params=params or {}, timeout=(5, 20)
        )
        return self._unwrap(response)

    def private_post(self, path: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        credentials = self._get_credentials()
        payload = dict(data or {})
        payload["nonce"] = self._nonce()
        headers = {
            "API-Key": credentials["api_key"],
            "API-Sign": self._signature(path, payload, credentials["api_secret"]),
            "Content-Type": "application/x-www-form-urlencoded",
        }
        response = self._session.post(
            f"{self.BASE_URL}{path}", data=payload, headers=headers, timeout=(5, 25)
        )
        return self._unwrap(response)

    def get_ohlc(self, pair: str, interval: int, since: int | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"pair": pair, "interval": interval}
        if since:
            params["since"] = since
        result = self.public_get("/0/public/OHLC", params)
        pair_key = next(key for key in result if key != "last")
        rows = result[pair_key]
        rows = rows[:-1]  # current, not-yet-committed candle
        return [
            {
                "timestamp": int(row[0]),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "vwap": float(row[5]),
                "volume": float(row[6]),
                "trades": int(row[7]),
            }
            for row in rows
        ]

    def get_ticker(self, pair: str) -> dict[str, Decimal]:
        result = self.public_get("/0/public/Ticker", {"pair": pair})
        ticker = next(iter(result.values()))
        ask = Decimal(str(ticker["a"][0]))
        bid = Decimal(str(ticker["b"][0]))
        last = Decimal(str(ticker["c"][0]))
        spread_bps = (ask - bid) / ((ask + bid) / 2) * Decimal("10000")
        return {"ask": ask, "bid": bid, "last": last, "spread_bps": spread_bps}

    def get_pair_rules(self, pair: str) -> PairRules:
        result = self.public_get("/0/public/AssetPairs", {"pair": pair})
        pair_key, item = next(iter(result.items()))
        fees = item.get("fees", [[0, 0.26]])
        taker_fee_rate = Decimal(str(fees[0][1])) / Decimal("100")
        return PairRules(
            pair_key=pair_key,
            altname=item.get("altname", pair),
            base_asset=item["base"],
            quote_asset=item["quote"],
            lot_decimals=int(item["lot_decimals"]),
            pair_decimals=int(item["pair_decimals"]),
            order_min=Decimal(str(item.get("ordermin", "0"))),
            cost_min=Decimal(str(item.get("costmin", "0"))),
            taker_fee_rate=taker_fee_rate,
        )

    def get_balances(self) -> dict[str, Decimal]:
        result = self.private_post("/0/private/Balance")
        return {key: Decimal(str(value)) for key, value in result.items()}

    def add_order(
        self,
        pair: str,
        side: str,
        ordertype: str,
        volume: Decimal,
        client_order_id: str,
        price: Decimal | None = None,
        validate: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "pair": pair,
            "type": side,
            "ordertype": ordertype,
            "volume": format(volume, "f"),
            "cl_ord_id": client_order_id,
            # Keep fees in quote currency so cost/fee accounting is unambiguous.
            "oflags": "fciq",
            "deadline": self._deadline(),
            "validate": "true" if validate else "false",
        }
        if price is not None:
            payload["price"] = format(price, "f")
        return self.private_post("/0/private/AddOrder", payload)

    @staticmethod
    def _fill_from_order(txid: str, order: dict[str, Any]) -> Fill:
        volume = Decimal(str(order.get("vol_exec", "0")))
        cost = Decimal(str(order.get("cost", "0")))
        fee = Decimal(str(order.get("fee", "0")))
        price = Decimal(str(order.get("price", "0")))
        if price <= 0 and volume > 0:
            price = cost / volume
        side = order.get("descr", {}).get("type", "")
        return Fill(
            txid=txid,
            side=side,
            status=str(order.get("status", "unknown")),
            volume=volume,
            price=price,
            cost=cost,
            fee=fee,
            raw=order,
        )

    def query_order(self, txid: str) -> Fill:
        result = self.private_post("/0/private/QueryOrders", {"txid": txid, "trades": "true"})
        if txid not in result:
            if len(result) != 1:
                raise KrakenError(f"Order {txid} not found")
            txid, order = next(iter(result.items()))
        else:
            order = result[txid]
        return self._fill_from_order(txid, order)

    def find_order_by_client_id(self, client_order_id: str) -> Fill | None:
        open_result = self.private_post(
            "/0/private/OpenOrders",
            {"trades": "true", "cl_ord_id": client_order_id},
        )
        open_orders = open_result.get("open", {})
        if open_orders:
            txid, order = max(
                open_orders.items(), key=lambda item: float(item[1].get("opentm", 0))
            )
            return self._fill_from_order(txid, order)

        closed_result = self.private_post(
            "/0/private/ClosedOrders",
            {
                "trades": "true",
                "cl_ord_id": client_order_id,
                "start": int(time.time()) - 7 * 24 * 60 * 60,
            },
        )
        closed_orders = closed_result.get("closed", {})
        if not closed_orders:
            return None
        txid, order = max(
            closed_orders.items(),
            key=lambda item: float(item[1].get("closetm", item[1].get("opentm", 0))),
        )
        return self._fill_from_order(txid, order)

    def wait_for_fill(self, txid: str, attempts: int = 8, delay_seconds: float = 0.5) -> Fill:
        last_fill: Fill | None = None
        for _ in range(attempts):
            last_fill = self.query_order(txid)
            if last_fill.status in {"closed", "canceled", "expired"}:
                return last_fill
            time.sleep(delay_seconds)
        assert last_fill is not None
        return last_fill

    def amend_order(self, txid: str, trigger_price: Decimal) -> dict[str, Any]:
        return self.private_post(
            "/0/private/AmendOrder",
            {
                "txid": txid,
                "trigger_price": format(trigger_price, "f"),
                "deadline": self._deadline(),
            },
        )

    def cancel_order(self, txid: str) -> dict[str, Any]:
        return self.private_post("/0/private/CancelOrder", {"txid": txid})

    @staticmethod
    def round_volume(volume: Decimal, decimals: int) -> Decimal:
        quantum = Decimal("1").scaleb(-decimals)
        return volume.quantize(quantum, rounding=ROUND_DOWN)

    @staticmethod
    def round_price(price: Decimal, decimals: int) -> Decimal:
        quantum = Decimal("1").scaleb(-decimals)
        return price.quantize(quantum, rounding=ROUND_DOWN)
