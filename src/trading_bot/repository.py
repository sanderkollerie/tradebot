from __future__ import annotations

import time
from decimal import Decimal
from typing import Any

import boto3
from botocore.exceptions import ClientError

from .domain import BotState


def _decimalize(value: Any) -> Any:
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, dict):
        return {key: _decimalize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decimalize(item) for item in value]
    return value


class Repository:
    def __init__(self, state_table: str, events_table: str):
        dynamodb = boto3.resource("dynamodb")
        self.state = dynamodb.Table(state_table)
        self.events = dynamodb.Table(events_table)

    def load_state(self, bot_id: str, mode: str, starting_cash: Decimal) -> BotState:
        response = self.state.get_item(Key={"bot_id": bot_id}, ConsistentRead=True)
        item = response.get("Item")
        if not item:
            state = BotState(bot_id=bot_id, mode=mode, paper_cash=starting_cash)
            self.save_state(state)
            return state

        fields = BotState.__dataclass_fields__
        data = {key: item[key] for key in fields if key in item}
        stored_mode = str(item.get("mode", mode))
        has_open_state = (
            Decimal(str(item.get("volume", "0"))) > 0
            or bool(item.get("pending_order_txid"))
            or bool(item.get("pending_client_order_id"))
            or bool(item.get("stop_order_txid"))
        )
        if stored_mode != mode and has_open_state:
            raise RuntimeError(
                f"Cannot switch state from {stored_mode} to {mode} while a position/order exists"
            )
        data["mode"] = mode
        data.setdefault("paper_cash", starting_cash)
        for key in [
            "paper_cash",
            "volume",
            "entry_price",
            "entry_cost",
            "peak_price",
            "stop_price",
            "realized_pnl",
            "total_fees",
            "equity_peak",
            "day_start_equity",
        ]:
            data[key] = Decimal(str(data.get(key, "0")))
        return BotState(**data)

    def save_state(self, state: BotState) -> None:
        state.version += 1
        item = _decimalize(state.as_dict())
        item["updated_at"] = int(time.time())
        self.state.put_item(Item=item)

    def acquire_candle(self, bot_id: str, candle_time: int) -> bool:
        event_id = f"CANDLE#{candle_time}"
        now = int(time.time())
        try:
            self.events.update_item(
                Key={"bot_id": bot_id, "event_id": event_id},
                UpdateExpression=(
                    "SET #s = :processing, created_at = :created, "
                    "updated_at = :created, #ttl = :ttl REMOVE error_message"
                ),
                ConditionExpression=(
                    "attribute_not_exists(event_id) OR #s = :error OR "
                    "(#s = :processing AND created_at < :stale)"
                ),
                ExpressionAttributeNames={"#s": "status", "#ttl": "ttl"},
                ExpressionAttributeValues={
                    ":processing": "PROCESSING",
                    ":error": "ERROR",
                    ":created": now,
                    ":stale": now - 10 * 60,
                    ":ttl": now + 60 * 60 * 24 * 90,
                },
            )
            return True
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
                return False
            raise

    def finish_candle(self, bot_id: str, candle_time: int, payload: dict[str, Any]) -> None:
        self.events.update_item(
            Key={"bot_id": bot_id, "event_id": f"CANDLE#{candle_time}"},
            UpdateExpression="SET #s = :status, finished_at = :finished, payload = :payload",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={
                ":status": "SUCCESS",
                ":finished": int(time.time()),
                ":payload": _decimalize(payload),
            },
        )

    def fail_candle(self, bot_id: str, candle_time: int, error: str) -> None:
        self.events.update_item(
            Key={"bot_id": bot_id, "event_id": f"CANDLE#{candle_time}"},
            UpdateExpression="SET #s = :status, finished_at = :finished, error_message = :error",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={
                ":status": "ERROR",
                ":finished": int(time.time()),
                ":error": error[:2000],
            },
        )

    def log_trade(self, bot_id: str, candle_time: int, payload: dict[str, Any]) -> None:
        timestamp = time.time_ns()
        self.events.put_item(
            Item={
                "bot_id": bot_id,
                "event_id": f"TRADE#{candle_time}#{timestamp}",
                "status": "RECORDED",
                "created_at": int(time.time()),
                "payload": _decimalize(payload),
                "ttl": int(time.time()) + 60 * 60 * 24 * 365 * 7,
            }
        )
