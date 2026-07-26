import numpy as np
import pandas as pd

from trading_bot.features import build_multimarket_features


def candles(count=220, interval=15, start=1_700_000_000):
    timestamp = np.arange(count) * interval * 60 + start
    close = 100 + np.linspace(0, 10, count) + np.sin(np.arange(count) / 5)
    return pd.DataFrame({
        "timestamp": timestamp,
        "open": close - 0.1,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "vwap": close,
        "volume": 10 + np.arange(count) % 7,
        "trades": 20 + np.arange(count) % 5,
    })


def test_multimarket_adds_prefixed_context_features():
    primary = candles()
    context = candles(count=80, interval=60)
    result = build_multimarket_features(primary, {"XBTEUR_60": context})
    assert "log_return_1" in result.columns
    assert "ctx_XBTEUR_60_log_return_1" in result.columns
    assert len(result) == len(primary)
