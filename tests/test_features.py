import numpy as np
import pandas as pd

from trading_bot.features import build_features


def candles(count: int = 250) -> pd.DataFrame:
    timestamp = np.arange(1_700_000_000, 1_700_000_000 + count * 900, 900)
    index = np.arange(count)
    close = 100 + index * 0.04 + np.sin(index / 5)
    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "vwap": close,
            "volume": 10 + np.arange(count) % 7,
            "trades": 50 + np.arange(count) % 11,
        }
    )


def test_feature_history_does_not_change_when_future_is_appended():
    first = candles(240)
    extended = candles(250)
    feature_first = build_features(first)
    feature_extended = build_features(extended)
    pd.testing.assert_series_equal(
        feature_first.iloc[220], feature_extended.iloc[220], check_names=False
    )


def test_latest_features_are_available_after_warmup():
    feature = build_features(candles())
    assert not feature.iloc[-1].isna().any()
