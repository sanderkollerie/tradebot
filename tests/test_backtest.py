import numpy as np
import pandas as pd

from trading_bot.backtest import BacktestConfig, run_backtest
from trading_bot.features import build_features


def test_backtest_runs_and_produces_metrics():
    count = 500
    timestamp = np.arange(1_700_000_000, 1_700_000_000 + count * 900, 900)
    close = 100 * np.exp(np.linspace(0, 0.25, count) + 0.01 * np.sin(np.arange(count) / 8))
    candles = pd.DataFrame(
        {
            "timestamp": timestamp,
            "open": close * 0.999,
            "high": close * 1.004,
            "low": close * 0.996,
            "close": close,
            "vwap": close,
            "volume": np.full(count, 100.0),
            "trades": np.full(count, 1000.0),
        }
    )
    features = build_features(candles).fillna(0.0)
    probabilities = np.full(count, 0.7)
    result = run_backtest(
        candles,
        features,
        probabilities,
        buy_threshold=0.65,
        sell_threshold=0.4,
        config=BacktestConfig(),
    )
    assert result.ending_equity > 0
    assert result.trade_count >= 1
    assert -1 < result.max_drawdown <= 0
