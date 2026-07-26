from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "vwap", "volume", "trades"]


def normalize_candles(frame: pd.DataFrame, interval_minutes: int) -> pd.DataFrame:
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing candle columns: {sorted(missing)}")

    df = frame.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df = df.drop_duplicates("timestamp", keep="last").set_index("timestamp")

    for column in ["open", "high", "low", "close", "vwap", "volume", "trades"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    frequency = f"{interval_minutes}min"
    full_index = pd.date_range(df.index.min(), df.index.max(), freq=frequency, tz="UTC")
    df = df.reindex(full_index)

    previous_close = df["close"].ffill()
    for column in ["open", "high", "low", "close", "vwap"]:
        df[column] = df[column].fillna(previous_close)
    df["volume"] = df["volume"].fillna(0.0)
    df["trades"] = df["trades"].fillna(0.0)

    df.index.name = "timestamp"
    return df.reset_index()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    previous_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False).mean()


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    atr = _atr(df, period).replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def _rolling_z(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0).replace(0, np.nan)
    return (series - mean) / std


def build_features(candles: pd.DataFrame) -> pd.DataFrame:
    df = candles.copy()
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    open_ = df["open"].astype(float)
    volume = df["volume"].astype(float)
    trades = df["trades"].astype(float)
    vwap = df["vwap"].astype(float)

    features = pd.DataFrame(index=df.index)

    log_close = np.log(close.replace(0, np.nan))
    for horizon in [1, 2, 4, 8, 16, 32, 64]:
        features[f"log_return_{horizon}"] = log_close.diff(horizon)

    ema8 = close.ewm(span=8, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema55 = close.ewm(span=55, adjust=False).mean()
    ema144 = close.ewm(span=144, adjust=False).mean()

    features["ema_8_21"] = ema8 / ema21 - 1
    features["ema_21_55"] = ema21 / ema55 - 1
    features["ema_55_144"] = ema55 / ema144 - 1
    features["close_ema21"] = close / ema21 - 1
    features["close_ema55"] = close / ema55 - 1

    atr14 = _atr(df, 14)
    features["atr_pct_14"] = atr14 / close
    features["atr_ratio_14_64"] = features["atr_pct_14"] / features["atr_pct_14"].rolling(64).mean()
    features["adx_14"] = _adx(df, 14) / 100
    features["rsi_14"] = (_rsi(close, 14) - 50) / 50
    features["rsi_28"] = (_rsi(close, 28) - 50) / 50

    returns = close.pct_change()
    features["volatility_16"] = returns.rolling(16).std(ddof=0)
    features["volatility_64"] = returns.rolling(64).std(ddof=0)
    features["volatility_ratio"] = features["volatility_16"] / features["volatility_64"].replace(0, np.nan)

    middle = close.rolling(20).mean()
    band_std = close.rolling(20).std(ddof=0).replace(0, np.nan)
    features["bollinger_z_20"] = (close - middle) / band_std

    rolling_high_20 = high.rolling(20).max()
    rolling_low_20 = low.rolling(20).min()
    rolling_high_55 = high.rolling(55).max()
    rolling_low_55 = low.rolling(55).min()
    features["distance_high_20"] = close / rolling_high_20 - 1
    features["distance_low_20"] = close / rolling_low_20 - 1
    features["distance_high_55"] = close / rolling_high_55 - 1
    features["distance_low_55"] = close / rolling_low_55 - 1

    candle_range = (high - low).replace(0, np.nan)
    features["range_pct"] = candle_range / close
    features["body_fraction"] = (close - open_) / candle_range
    features["upper_wick_fraction"] = (high - np.maximum(open_, close)) / candle_range
    features["lower_wick_fraction"] = (np.minimum(open_, close) - low) / candle_range
    features["vwap_gap"] = close / vwap.replace(0, np.nan) - 1

    features["volume_z_20"] = _rolling_z(np.log1p(volume), 20)
    features["volume_z_60"] = _rolling_z(np.log1p(volume), 60)
    features["trades_z_20"] = _rolling_z(np.log1p(trades), 20)

    signed_volume = np.sign(close.diff()).fillna(0) * volume
    obv = signed_volume.cumsum()
    features["obv_slope_20"] = (obv - obv.shift(20)) / volume.rolling(20).sum().replace(0, np.nan)

    if pd.api.types.is_numeric_dtype(df["timestamp"]):
        timestamp = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    else:
        timestamp = pd.to_datetime(df["timestamp"], utc=True)
    minutes = timestamp.dt.hour * 60 + timestamp.dt.minute
    features["time_sin"] = np.sin(2 * np.pi * minutes / 1440)
    features["time_cos"] = np.cos(2 * np.pi * minutes / 1440)
    features["week_sin"] = np.sin(2 * np.pi * timestamp.dt.dayofweek / 7)
    features["week_cos"] = np.cos(2 * np.pi * timestamp.dt.dayofweek / 7)

    features = features.replace([np.inf, -np.inf], np.nan)
    return features


def regime_allows_entry(feature_row: pd.Series) -> tuple[bool, str]:
    atr_ratio = float(feature_row.get("atr_ratio_14_64", np.nan))
    volatility_ratio = float(feature_row.get("volatility_ratio", np.nan))
    adx = float(feature_row.get("adx_14", np.nan))
    ema_trend = float(feature_row.get("ema_21_55", np.nan))

    if any(np.isnan(value) for value in [atr_ratio, volatility_ratio, adx, ema_trend]):
        return False, "incomplete_features"
    if atr_ratio > 2.8 or volatility_ratio > 2.8:
        return False, "volatility_shock"
    if adx < 0.12 and abs(ema_trend) < 0.001:
        return False, "no_edge_regime"
    return True, "regime_ok"

CONTEXT_FEATURE_COLUMNS = [
    "log_return_1",
    "log_return_4",
    "log_return_16",
    "ema_8_21",
    "ema_21_55",
    "atr_pct_14",
    "adx_14",
    "rsi_14",
    "volatility_16",
    "volume_z_20",
]


def build_multimarket_features(
    primary_candles: pd.DataFrame,
    context_candles: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Build primary features plus backward-looking context features.

    Context keys should be stable identifiers such as ``XBTEUR_60``. Every
    context series is aligned backward to the primary candle timestamp, so no
    future candle can leak into the current feature row.
    """
    primary = primary_candles.copy().reset_index(drop=True)
    result = build_features(primary)
    context_candles = context_candles or {}

    primary_times = pd.to_datetime(primary["timestamp"], unit="s", utc=True, errors="coerce")
    left = pd.DataFrame({"timestamp": primary_times}).sort_values("timestamp")

    context_blocks: list[pd.DataFrame] = []
    for name in sorted(context_candles):
        context = context_candles[name].copy().reset_index(drop=True)
        if context.empty:
            continue
        context_features = build_features(context)
        context_times = pd.to_datetime(
            context["timestamp"], unit="s", utc=True, errors="coerce"
        )
        selected = context_features.reindex(columns=CONTEXT_FEATURE_COLUMNS).copy()
        selected.insert(0, "timestamp", context_times)
        selected = selected.dropna(subset=["timestamp"]).sort_values("timestamp")
        selected = selected.rename(
            columns={column: f"ctx_{name}_{column}" for column in CONTEXT_FEATURE_COLUMNS}
        )
        aligned = pd.merge_asof(
            left,
            selected,
            on="timestamp",
            direction="backward",
            allow_exact_matches=True,
        )
        block = aligned.drop(columns=["timestamp"]).reset_index(drop=True)
        context_blocks.append(block)

    if context_blocks:
        result = pd.concat([result.reset_index(drop=True), *context_blocks], axis=1)
    return result.replace([np.inf, -np.inf], np.nan)
