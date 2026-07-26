from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .backtest import BacktestConfig, run_backtest
from .features import build_features, build_multimarket_features, normalize_candles
from .kraken import KrakenClient
from .model import ProbabilityModel, write_manifest



DEFAULT_CONTEXT_MARKETS = [
    {"pair": "XBTEUR", "interval": 60},
    {"pair": "XBTEUR", "interval": 240},
    {"pair": "XBTUSD", "interval": 15},
    {"pair": "XBTUSD", "interval": 60},
    {"pair": "XBTUSD", "interval": 240},
    {"pair": "ETHEUR", "interval": 15},
    {"pair": "ETHEUR", "interval": 60},
    {"pair": "ETHEUR", "interval": 240},
    {"pair": "ETHUSD", "interval": 15},
    {"pair": "ETHUSD", "interval": 60},
    {"pair": "ETHUSD", "interval": 240},
]


def _find_csv(data_dir: str | Path, pair: str, interval: int) -> Path:
    root = Path(data_dir)
    candidates = [
        root / f"{pair}_{interval}.csv",
        root / f"{pair}{interval}.csv",
        root / f"{pair.replace('XBT', 'BTC')}_{interval}.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    normalized_pair = pair.replace("/", "").replace("-", "").upper()
    for candidate in root.glob("*.csv"):
        stem = candidate.stem.replace("_", "").replace("-", "").upper()
        if normalized_pair in stem and stem.endswith(str(interval)):
            return candidate
    raise FileNotFoundError(f"No CSV found for {pair} {interval}m in {root}")


def load_multimarket_data(
    data_dir: str | Path, primary_pair: str, primary_interval: int
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[dict[str, Any]]]:
    primary_path = _find_csv(data_dir, primary_pair, primary_interval)
    primary = load_kraken_csv(str(primary_path), primary_interval)
    contexts: dict[str, pd.DataFrame] = {}
    specs: list[dict[str, Any]] = []
    for spec in DEFAULT_CONTEXT_MARKETS:
        pair = str(spec["pair"])
        interval = int(spec["interval"])
        if pair == primary_pair and interval == primary_interval:
            continue
        path = _find_csv(data_dir, pair, interval)
        key = f"{pair}_{interval}"
        contexts[key] = load_kraken_csv(str(path), interval)
        specs.append({"name": key, "pair": pair, "interval": interval})
    return primary, contexts, specs


def load_kraken_csv(path: str, interval_minutes: int) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None)
    if raw.shape[1] < 7:
        raise ValueError("Expected Kraken OHLCVT CSV with at least 7 columns")
    columns = ["timestamp", "open", "high", "low", "close", "volume", "trades"]
    raw = raw.iloc[:, :7]
    raw.columns = columns
    raw["vwap"] = raw["close"]
    return normalize_candles(raw, interval_minutes)


def load_live_bootstrap(pair: str, interval_minutes: int) -> pd.DataFrame:
    rows = KrakenClient().get_ohlc(pair, interval_minutes)
    return normalize_candles(pd.DataFrame(rows), interval_minutes)


def make_target(
    candles: pd.DataFrame,
    features: pd.DataFrame,
    horizon: int,
    round_trip_cost: float,
) -> pd.Series:
    labels = np.full(len(candles), np.nan)
    for i in range(len(candles) - horizon - 1):
        entry = float(candles.iloc[i + 1]["open"])
        atr_fraction = float(features.iloc[i].get("atr_pct_14", np.nan))
        if np.isnan(atr_fraction) or entry <= 0:
            continue
        barrier = max(round_trip_cost * 1.5, atr_fraction * 1.1)
        upper = entry * (1 + barrier)
        lower = entry * (1 - barrier)
        label: float | None = None
        for j in range(i + 1, i + horizon + 1):
            high = float(candles.iloc[j]["high"])
            low = float(candles.iloc[j]["low"])
            if low <= lower and high >= upper:
                label = 0.0  # pessimistic ordering
                break
            if low <= lower:
                label = 0.0
                break
            if high >= upper:
                label = 1.0
                break
        if label is None:
            final_close = float(candles.iloc[i + horizon]["close"])
            label = float(final_close / entry - 1 > round_trip_cost)
        labels[i] = label
    return pd.Series(labels, index=candles.index, name="target")


def build_estimator(random_state: int = 42) -> VotingClassifier:
    logistic = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.25,
                    max_iter=2500,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )
    gradient = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.035,
                    max_iter=260,
                    max_leaf_nodes=15,
                    min_samples_leaf=40,
                    l2_regularization=1.5,
                    random_state=random_state,
                ),
            ),
        ]
    )
    forest = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=280,
                    max_depth=8,
                    min_samples_leaf=35,
                    max_features=0.65,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )
    return VotingClassifier(
        estimators=[("logistic", logistic), ("gradient", gradient), ("forest", forest)],
        voting="soft",
        weights=[1, 2, 1],
        n_jobs=-1,
    )


def calibrate(base_model: VotingClassifier, x: pd.DataFrame, y: pd.Series) -> Any:
    probability = base_model.predict_proba(x)[:, 1]
    labels = y.astype(int)
    if labels.nunique() < 2:
        calibrator = DummyClassifier(strategy="constant", constant=int(labels.iloc[0]))
    else:
        calibrator = LogisticRegression(C=1.0, max_iter=1000)
    calibrator.fit(probability.reshape(-1, 1), labels)
    return calibrator


def calibrated_probability(
    base_model: VotingClassifier, calibrator: Any, x: pd.DataFrame
) -> np.ndarray:
    raw = base_model.predict_proba(x)[:, 1]
    probabilities = calibrator.predict_proba(raw.reshape(-1, 1))
    if probabilities.shape[1] == 1:
        only_class = int(calibrator.classes_[0])
        return probabilities[:, 0] if only_class == 1 else 1.0 - probabilities[:, 0]
    positive_index = list(calibrator.classes_).index(1)
    return probabilities[:, positive_index]


def choose_thresholds(
    candles: pd.DataFrame,
    features: pd.DataFrame,
    probabilities: np.ndarray,
    config: BacktestConfig,
) -> tuple[float, float, dict[str, Any]]:
    best_score = -float("inf")
    best: tuple[float, float, dict[str, Any]] | None = None
    for buy_threshold in np.arange(0.55, 0.76, 0.025):
        for sell_threshold in np.arange(0.30, 0.51, 0.025):
            if sell_threshold >= buy_threshold - 0.05:
                continue
            result = run_backtest(
                candles,
                features,
                probabilities,
                float(round(buy_threshold, 3)),
                float(round(sell_threshold, 3)),
                config,
            )
            summary = result.summary()
            if result.trade_count < 8:
                continue
            score = (
                result.annualized_sharpe
                + 2.0 * result.net_return
                + 0.5 * min(result.profit_factor, 3.0)
                + result.max_drawdown
            )
            if score > best_score:
                best_score = score
                best = (float(round(buy_threshold, 3)), float(round(sell_threshold, 3)), summary)
    if best is None:
        return 0.65, 0.40, {"warning": "No threshold pair produced enough trades"}
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Kraken trading model")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", help="Path to one extracted Kraken OHLCVT CSV")
    source.add_argument("--data-dir", help="Directory containing the 12 multimarket CSV files")
    source.add_argument("--live-bootstrap", action="store_true", help="Use only Kraken's recent 720 candles; paper testing only")
    parser.add_argument("--pair", default="XBTEUR")
    parser.add_argument("--interval", type=int, default=15)
    parser.add_argument("--output", default="model.joblib")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--fee-rate", type=float, default=0.0026)
    parser.add_argument("--slippage-bps", type=float, default=8.0)
    parser.add_argument("--approve-live", action="store_true")
    args = parser.parse_args()

    context_specs: list[dict[str, Any]] = []
    if args.data_dir:
        candles, context_frames, context_specs = load_multimarket_data(
            args.data_dir, args.pair, args.interval
        )
        features = build_multimarket_features(candles, context_frames)
    elif args.csv:
        candles = load_kraken_csv(args.csv, args.interval)
        features = build_features(candles)
    else:
        candles = load_live_bootstrap(args.pair, args.interval)
        features = build_features(candles)
    round_trip_cost = 2 * args.fee_rate + 2 * args.slippage_bps / 10_000
    target = make_target(candles, features, args.horizon, round_trip_cost)

    valid = target.notna() & features.notna().sum(axis=1).ge(len(features.columns) * 0.85)
    candles = candles.loc[valid].reset_index(drop=True)
    features = features.loc[valid].reset_index(drop=True)
    target = target.loc[valid].astype(int).reset_index(drop=True)

    if len(candles) < 500:
        raise RuntimeError("At least 500 usable candles are required")

    n = len(candles)
    train_boundary = int(n * 0.60)
    test_boundary = int(n * 0.78)
    purge = max(args.horizon, 1)

    # Purge the target look-ahead horizon at every split boundary. Without
    # this gap, labels immediately before a boundary would use prices from
    # the next period and leak future information into training/calibration.
    train_end = train_boundary - purge
    calibration_start = train_boundary
    calibration_end = test_boundary - purge
    test_start = test_boundary

    if train_end <= 0 or calibration_end <= calibration_start or test_start >= n:
        raise RuntimeError("Not enough candles for purged chronological splits")

    x_train, y_train = features.iloc[:train_end], target.iloc[:train_end]
    x_cal = features.iloc[calibration_start:calibration_end]
    y_cal = target.iloc[calibration_start:calibration_end]
    x_test, y_test = features.iloc[test_start:], target.iloc[test_start:]

    estimator = build_estimator()
    estimator.fit(x_train, y_train)
    calibrator = calibrate(estimator, x_cal, y_cal)

    config = BacktestConfig(
        interval_minutes=args.interval,
        fee_rate=args.fee_rate,
        slippage_bps=args.slippage_bps,
    )
    cal_prob = calibrated_probability(estimator, calibrator, x_cal)
    buy_threshold, sell_threshold, calibration_metrics = choose_thresholds(
        candles.iloc[calibration_start:calibration_end].reset_index(drop=True),
        features.iloc[calibration_start:calibration_end].reset_index(drop=True),
        cal_prob,
        config,
    )

    test_prob = calibrated_probability(estimator, calibrator, x_test)
    test_result = run_backtest(
        candles.iloc[test_start:].reset_index(drop=True),
        features.iloc[test_start:].reset_index(drop=True),
        test_prob,
        buy_threshold,
        sell_threshold,
        config,
    )
    test_metrics = test_result.summary()

    metric_gate = (
        len(candles) >= 50_000
        and test_result.trade_count >= 30
        and test_result.net_return > 0
        and test_result.annualized_sharpe >= 0.35
        and test_result.profit_factor >= 1.05
        and test_result.max_drawdown >= -0.25
    )
    approved_for_live = bool(args.approve_live and (args.csv or args.data_dir) and metric_gate)

    # Final artifact: train the base model on all pre-test data and calibrate on the final 15% of it.
    pretest_end = test_boundary
    final_calibration_start = int(pretest_end * 0.82)
    final_base_end = final_calibration_start - purge
    final_calibration_end = pretest_end - purge
    final_estimator = build_estimator()
    final_estimator.fit(features.iloc[:final_base_end], target.iloc[:final_base_end])
    final_calibrator = calibrate(
        final_estimator,
        features.iloc[final_calibration_start:final_calibration_end],
        target.iloc[final_calibration_start:final_calibration_end],
    )

    trained_at = datetime.now(timezone.utc).isoformat()
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model = ProbabilityModel(
        estimator=final_estimator,
        calibrator=final_calibrator,
        feature_names=list(features.columns),
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        version=version,
        pair=args.pair,
        interval_minutes=args.interval,
        approved_for_live=approved_for_live,
        context_markets=context_specs,
        trained_at=trained_at,
        metrics={
            "usable_candles": len(candles),
            "positive_class_rate": float(target.mean()),
            "purged_bars_per_boundary": purge,
            "calibration": calibration_metrics,
            "test": test_metrics,
            "approval_metric_gate": metric_gate,
            "approve_live_requested": bool(args.approve_live),
        },
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output, compress=3)
    manifest = write_manifest(model, output)

    print(json.dumps({
        "model": str(output),
        "manifest": str(manifest),
        "approved_for_live": approved_for_live,
        "buy_threshold": buy_threshold,
        "sell_threshold": sell_threshold,
        "test": test_metrics,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
