from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import boto3
import joblib
import numpy as np
import pandas as pd


@dataclass
class ProbabilityModel:
    estimator: Any
    calibrator: Any
    feature_names: list[str]
    buy_threshold: float
    sell_threshold: float
    version: str
    pair: str
    interval_minutes: int
    approved_for_live: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)
    trained_at: str = ""
    context_markets: list[dict[str, Any]] = field(default_factory=list)

    def predict_probability(self, features: pd.DataFrame) -> np.ndarray:
        ordered = features.reindex(columns=self.feature_names)
        base_probability = self.estimator.predict_proba(ordered)[:, 1]
        probabilities = self.calibrator.predict_proba(base_probability.reshape(-1, 1))
        if probabilities.shape[1] == 1:
            only_class = int(self.calibrator.classes_[0])
            calibrated = probabilities[:, 0] if only_class == 1 else 1.0 - probabilities[:, 0]
        else:
            positive_index = list(self.calibrator.classes_).index(1)
            calibrated = probabilities[:, positive_index]
        return calibrated


class ModelStore:
    def __init__(self, bucket: str, key: str):
        self.bucket = bucket
        self.key = key
        self._cached_model: ProbabilityModel | None = None
        self._cached_etag: str = ""
        self._s3 = boto3.client("s3") if bucket else None

    def load(self) -> ProbabilityModel:
        if not self.bucket:
            local_path = os.getenv("LOCAL_MODEL_PATH", "/opt/model/model.joblib")
            if not Path(local_path).exists():
                raise RuntimeError("No model configured. Set MODEL_BUCKET or LOCAL_MODEL_PATH.")
            if self._cached_model is None:
                self._cached_model = joblib.load(local_path)
            return self._cached_model

        assert self._s3 is not None
        head = self._s3.head_object(Bucket=self.bucket, Key=self.key)
        etag = str(head.get("ETag", "")).strip('"')
        if self._cached_model is not None and etag == self._cached_etag:
            return self._cached_model

        local_path = "/tmp/trading-model.joblib"
        self._s3.download_file(self.bucket, self.key, local_path)
        model = joblib.load(local_path)
        if not isinstance(model, ProbabilityModel):
            raise TypeError("Model artifact is not a ProbabilityModel")
        self._cached_model = model
        self._cached_etag = etag
        return model


def artifact_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(model: ProbabilityModel, model_path: str | Path) -> Path:
    path = Path(model_path)
    manifest_path = path.with_suffix(path.suffix + ".json")
    manifest = {
        "version": model.version,
        "pair": model.pair,
        "interval_minutes": model.interval_minutes,
        "approved_for_live": model.approved_for_live,
        "buy_threshold": model.buy_threshold,
        "sell_threshold": model.sell_threshold,
        "trained_at": model.trained_at,
        "feature_count": len(model.feature_names),
        "context_markets": model.context_markets,
        "metrics": model.metrics,
        "sha256": artifact_sha256(path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest_path
