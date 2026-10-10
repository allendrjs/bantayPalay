"""Use the saved model: forecast rainfed palay change per region and turn it into a risk level."""
from __future__ import annotations

from functools import lru_cache

import joblib
import pandas as pd

from model.config import ARTIFACT, FEATURES, HIGH_AT, MODERATE_AT
from model.features import build_latest_features


def level_for(change: float) -> str:
    if change <= HIGH_AT:
        return "High"
    if change <= MODERATE_AT:
        return "Moderate"
    return "Low"


@lru_cache(maxsize=1)
def _load(mtime: float) -> dict:
    return joblib.load(ARTIFACT)


def load_model() -> dict | None:
    """The saved model bundle, or None if `python -m model.train` hasn't been run."""
    if not ARTIFACT.exists():
        return None
    return _load(ARTIFACT.stat().st_mtime)


def forecast(oni: pd.DataFrame, prod: pd.DataFrame, rainfall: pd.DataFrame,
             oni_max: float | None = None, rain_dev: float | None = None) -> list[dict] | None:
    """Predicted rainfed change for each region for the latest year with ONI + rainfall.

    oni_max / rain_dev let you run a what-if (e.g. a forecast ONI peak of 2.0 from NOAA)
    instead of the observed values. Returns None when no model is saved."""
    bundle = load_model()
    if bundle is None:
        return None
    X = build_latest_features(oni, prod, rainfall)
    if oni_max is not None:
        X["oni_max"] = oni_max
    if rain_dev is not None:
        X["rain_dev"] = rain_dev
    X["predicted_change"] = bundle["model"].predict(X[FEATURES])
    return [
        {
            "region_code": r.region_code,
            "year": int(r.year),
            "predicted_change": round(float(r.predicted_change), 1),
            "level": level_for(r.predicted_change),
        }
        for r in X.itertuples()
    ]
