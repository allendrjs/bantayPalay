"""Turn the raw tables into one row per region per year that the model can learn from.

Every feature for year t uses only information from year t and earlier, and the
target compares year t with the 3 years BEFORE it, so nothing from the future
leaks into training.
"""
from __future__ import annotations

import pandas as pd

from model.config import FEATURES, SEASON_MONTHS, TARGET


def oni_by_year(oni: pd.DataFrame) -> pd.DataFrame:
    g = oni.groupby("year")["anomaly"]
    out = pd.DataFrame({"oni_max": g.max(), "oni_mean": g.mean()})
    out["oni_max_prev"] = out["oni_max"].shift(1)
    return out.reset_index()


def rain_by_region_year(rainfall: pd.DataFrame) -> pd.DataFrame:
    season = rainfall[rainfall["month"].isin(SEASON_MONTHS)]
    s = season.groupby(["region_code", "year"])[["rainfall_mm", "normal_mm"]].sum()
    s["rain_dev"] = (s["rainfall_mm"] / s["normal_mm"] - 1) * 100
    s = s[["rain_dev"]].reset_index().sort_values(["region_code", "year"])
    s["rain_dev_prev"] = s.groupby("region_code")["rain_dev"].shift(1)
    return s


def palay_by_region_year(prod: pd.DataFrame) -> pd.DataFrame:
    p = prod.pivot_table(index=["region_code", "year"], columns="ecosystem", values=["production_mt", "area_ha"], aggfunc="sum")
    p.columns = [f"{a}_{b}" for a, b in p.columns]
    p = p.reset_index().sort_values(["region_code", "year"])
    g = p.groupby("region_code")
    # Target: this year's rainfed production vs. the mean of the 3 years before it.
    baseline = g["production_mt_rainfed"].transform(lambda s: s.shift(1).rolling(3).mean())
    p[TARGET] = (p["production_mt_rainfed"] / baseline - 1) * 100
    share = p["area_ha_rainfed"] / (p["area_ha_rainfed"] + p["area_ha_irrigated"]) * 100
    p["rainfed_share"] = share.groupby(p["region_code"]).shift(1)
    return p[["region_code", "year", TARGET, "rainfed_share"]]


def build_training_table(oni: pd.DataFrame, prod: pd.DataFrame, rainfall: pd.DataFrame) -> pd.DataFrame:
    """One row per region-year with all FEATURES and the TARGET; incomplete rows dropped."""
    df = (
        palay_by_region_year(prod)
        .merge(oni_by_year(oni), on="year", how="inner")
        .merge(rain_by_region_year(rainfall), on=["region_code", "year"], how="inner")
    )
    return df[["region_code", "year", *FEATURES, TARGET]].dropna().reset_index(drop=True)


def build_latest_features(oni: pd.DataFrame, prod: pd.DataFrame, rainfall: pd.DataFrame) -> pd.DataFrame:
    """Features for the most recent year that has ONI and rainfall, per region —
    the input for a forecast. Production for that year is not needed."""
    year = int(min(oni["year"].max(), rainfall["year"].max()))
    o = oni_by_year(oni)
    o = o[o["year"] == year]
    r = rain_by_region_year(rainfall)
    r = r[r["year"] == year]
    p = prod.pivot_table(index=["region_code", "year"], columns="ecosystem", values="area_ha", aggfunc="sum").reset_index()
    last_prod_year = p[p["year"] < year]["year"].max()
    p = p[p["year"] == last_prod_year]
    p["rainfed_share"] = p["rainfed"] / (p["rainfed"] + p["irrigated"]) * 100
    df = r.assign(year=year).merge(o, on="year").merge(p[["region_code", "rainfed_share"]], on="region_code")
    return df[["region_code", "year", *FEATURES]].dropna().reset_index(drop=True)
