"""All the calculations behind the dashboard, kept separate from the API so they
are easy to test and for the CS side to improve (swap in a real model here)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Event:
    key: str
    start_year: int
    end_year: int

    @property
    def baseline_years(self) -> list[int]:
        """The 3 years before the event starts."""
        return [self.start_year - 3, self.start_year - 2, self.start_year - 1]


EVENTS = {
    e.key: e
    for e in (
        Event("1997-98", 1997, 1998),
        Event("2015-16", 2015, 2016),
        Event("2023-24", 2023, 2024),
    )
}

ECOSYSTEMS = ("irrigated", "rainfed")


def oni_status(anomaly: float) -> str:
    if anomaly >= 0.5:
        return "El Niño"
    if anomaly <= -0.5:
        return "La Niña"
    return "Neutral"


def latest_oni(oni: pd.DataFrame) -> dict:
    order = {s: i for i, s in enumerate(["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"])}
    o = oni.assign(_o=oni["season"].map(order)).sort_values(["year", "_o"])
    last = o.iloc[-1]
    return {"season": last["season"], "year": int(last["year"]), "anomaly": float(last["anomaly"]), "status": oni_status(last["anomaly"])}


def event_peak_oni(oni: pd.DataFrame, event: Event) -> float:
    mask = oni["year"].between(event.start_year, event.end_year)
    return float(oni.loc[mask, "anomaly"].max())


def production_change(prod: pd.DataFrame, event: Event, region: str | None = None) -> dict[str, float | None]:
    """% change in production in the event's end year vs. the baseline average,
    per ecosystem. Returns None for an ecosystem with no data."""
    df = prod if region is None else prod[prod["region_code"] == region]
    out: dict[str, float | None] = {}
    for eco in ECOSYSTEMS:
        d = df[df["ecosystem"] == eco].groupby("year")["production_mt"].sum()
        base = d.reindex(event.baseline_years).dropna()
        if base.empty or event.end_year not in d.index or base.mean() == 0:
            out[eco] = None
            continue
        out[eco] = round(float((d[event.end_year] / base.mean() - 1) * 100), 1)
    return out


def rice_price_change(prices: pd.DataFrame, event: Event) -> dict:
    """Peak monthly price during the event vs. the average of the 12 months before it."""
    p = prices.copy()
    p["t"] = p["year"] * 12 + p["month"]
    start_t = event.start_year * 12 + 1
    before = p[(p["t"] >= start_t - 12) & (p["t"] < start_t)]["price_php_per_kg"]
    during = p[p["year"].between(event.start_year, event.end_year)]["price_php_per_kg"]
    if before.empty or during.empty:
        return {"change_php": None, "change_pct": None}
    b, peak = float(before.mean()), float(during.max())
    return {"change_php": round(peak - b, 2), "change_pct": round((peak / b - 1) * 100, 1)}


def oni_vs_rainfed_correlation(oni: pd.DataFrame, prod: pd.DataFrame) -> dict:
    """Pearson r between each event's ONI peak and each region's rainfed change."""
    xs, ys = [], []
    for event in EVENTS.values():
        peak = event_peak_oni(oni, event)
        for region in prod["region_code"].unique():
            change = production_change(prod, event, region)["rainfed"]
            if change is not None:
                xs.append(peak)
                ys.append(change)
    if len(xs) < 3 or np.std(xs) == 0 or np.std(ys) == 0:
        return {"r": None, "n": len(xs)}
    return {"r": round(float(np.corrcoef(xs, ys)[0, 1]), 2), "n": len(xs)}


def _norm(s: pd.Series) -> pd.Series:
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng else s * 0


def regional_risk(prod: pd.DataFrame, rainfall: pd.DataFrame, regions: pd.DataFrame) -> list[dict]:
    """Simple, explainable risk score (0-100) per region:
        40% rainfed share of palay area (latest year)
        30% rainfall deficit over the last 3 months on record
        30% average rainfed loss in past El Niño events
    The CS teammate can replace this with a trained model later."""
    latest = prod["year"].max()
    area = prod[prod["year"] == latest].pivot_table(index="region_code", columns="ecosystem", values="area_ha", aggfunc="sum")
    rainfed_share = (area["rainfed"] / area.sum(axis=1) * 100).rename("rainfed_share")

    r = rainfall.copy()
    r["t"] = r["year"] * 12 + r["month"]
    last3 = r[r["t"] > r["t"].max() - 3].groupby("region_code")[["rainfall_mm", "normal_mm"]].sum()
    rain_vs_normal = ((last3["rainfall_mm"] / last3["normal_mm"] - 1) * 100).rename("rain_vs_normal")

    losses = {
        reg: np.mean([c for e in EVENTS.values() if (c := production_change(prod, e, reg)["rainfed"]) is not None] or [np.nan])
        for reg in rainfed_share.index
    }
    past_loss = pd.Series(losses, name="past_rainfed_change")

    df = pd.concat([rainfed_share, rain_vs_normal, past_loss], axis=1).dropna()
    score = 0.4 * _norm(df["rainfed_share"]) + 0.3 * _norm(-df["rain_vs_normal"]) + 0.3 * _norm(-df["past_rainfed_change"])
    df["score"] = (score * 100).round()
    df["level"] = pd.cut(df["score"], bins=[-1, 33, 66, 101], labels=["Low", "Moderate", "High"]).astype(str)

    names = regions.set_index("code")["name"]
    df = df.sort_values("score", ascending=False)
    return [
        {
            "region_code": code,
            "region_name": names.get(code, code),
            "score": int(row["score"]),
            "level": row["level"],
            "rainfed_share": round(float(row["rainfed_share"]), 1),
            "rain_vs_normal": round(float(row["rain_vs_normal"]), 1),
            "past_rainfed_change": round(float(row["past_rainfed_change"]), 1),
        }
        for code, row in df.iterrows()
    ]
