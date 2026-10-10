"""El Niño Watch PH – API + dashboard server.

Run:  uvicorn app.main:app --reload
Then open http://127.0.0.1:8000
API docs (auto-generated): http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

from app import analysis
from app.db import read_table
from model import advice, predict
from model.config import HIGH_AT, METRICS, MODERATE_AT

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="El Niño Watch PH API", version="0.1.0")


def _event(key: str) -> analysis.Event:
    if key not in analysis.EVENTS:
        raise HTTPException(404, f"Unknown event '{key}'. Use one of {list(analysis.EVENTS)}")
    return analysis.EVENTS[key]


def _check_region(region: str | None) -> None:
    if region and region not in set(read_table("regions")["code"]):
        raise HTTPException(404, f"Unknown region '{region}'")


@app.get("/api/meta")
def meta():
    """Which tables came from real files vs. sample data."""
    m = read_table("meta")
    return {
        "using_sample_data": bool((m["source"] == "sample").any()),
        "tables": m.to_dict(orient="records"),
    }


@app.get("/api/regions")
def regions():
    return read_table("regions").to_dict(orient="records")


@app.get("/api/events")
def events():
    return [{"key": e.key, "start_year": e.start_year, "end_year": e.end_year} for e in analysis.EVENTS.values()]


@app.get("/api/oni")
def oni(from_year: int = 1990, to_year: int = 2100):
    df = read_table("oni")
    df = df[df["year"].between(from_year, to_year)]
    return {
        "latest": analysis.latest_oni(read_table("oni")),
        "series": df.to_dict(orient="records"),
    }


@app.get("/api/summary")
def summary(event: str = "2023-24", region: str | None = Query(None, description="Region code, e.g. III")):
    _check_region(region)
    ev = _event(event)
    prod = read_table("palay_production")
    oni_df = read_table("oni")
    return {
        "event": ev.key,
        "region": region,
        "latest_oni": analysis.latest_oni(oni_df),
        "event_peak_oni": analysis.event_peak_oni(oni_df, ev),
        "production_change_pct": analysis.production_change(prod, ev, region),
        "rice_price_change": analysis.rice_price_change(read_table("rice_prices"), ev),
        "oni_vs_rainfed_correlation": analysis.oni_vs_rainfed_correlation(oni_df, prod),
    }


@app.get("/api/production/by-event")
def production_by_event(region: str | None = None):
    _check_region(region)
    prod = read_table("palay_production")
    return [{"event": e.key, **analysis.production_change(prod, e, region)} for e in analysis.EVENTS.values()]


def _forecast(oni_max: float | None = None, rain_dev: float | None = None):
    return predict.forecast(read_table("oni"), read_table("palay_production"), read_table("rainfall"), oni_max, rain_dev)


@app.get("/api/regions/risk")
def regions_risk():
    """Weighted-formula risk per region, plus the model's forecast when a model is saved."""
    risk = analysis.regional_risk(read_table("palay_production"), read_table("rainfall"), read_table("regions"))
    fc = {f["region_code"]: f for f in (_forecast() or [])}
    for r in risk:
        f = fc.get(r["region_code"])
        r["model_predicted_change"] = f["predicted_change"] if f else None
        r["model_level"] = f["level"] if f else None
    return risk


@app.get("/api/model/metrics")
def model_metrics():
    """Evaluation results written by `python -m model.train`."""
    if not METRICS.exists():
        raise HTTPException(404, "No model trained yet. Run: python -m model.train")
    return json.loads(METRICS.read_text())


@app.get("/api/forecast")
def forecast(
    oni_max: float | None = Query(None, description="What-if: ONI peak for the season, e.g. a NOAA forecast of 2.0"),
    rain_dev: float | None = Query(None, description="What-if: growing-season rainfall vs. normal, in %"),
):
    """Predicted rainfed palay change and risk level per region."""
    result = _forecast(oni_max, rain_dev)
    if result is None:
        raise HTTPException(404, "No model trained yet. Run: python -m model.train")
    return {"model": predict.load_model()["name"], "thresholds": {"high_at": HIGH_AT, "moderate_at": MODERATE_AT}, "regions": result}


@app.get("/api/advice")
def advice_for(region: str, lang: str = "tl", area: str | None = Query(None, description="Name shown in the SMS; defaults to the region name")):
    """The SMS alert and TIPS reply a farmer in this region would get right now."""
    _check_region(region)
    if lang not in advice.LANGS:
        raise HTTPException(400, f"lang must be one of {list(advice.LANGS)}")
    fc = {f["region_code"]: f for f in (_forecast() or [])}
    if region not in fc:
        raise HTTPException(404, "No forecast for this region. Train the model first: python -m model.train")
    level = fc[region]["level"]
    name = area or read_table("regions").set_index("code")["name"][region]
    return {
        "region": region,
        "level": level,
        "predicted_change": fc[region]["predicted_change"],
        "alert": advice.alert_text(level, lang, name),
        "tips": advice.tips_text(level, lang),
        "validated": advice.VALIDATED,
    }


# Serve the dashboard last so /api/* routes win.
app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
