"""Run:  pytest -q"""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import analysis


# ---------- analysis unit tests (tiny hand-made data, easy to check by hand) ----------

def make_prod():
    rows = []
    for year, irr, rf in [(2012, 100, 50), (2013, 100, 50), (2014, 100, 50), (2016, 90, 25)]:
        rows += [
            {"region_code": "X", "year": year, "ecosystem": "irrigated", "production_mt": irr, "area_ha": 20},
            {"region_code": "X", "year": year, "ecosystem": "rainfed", "production_mt": rf, "area_ha": 20},
        ]
    return pd.DataFrame(rows)


def test_production_change_vs_baseline():
    change = analysis.production_change(make_prod(), analysis.EVENTS["2015-16"])
    assert change == {"irrigated": -10.0, "rainfed": -50.0}


def test_production_change_missing_data_returns_none():
    change = analysis.production_change(make_prod(), analysis.EVENTS["1997-98"])
    assert change == {"irrigated": None, "rainfed": None}


@pytest.mark.parametrize("value,status", [(0.5, "El Niño"), (2.1, "El Niño"), (0.0, "Neutral"), (-0.5, "La Niña")])
def test_oni_status_thresholds(value, status):
    assert analysis.oni_status(value) == status


def test_rice_price_change_peak_vs_prior_year():
    prices = pd.DataFrame(
        [{"year": 2014, "month": m, "price_php_per_kg": 40.0} for m in range(1, 13)]
        + [{"year": y, "month": m, "price_php_per_kg": 44.0 if (y, m) == (2016, 3) else 41.0} for y in (2015, 2016) for m in range(1, 13)]
    )
    assert analysis.rice_price_change(prices, analysis.EVENTS["2015-16"]) == {"change_php": 4.0, "change_pct": 10.0}


# ---------- API tests (use the loaded database) ----------

@pytest.fixture(scope="module")
def client():
    from app.main import app
    return TestClient(app)


def test_summary_endpoint(client):
    r = client.get("/api/summary", params={"event": "2015-16"})
    assert r.status_code == 200
    body = r.json()
    assert set(body["production_change_pct"]) == {"irrigated", "rainfed"}
    assert body["latest_oni"]["status"] in {"El Niño", "Neutral", "La Niña"}


def test_unknown_event_and_region_are_404(client):
    assert client.get("/api/summary", params={"event": "1900-01"}).status_code == 404
    assert client.get("/api/summary", params={"region": "ZZZ"}).status_code == 404


def test_risk_is_sorted_and_scored(client):
    risk = client.get("/api/regions/risk").json()
    scores = [r["score"] for r in risk]
    assert scores == sorted(scores, reverse=True)
    assert all(0 <= s <= 100 for s in scores)
    assert {r["level"] for r in risk} <= {"Low", "Moderate", "High"}


def test_dashboard_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "El Niño Watch PH" in r.text
