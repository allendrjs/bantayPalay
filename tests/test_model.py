"""Tests for the CS track. Run:  pytest -q"""
from datetime import date

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from model import advice
from model.config import FEATURES, TARGET
from model.features import build_training_table, palay_by_region_year
from model.predict import level_for
from model.train import time_split


def tiny_prod():
    rows = []
    for year, rf in [(2010, 100), (2011, 100), (2012, 100), (2013, 70)]:
        rows += [
            {"region_code": "X", "year": year, "ecosystem": "rainfed", "production_mt": rf, "area_ha": 30},
            {"region_code": "X", "year": year, "ecosystem": "irrigated", "production_mt": 200, "area_ha": 70},
        ]
    return pd.DataFrame(rows)


# ---------- features ----------

def test_target_uses_only_the_three_previous_years():
    p = palay_by_region_year(tiny_prod()).set_index("year")
    assert p.loc[2013, TARGET] == pytest.approx(-30.0)
    assert pd.isna(p.loc[2012, TARGET])  # not enough history yet


def test_rainfed_share_comes_from_the_previous_year():
    p = palay_by_region_year(tiny_prod()).set_index("year")
    assert p.loc[2013, "rainfed_share"] == pytest.approx(30.0)
    assert pd.isna(p.loc[2010, "rainfed_share"])


def test_training_table_has_every_feature_and_no_gaps():
    from app.db import read_table
    t = build_training_table(read_table("oni"), read_table("palay_production"), read_table("rainfall"))
    assert list(t.columns) == ["region_code", "year", *FEATURES, TARGET]
    assert not t.isna().any().any()


def test_time_split_never_overlaps():
    t = pd.DataFrame({"year": range(2000, 2030)})
    train, test = time_split(t, 2014)
    assert train["year"].max() == 2014 and test["year"].min() == 2015


# ---------- risk levels ----------

@pytest.mark.parametrize("change,level", [(-15, "High"), (-10, "High"), (-6, "Moderate"), (-4, "Moderate"), (-1, "Low"), (5, "Low")])
def test_level_thresholds(change, level):
    assert level_for(change) == level


# ---------- advisor rules ----------

D = date(2026, 10, 1)

@pytest.mark.parametrize("prev,cur,last,today,send", [
    (None, "Low", None, D, False),               # nothing to warn about
    (None, "High", None, D, True),               # first reading, high
    ("Low", "Moderate", None, D, True),          # risk went up
    ("Moderate", "High", D, D, True),            # up again, even the same day
    ("Moderate", "Moderate", D, D, False),       # no repeat
    ("High", "Moderate", D, D, False),           # easing, not yet clear
    ("Moderate", "Low", D, D, True),             # all clear
    ("High", "High", date(2026, 9, 25), D, False),  # reminded 6 days ago
    ("High", "High", date(2026, 8, 31), D, True),   # 31 days ago → remind
])
def test_should_alert(prev, cur, last, today, send):
    assert advice.should_alert(prev, cur, last, today) is send


@pytest.mark.parametrize("lang", advice.LANGS)
@pytest.mark.parametrize("level", ["High", "Moderate", "Low"])
def test_messages_fit_one_sms(level, lang):
    # 30 characters is room for a long municipality name
    assert len(advice.alert_text(level, lang, "X" * 30)) <= advice.SMS_LIMIT
    assert len(advice.tips_text(level, lang)) <= advice.SMS_LIMIT


# ---------- API (needs `python -m model.train` to have run) ----------

@pytest.fixture(scope="module")
def client():
    from app.main import app
    return TestClient(app)


def test_forecast_and_what_if(client):
    base = client.get("/api/forecast").json()
    assert {r["level"] for r in base["regions"]} <= {"Low", "Moderate", "High"}
    strong = client.get("/api/forecast", params={"oni_max": 2.5, "rain_dev": -30}).json()
    mean = lambda d: sum(r["predicted_change"] for r in d["regions"]) / len(d["regions"])
    assert mean(strong) < mean(base)  # a strong El Niño should predict a bigger loss


def test_advice_endpoint(client):
    r = client.get("/api/advice", params={"region": "III", "lang": "ceb", "area": "Tarlac"})
    assert r.status_code == 200
    body = r.json()
    assert "Tarlac" in body["alert"] and body["validated"] is False
    assert client.get("/api/advice", params={"region": "III", "lang": "fr"}).status_code == 400


def test_risk_includes_model_columns(client):
    row = client.get("/api/regions/risk").json()[0]
    assert "model_level" in row and "model_predicted_change" in row
