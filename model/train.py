"""Train and evaluate the rainfed-palay forecasting models, then save the best one.

Run (after scripts/load_data.py):  python -m model.train

What it does
1. Builds one row per region-year (model/features.py).
2. Splits by TIME: trains on target years up to LAST_TRAIN_YEAR, tests on later years.
3. Compares four models on the test years (MAE, RMSE) — including a naive baseline,
   so we can show the model actually learns something.
4. Checks how well each approach RANKS regions by loss in the test El Niño years,
   against the dashboard's original weighted-formula score.
5. Refits the best model on all years and saves it to model/artifacts/.
6. Writes reports/model_report.md for Chapter 4.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.analysis import EVENTS  # noqa: E402
from app.db import read_table  # noqa: E402
from model.config import ARTIFACT, FEATURES, LAST_TRAIN_YEAR, METRICS, REPORT, TARGET  # noqa: E402
from model.features import build_training_table  # noqa: E402


def candidate_models() -> dict:
    return {
        "Baseline (always predicts the training median)": DummyRegressor(strategy="median"),
        "Linear regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Random forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=42),
        "Gradient boosting": GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, random_state=42),
    }


def time_split(table: pd.DataFrame, last_train_year: int = LAST_TRAIN_YEAR):
    train = table[table["year"] <= last_train_year]
    test = table[table["year"] > last_train_year]
    return train, test


def _norm(s: pd.Series) -> pd.Series:
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng else s * 0


def formula_score(rows: pd.DataFrame, past_loss: pd.Series) -> pd.Series:
    """The dashboard's original weighted formula, using only information known at the time."""
    pl = rows["region_code"].map(past_loss)
    return 0.4 * _norm(rows["rainfed_share"]) + 0.3 * _norm(-rows["rain_dev"]) + 0.3 * _norm(-pl)


def ranking_check(table: pd.DataFrame, fitted: dict) -> list[dict]:
    """For each test El Niño year: does the approach rank the worst-hit regions first?
    Spearman correlation between predicted risk and actual loss (1 = perfect order)."""
    results = []
    event_years = sorted(e.end_year for e in EVENTS.values())
    for year in event_years:
        if year <= LAST_TRAIN_YEAR:
            continue
        rows = table[table["year"] == year]
        if len(rows) < 3:
            continue
        earlier = [y for y in event_years if y < year]
        past_loss = table[table["year"].isin(earlier)].groupby("region_code")[TARGET].mean()
        actual_loss = -rows[TARGET]
        entry = {"year": year, "regions": len(rows)}
        entry["Weighted formula (dashboard v1)"] = round(float(spearmanr(formula_score(rows, past_loss), actual_loss).statistic), 2)
        for name, m in fitted.items():
            if name.startswith("Baseline"):
                continue  # a constant prediction has no ranking
            entry[name] = round(float(spearmanr(-m.predict(rows[FEATURES]), actual_loss).statistic), 2)
        results.append(entry)
    return results


def evaluate(table: pd.DataFrame) -> tuple[list[dict], dict, pd.DataFrame]:
    train, test = time_split(table)
    if train.empty or test.empty:
        raise SystemExit(f"Need data on both sides of {LAST_TRAIN_YEAR}: train={len(train)} test={len(test)} rows")
    scores, fitted = [], {}
    for name, model in candidate_models().items():
        model.fit(train[FEATURES], train[TARGET])
        pred = model.predict(test[FEATURES])
        elnino = test["year"].isin([e.end_year for e in EVENTS.values()])
        scores.append({
            "model": name,
            "mae": round(float(mean_absolute_error(test[TARGET], pred)), 2),
            "rmse": round(float(np.sqrt(mean_squared_error(test[TARGET], pred))), 2),
            "mae_elnino_years": round(float(mean_absolute_error(test.loc[elnino, TARGET], pred[elnino.values])), 2) if elnino.any() else None,
        })
        fitted[name] = model
    return scores, fitted, test


def main() -> None:
    meta = read_table("meta")
    sample = bool((meta["source"] == "sample").any())
    table = build_training_table(read_table("oni"), read_table("palay_production"), read_table("rainfall"))
    train, test = time_split(table)
    print(f"{len(table)} region-years · train {train['year'].min()}–{train['year'].max()} ({len(train)}) · test {test['year'].min()}–{test['year'].max()} ({len(test)})")
    if sample:
        print("WARNING: trained on SAMPLE data — these results only show the pipeline works.")

    scores, fitted, test = evaluate(table)
    for s in scores:
        print(f"  {s['model']:<48} MAE {s['mae']:>6}  RMSE {s['rmse']:>6}  MAE in El Niño years {s['mae_elnino_years']}")
    ranking = ranking_check(table, fitted)

    best = min((s for s in scores if not s["model"].startswith("Baseline")), key=lambda s: s["mae"])
    best_model = fitted[best["model"]]
    imp = permutation_importance(best_model, test[FEATURES], test[TARGET], n_repeats=20, random_state=42, scoring="neg_mean_absolute_error")
    importance = sorted(
        ({"feature": f, "mae_increase": round(float(m), 2)} for f, m in zip(FEATURES, imp.importances_mean)),
        key=lambda d: -d["mae_increase"],
    )

    # Refit the chosen model on every year before saving it for use.
    final = candidate_models()[best["model"]].fit(table[FEATURES], table[TARGET])
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    trained_at = datetime.now(timezone.utc).isoformat()
    joblib.dump({"model": final, "name": best["model"], "features": FEATURES, "trained_on_sample_data": sample,
                 "years": [int(table["year"].min()), int(table["year"].max())], "trained_at": trained_at}, ARTIFACT)
    metrics = {"chosen_model": best["model"], "trained_on_sample_data": sample, "last_train_year": LAST_TRAIN_YEAR,
               "rows": {"train": len(train), "test": len(test)}, "scores": scores, "ranking": ranking,
               "feature_importance": importance, "trained_at": trained_at}
    METRICS.write_text(json.dumps(metrics, indent=2))
    write_report(metrics)
    print(f"Chose {best['model']}. Saved {ARTIFACT.relative_to(ARTIFACT.parents[2])} and {REPORT.relative_to(REPORT.parents[1])}")


def write_report(m: dict) -> None:
    lines = ["# Model evaluation report", ""]
    if m["trained_on_sample_data"]:
        lines += ["> **SAMPLE DATA.** These numbers come from synthetic data and only prove the pipeline works.",
                  "> Re-run `python -m model.train` after loading the real files.", ""]
    lines += [f"Trained {m['trained_at'][:10]}. Train: target years up to {m['last_train_year']} ({m['rows']['train']} region-years). "
              f"Test: later years ({m['rows']['test']} region-years).", "",
              "Target: rainfed palay production change (%) vs. the average of the 3 previous years.", "",
              "## Accuracy on the test years", "",
              "| Model | MAE (pp) | RMSE (pp) | MAE in El Niño years (pp) |", "| --- | --- | --- | --- |"]
    for s in m["scores"]:
        chosen = " **(chosen)**" if s["model"] == m["chosen_model"] else ""
        lines.append(f"| {s['model']}{chosen} | {s['mae']} | {s['rmse']} | {s['mae_elnino_years']} |")
    lines += ["", "pp = percentage points. Lower is better. The baseline shows what a model that learned nothing would score.", ""]
    if m["ranking"]:
        names = [k for k in m["ranking"][0] if k not in ("year", "regions")]
        lines += ["## Ranking the worst-hit regions in test El Niño years", "",
                  "Spearman correlation between predicted risk and actual loss across regions (1 = perfect order, 0 = no better than chance).", "",
                  "| Year | Regions | " + " | ".join(names) + " |", "| --- | --- | " + " | ".join("---" for _ in names) + " |"]
        for r in m["ranking"]:
            lines.append(f"| {r['year']} | {r['regions']} | " + " | ".join(str(r[n]) for n in names) + " |")
        lines.append("")
    lines += ["## What the chosen model relies on", "",
              "Permutation importance: how much the test MAE gets worse when a feature is shuffled.", "",
              "| Feature | MAE increase (pp) |", "| --- | --- |"]
    lines += [f"| {f['feature']} | {f['mae_increase']} |" for f in m["feature_importance"]]
    lines += ["", "## Limitations", "",
              "- Only three strong El Niño events (1997–98, 2015–16, 2023–24) — few examples of the extreme cases.",
              "- Regional totals hide differences between provinces; a provincial model needs provincial rainfall.",
              "- Satellite rainfall (NASA POWER) is less precise than PAGASA station data.",
              "- A forecast needs a forecast ONI (from NOAA or PAGASA) as input; the model does not predict ONI itself.", ""]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
