# El Niño Watch PH — Impact Dashboard (starter)

A web dashboard showing how El Niño affects palay production, rice prices and
rainfall across Philippine regions, with a simple regional drought-risk score.

> **The included data is SAMPLE (synthetic) data.** The dashboard shows a yellow
> banner until every table comes from a real file in `data/raw/`.

## Stack

| Part | Tech |
|---|---|
| Data cleaning / analysis | Python, pandas, NumPy |
| Backend API | FastAPI (Python) |
| Database | SQLite by default; MySQL/PostgreSQL via `DATABASE_URL` |
| Frontend | HTML, CSS, JavaScript, Chart.js (bundled in `frontend/vendor/`, works offline) |

## Run it

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/make_sample_data.py # only needed if data/sample/ is empty
python scripts/load_data.py        # cleans CSVs → database
python -m model.train              # trains + evaluates the model, writes reports/model_report.md
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000 for the dashboard and http://127.0.0.1:8000/docs
for the auto-generated API docs. Run the tests with `pytest -q`.

## Project layout

```
app/
  main.py        API routes + serves the frontend
  analysis.py    all calculations (production change, price change, correlation, risk score)
  db.py          database connection (DATABASE_URL)
model/           CS track
  config.py      settings both sides agree on (train/test split year, risk thresholds, features)
  features.py    one row per region-year: ONI, rainfall, rainfed share → rainfed change
  train.py       trains 4 models, time-split evaluation, picks the best, saves it
  predict.py     forecast per region + what-if, risk level from the prediction
  advice.py      SMS alert + TIPS text (Tagalog, Bisaya, English) and when to send
  artifacts/     saved model + metrics.json (created by training)
notebooks/01_eda.ipynb   data exploration (rebuild with scripts/make_eda_notebook.py)
reports/model_report.md  evaluation tables for Chapter 4 (created by training)
scripts/
  make_sample_data.py   writes synthetic CSVs to data/sample/
  load_data.py          cleans data/raw/ (or falls back to data/sample/) → database
  make_eda_notebook.py  builds the exploration notebook
frontend/        index.html, styles.css, app.js, vendor/chart.umd.min.js
data/
  raw/           put your real downloads here
  sample/        synthetic data
tests/           pytest tests for analysis + API
```

## API

| Endpoint | Returns |
|---|---|
| `GET /api/meta` | which tables are real vs. sample |
| `GET /api/regions` | region codes and names |
| `GET /api/events` | El Niño events analysed (1997–98, 2015–16, 2023–24) |
| `GET /api/oni` | ONI time series + latest value and status |
| `GET /api/summary?event=2015-16&region=III` | KPI numbers for the filters |
| `GET /api/production/by-event?region=III` | irrigated vs. rainfed change per event |
| `GET /api/regions/risk` | regions sorted by drought-risk score, plus the model's forecast |
| `GET /api/model/metrics` | evaluation results from the last training run |
| `GET /api/forecast?oni_max=2.0&rain_dev=-25` | predicted rainfed change + risk level per region (what-if optional) |
| `GET /api/advice?region=III&lang=ceb&area=Tarlac` | the SMS alert and TIPS reply for that region now |

## Swapping in real data

Save each file in `data/raw/` with exactly these names and columns, then re-run
`python scripts/load_data.py`. Any file you haven't added yet keeps using the
sample version.

| File | Columns | Where to get it |
|---|---|---|
| `oni.ascii.txt` *(or `oni.csv`)* | NOAA's raw file as-is *(or `season,year,anomaly`)* | NOAA Climate Prediction Center ONI table |
| `regions.csv` | `code,name` | your own list (keep codes consistent across files) |
| `palay_production.csv` | `region_code,year,ecosystem,production_mt,area_ha` | PSA OpenSTAT: Palay volume of production + area harvested, by ecosystem/crop type |
| `rice_prices.csv` | `year,month,price_php_per_kg` | PSA OpenSTAT retail prices (regular-milled rice) |
| `rainfall.csv` | `region_code,year,month,rainfall_mm,normal_mm` | PAGASA request (CADS-01B) or NASA POWER |

Notes:
- PSA downloads are "wide" (one column per year/quarter). Reshape them to the
  long format above with `pandas.melt` — sum quarters to get annual totals, and
  use `irrigated` / `rainfed` as the `ecosystem` values.
- `normal_mm` is the long-term average for that month (e.g. the 1991–2020 mean).
- The loader strips commas, turns PSA's `..` / `-` placeholders into blanks, and
  drops rows with missing values, printing how many it dropped.

## How the numbers are calculated (`app/analysis.py`)

- **Production change:** event's second year (e.g. 2016) vs. the average of the
  3 years before the event starts (2012–2014).
- **Rice price change:** highest monthly price during the event vs. the average of
  the 12 months before it.
- **Correlation:** Pearson r between each event's ONI peak and each region's
  rainfed production change.
- **Risk score (0–100):** 40% rainfed share of palay area + 30% rainfall deficit
  in the last 3 months on record + 30% average rainfed loss in past events,
  each scaled 0–1 across regions. 0–33 Low, 34–66 Moderate, 67–100 High.
  This is a transparent baseline; the CS side can replace `regional_risk()`
  with a trained model and compare it against this baseline in the paper.

## The model (CS track)

**Question:** how much will rainfed palay production change this season, per region?

- **Target:** rainfed production change (%) vs. the average of the 3 previous years.
- **Features:** yearly peak and mean ONI, previous year's peak ONI (harvest lag),
  growing-season rainfall vs. normal (June–November) this year and last year,
  rainfed share of palay area the year before. No feature uses future information.
- **Evaluation:** trained on years up to 2014, tested on 2015–2025 (a time split,
  so it's tested the way it will be used). Four models are compared, including a
  baseline that always predicts the training median: a model must beat it to be
  worth anything. A second check asks whether each approach ranks the worst-hit
  regions first in the 2016 and 2024 El Niño years, against the dashboard's
  original weighted formula.
- **Output:** the best model by test MAE is refit on all years and saved. Its
  prediction becomes a risk level: High at −10% or worse, Moderate at −4% or worse
  (change in `model/config.py`).
- **Forecasting:** the model needs the season's ONI. Use NOAA's or PAGASA's ONI
  forecast through `/api/forecast?oni_max=…` — the model doesn't predict ONI itself.

Ideas to improve it: a feature for each region's own past sensitivity (the
current features can't tell a resilient region from a fragile one with the same
rainfed share), provincial instead of regional data, and irrigation coverage
from NIA.

## Advisor rules (`model/advice.py`)

- Alerts are sent when risk goes up, as an all-clear when it drops back to Low,
  and as a reminder every 30 days while it stays High — never repeated daily.
- Every message fits one 160-character SMS (tested for a 30-character place name).
- **The Tagalog, Bisaya and English text is a draft.** Have the Municipal
  Agriculture Office check the advice and a native speaker check each language,
  then set `VALIDATED = True`.

## Next steps

- Replace the sample files with real data (see above), then re-run
  `load_data.py`, `model.train` and the notebook, and rewrite its findings.
- IT: advisor web app + SMS gateway, calling `/api/advice` and `should_alert()`.
- Deploy: MySQL on Hostinger via `DATABASE_URL`, run with `uvicorn`/`gunicorn`.
