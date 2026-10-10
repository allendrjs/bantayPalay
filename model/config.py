"""Settings the CS and IT sides agree on. Change them here, not in the code."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = ROOT / "model" / "artifacts" / "model.joblib"
METRICS = ROOT / "model" / "artifacts" / "metrics.json"
REPORT = ROOT / "reports" / "model_report.md"

# Train on target years up to this one; test on everything after it.
# A time split (not a random one) so the model is tested on years it has never seen,
# the same way it will be used: trained on the past, predicting the future.
LAST_TRAIN_YEAR = 2014

# Predicted rainfed production change (%) → risk level.
# HIGH if the predicted change is at or below HIGH_AT, MODERATE if at or below MODERATE_AT.
HIGH_AT = -10.0
MODERATE_AT = -4.0

# Months that count as the main growing season when summarising rainfall.
# The Philippine wet-season crop is planted around June and harvested by November.
SEASON_MONTHS = [6, 7, 8, 9, 10, 11]

FEATURES = [
    "oni_max",          # strongest ONI value in the target year
    "oni_max_prev",     # strongest ONI value in the year before (harvest lag)
    "oni_mean",         # average ONI in the target year
    "rain_dev",         # growing-season rainfall vs. normal in the target year (%)
    "rain_dev_prev",    # same, previous year (%)
    "rainfed_share",    # rainfed share of palay area in the previous year (%)
]
TARGET = "rainfed_change"  # % change vs. the average of the 3 previous years
