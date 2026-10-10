"""Clean the CSV files and load them into the database.

Looks in data/raw/ first (your real downloads). Any file missing there is taken
from data/sample/ instead, and the source used is recorded in the `meta` table
so the dashboard can warn when it is showing sample data.

Run:  python scripts/load_data.py
"""
from __future__ import annotations

import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.db import get_engine  # noqa: E402

RAW = ROOT / "data" / "raw"
SAMPLE = ROOT / "data" / "sample"

REQUIRED = {
    "regions": ["code", "name"],
    "oni": ["season", "year", "anomaly"],
    "palay_production": ["region_code", "year", "ecosystem", "production_mt", "area_ha"],
    "rice_prices": ["year", "month", "price_php_per_kg"],
    "rainfall": ["region_code", "year", "month", "rainfall_mm", "normal_mm"],
}


def parse_noaa_oni_ascii(path: Path) -> pd.DataFrame:
    """Parse NOAA CPC's oni.ascii.txt (columns: SEAS YR TOTAL ANOM)."""
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 4 and re.fullmatch(r"[A-Z]{3}", parts[0]) and parts[1].isdigit():
            rows.append({"season": parts[0], "year": int(parts[1]), "anomaly": float(parts[3])})
    if not rows:
        raise ValueError(f"No ONI rows found in {path}")
    return pd.DataFrame(rows)


def find(name: str) -> tuple[pd.DataFrame, str]:
    if name == "oni" and (RAW / "oni.ascii.txt").exists():
        return parse_noaa_oni_ascii(RAW / "oni.ascii.txt"), "raw (NOAA oni.ascii.txt)"
    raw_file = RAW / f"{name}.csv"
    if raw_file.exists():
        return pd.read_csv(raw_file), "raw"
    return pd.read_csv(SAMPLE / f"{name}.csv"), "sample"


def clean(name: str, df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED[name] if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}. Expected {REQUIRED[name]}")
    df = df[REQUIRED[name]].copy()
    if "ecosystem" in df:
        df["ecosystem"] = df["ecosystem"].str.strip().str.lower()
    if "region_code" in df:
        df["region_code"] = df["region_code"].astype(str).str.strip()
    numeric = [c for c in df.columns if c not in ("season", "ecosystem", "region_code", "code", "name")]
    for c in numeric:
        # PSA files often use "..", "-" or commas for thousands
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce")
    before = len(df)
    df = df.dropna()
    if before - len(df):
        print(f"  {name}: dropped {before - len(df)} rows with missing values")
    return df


def main() -> None:
    engine = get_engine()
    meta = []
    for name in REQUIRED:
        df, source = find(name)
        df = clean(name, df)
        df.to_sql(name, engine, if_exists="replace", index=False)
        meta.append({"table_name": name, "source": source, "rows": len(df)})
        print(f"Loaded {name:<17} {len(df):>6} rows  from {source}")
    pd.DataFrame(meta).assign(loaded_at=datetime.now(timezone.utc).isoformat()).to_sql(
        "meta", engine, if_exists="replace", index=False
    )


if __name__ == "__main__":
    main()
