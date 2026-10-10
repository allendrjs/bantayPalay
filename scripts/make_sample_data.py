"""Generate SAMPLE (synthetic) CSV files so the dashboard runs before real data arrives.

Every number produced here is made up. Replace the files in data/raw/ with the
real downloads (see README) and re-run scripts/load_data.py.

Run:  python scripts/make_sample_data.py
"""
from __future__ import annotations

import csv
import math
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "sample"
SEASONS = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]

REGIONS = [
    ("NCR", "National Capital Region"),
    ("CAR", "Cordillera Administrative Region"),
    ("I", "Region I – Ilocos Region"),
    ("II", "Region II – Cagayan Valley"),
    ("III", "Region III – Central Luzon"),
    ("IVA", "Region IV-A – CALABARZON"),
    ("IVB", "Region IV-B – MIMAROPA"),
    ("V", "Region V – Bicol Region"),
    ("VI", "Region VI – Western Visayas"),
    ("VII", "Region VII – Central Visayas"),
    ("VIII", "Region VIII – Eastern Visayas"),
    ("IX", "Region IX – Zamboanga Peninsula"),
    ("X", "Region X – Northern Mindanao"),
    ("XI", "Region XI – Davao Region"),
    ("XII", "Region XII – SOCCSKSARGEN"),
    ("XIII", "Region XIII – Caraga"),
    ("BARMM", "BARMM"),
]

# Synthetic ONI anchor points (year as decimal, value). Shape loosely follows
# the well-known events, but these are NOT the official NOAA values.
ONI_ANCHORS = [
    (1990.0, 0.3), (1992.0, 1.7), (1993.0, 0.2), (1995.0, 1.1), (1996.0, -0.5),
    (1997.9, 2.4), (1999.5, -1.7), (2001.0, -0.3), (2002.9, 1.3), (2005.0, 0.3),
    (2007.9, -1.6), (2009.9, 1.6), (2010.9, -1.6), (2013.0, -0.3), (2015.9, 2.6),
    (2017.9, -0.9), (2019.0, 0.5), (2020.9, -1.3), (2022.0, -1.0), (2023.9, 2.0),
    (2025.9, -0.5),
]


def interp(t: float) -> float:
    for (t0, v0), (t1, v1) in zip(ONI_ANCHORS, ONI_ANCHORS[1:]):
        if t0 <= t <= t1:
            f = (t - t0) / (t1 - t0)
            # cosine easing so peaks look rounded
            f = (1 - math.cos(math.pi * f)) / 2
            return v0 + (v1 - v0) * f
    return ONI_ANCHORS[-1][1]


def oni_rows():
    for year in range(1990, 2026):
        for i, seas in enumerate(SEASONS):
            yield {"season": seas, "year": year, "anomaly": round(interp(year + (i + 0.5) / 12), 2)}


def annual_peak_oni() -> dict[int, float]:
    peaks: dict[int, float] = {}
    for r in oni_rows():
        peaks[r["year"]] = max(peaks.get(r["year"], -9), r["anomaly"])
    return peaks


def main() -> None:
    random.seed(42)
    OUT.mkdir(parents=True, exist_ok=True)
    peaks = annual_peak_oni()

    with open(OUT / "oni.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["season", "year", "anomaly"])
        w.writeheader()
        w.writerows(oni_rows())

    with open(OUT / "regions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["code", "name"])
        w.writerows(REGIONS)

    # Palay production (metric tons), annual, by region and ecosystem.
    with open(OUT / "palay_production.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["region_code", "year", "ecosystem", "production_mt", "area_ha"])
        for code, _ in REGIONS:
            if code == "NCR":
                continue
            base_irr = random.uniform(150_000, 1_800_000)
            rainfed_share = random.uniform(0.15, 0.65)
            base_rf = base_irr * rainfed_share / (1 - rainfed_share)
            sensitivity = random.uniform(0.6, 1.4)  # how hard El Niño hits this region
            for year in range(1990, 2026):
                growth = 1 + 0.012 * (year - 1990)
                # The year after a strong peak takes the hit (harvest lag).
                hit = max(0.0, peaks.get(year - 1, 0) - 0.5) * 0.5 + max(0.0, peaks.get(year, 0) - 1.0) * 0.5
                irr = base_irr * growth * (1 - 0.04 * hit * sensitivity) * random.uniform(0.97, 1.03)
                rf = base_rf * growth * (1 - 0.11 * hit * sensitivity) * random.uniform(0.95, 1.05)
                for eco, prod, yld in (("irrigated", irr, 4.6), ("rainfed", rf, 3.2)):
                    w.writerow([code, year, eco, round(prod), round(prod / yld)])

    # National regular-milled rice retail price (PHP/kg), monthly.
    with open(OUT / "rice_prices.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["year", "month", "price_php_per_kg"])
        price = 17.0
        for year in range(1990, 2026):
            for month in range(1, 13):
                oni = interp(year + (month - 0.5) / 12)
                price *= 1.0035 + max(0.0, oni - 0.5) * 0.004 + random.uniform(-0.003, 0.003)
                w.writerow([year, month, round(price, 2)])

    # Rainfall (mm) vs. the monthly normal, monthly by region, 1990-2025.
    with open(OUT / "rainfall.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["region_code", "year", "month", "rainfall_mm", "normal_mm"])
        for code, _ in REGIONS:
            for year in range(1990, 2026):
                for month in range(1, 13):
                    normal = 120 + 140 * math.sin(math.pi * (month - 2) / 12) ** 2
                    oni = interp(year + (month - 0.5) / 12)
                    actual = normal * (1 - max(0.0, oni) * 0.18 * random.uniform(0.6, 1.4)) * random.uniform(0.85, 1.15)
                    w.writerow([code, year, month, round(actual, 1), round(normal, 1)])

    print(f"Sample data written to {OUT}")


if __name__ == "__main__":
    main()
