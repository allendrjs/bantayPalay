# Model evaluation report

> **SAMPLE DATA.** These numbers come from synthetic data and only prove the pipeline works.
> Re-run `python -m model.train` after loading the real files.

Trained 2026-10-10. Train: target years up to 2014 (352 region-years). Test: later years (176 region-years).

Target: rainfed palay production change (%) vs. the average of the 3 previous years.

## Accuracy on the test years

| Model | MAE (pp) | RMSE (pp) | MAE in El Niño years (pp) |
| --- | --- | --- | --- |
| Baseline (always predicts the training median) | 9.65 | 12.43 | 15.7 |
| Linear regression | 6.58 | 8.83 | 4.28 |
| Random forest | 6.8 | 8.86 | 4.66 |
| Gradient boosting **(chosen)** | 6.23 | 8.41 | 4.4 |

pp = percentage points. Lower is better. The baseline shows what a model that learned nothing would score.

## Ranking the worst-hit regions in test El Niño years

Spearman correlation between predicted risk and actual loss across regions (1 = perfect order, 0 = no better than chance).

| Year | Regions | Weighted formula (dashboard v1) | Linear regression | Random forest | Gradient boosting |
| --- | --- | --- | --- | --- | --- |
| 2016 | 16 | 0.27 | 0.36 | 0.64 | 0.16 |
| 2024 | 16 | 0.03 | 0.13 | -0.19 | -0.01 |

## What the chosen model relies on

Permutation importance: how much the test MAE gets worse when a feature is shuffled.

| Feature | MAE increase (pp) |
| --- | --- |
| oni_max | 2.58 |
| oni_max_prev | 1.06 |
| rain_dev_prev | 0.55 |
| oni_mean | 0.23 |
| rain_dev | 0.05 |
| rainfed_share | 0.04 |

## Limitations

- Only three strong El Niño events (1997–98, 2015–16, 2023–24) — few examples of the extreme cases.
- Regional totals hide differences between provinces; a provincial model needs provincial rainfall.
- Satellite rainfall (NASA POWER) is less precise than PAGASA station data.
- A forecast needs a forecast ONI (from NOAA or PAGASA) as input; the model does not predict ONI itself.
