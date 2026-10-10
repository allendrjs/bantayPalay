"""Builds notebooks/01_eda.ipynb (data exploration). Re-run, then execute it:
    python scripts/make_eda_notebook.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("# 01 · Exploring the El Niño and palay data\n\n"
       "Goal: check the data is clean and consistent, and see how El Niño relates to palay production "
       "before building the model.\n\n"
       "Run `python scripts/load_data.py` first. **If the cell below says SAMPLE, every number here is synthetic** — "
       "re-run this notebook after loading the real files and rewrite the findings at the end."),
    code("import sys; sys.path.insert(0, '..')\n"
         "import pandas as pd, matplotlib.pyplot as plt\n"
         "from app.db import read_table\n"
         "from model.features import build_training_table\n"
         "meta = read_table('meta')\n"
         "print('Data source:', 'SAMPLE (synthetic)' if (meta['source'] == 'sample').any() else 'REAL')\n"
         "meta"),
    md("## 1. Profile each table\nRows, year range and missing values."),
    code("tables = {n: read_table(n) for n in ['oni', 'palay_production', 'rice_prices', 'rainfall', 'regions']}\n"
         "pd.DataFrame([{'table': n, 'rows': len(t), 'years': f\"{t['year'].min()}–{t['year'].max()}\" if 'year' in t else '',\n"
         "               'missing values': int(t.isna().sum().sum())} for n, t in tables.items()])"),
    md("## 2. Region codes that don't match\nA region in one file but not another is silently dropped when tables are joined."),
    code("codes = {n: set(t['region_code']) for n, t in tables.items() if 'region_code' in t}\n"
         "codes['regions'] = set(tables['regions']['code'])\n"
         "everything = set().union(*codes.values())\n"
         "pd.DataFrame({n: {c: c in s for c in sorted(everything)} for n, s in codes.items()}).loc[lambda d: ~d.all(axis=1)]"),
    md("An empty table above means every file uses the same region codes. NCR has no palay, so it may appear here — that's expected."),
    md("## 3. Outliers\nYear-on-year production jumps over 40% are worth checking against the PSA source (typos, boundary changes)."),
    code("p = tables['palay_production'].sort_values(['region_code', 'ecosystem', 'year'])\n"
         "p['yoy_pct'] = p.groupby(['region_code', 'ecosystem'])['production_mt'].pct_change() * 100\n"
         "p.loc[p['yoy_pct'].abs() > 40, ['region_code', 'ecosystem', 'year', 'production_mt', 'yoy_pct']].round(1)"),
    md("## 4. ONI against national palay production"),
    code("oni_y = tables['oni'].groupby('year')['anomaly'].max()\n"
         "nat = p.groupby(['year', 'ecosystem'])['production_mt'].sum().unstack() / 1e6\n"
         "fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)\n"
         "ax[0].bar(oni_y.index, oni_y, color=['#B45309' if v >= 0.5 else '#9CA3AF' for v in oni_y]); ax[0].axhline(0.5, ls='--', c='#B45309', lw=1)\n"
         "ax[0].set_ylabel('Yearly peak ONI'); ax[0].set_title('El Niño years (orange) and palay production')\n"
         "ax[1].plot(nat.index, nat['irrigated'], c='#0F766E', label='Irrigated'); ax[1].plot(nat.index, nat['rainfed'], c='#E8913A', label='Rainfed')\n"
         "ax[1].set_ylabel('Million metric tons'); ax[1].legend(frameon=False)\n"
         "for a in ax: a.spines[['top', 'right']].set_visible(False)\n"
         "plt.tight_layout(); plt.show()"),
    md("## 5. The harvest lag: same year or the year after?\n"
       "Correlation between rainfed production change and the ONI peak of the same year vs. the year before. "
       "The stronger one tells us which ONI to feed the model (we use both)."),
    code("t = build_training_table(tables['oni'], tables['palay_production'], tables['rainfall'])\n"
         "t[['rainfed_change', 'oni_max', 'oni_max_prev', 'rain_dev', 'rain_dev_prev', 'rainfed_share']].corr()['rainfed_change'].drop('rainfed_change').round(2).to_frame('correlation with rainfed change')"),
    md("## 6. Rainfed vs. irrigated loss in each El Niño event"),
    code("from app.analysis import EVENTS, production_change\n"
         "pd.DataFrame([{'event': e.key, **production_change(tables['palay_production'], e)} for e in EVENTS.values()]).set_index('event')"),
    md("## Findings\n\n"
       "*Rewrite this after running on the real data.* Answer, with numbers from the cells above:\n\n"
       "1. Is the data clean? Missing values, mismatched region codes and outliers found and how they were fixed.\n"
       "2. Does production fall in the El Niño year itself or the year after (section 5)?\n"
       "3. How much more do rainfed farms lose than irrigated ones (section 6)?\n"
       "4. Which features look most useful for the model?"),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
out = ROOT / "notebooks" / "01_eda.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print(f"Wrote {out}")
