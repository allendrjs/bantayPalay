// El Niño Watch PH – dashboard logic. Talks only to our own API (/api/*).
const state = { region: "", event: "2023-24", eco: "both", oni: [], production: [], risk: [], summary: null };
let oniChart, prodChart;

const $ = (id) => document.getElementById(id);
const fmtPct = (v) => (v === null || v === undefined ? "n/a" : `${v > 0 ? "+" : ""}${v.toFixed(1)}%`);
const getJSON = async (url) => {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} → ${res.status}`);
  return res.json();
};
const qs = (params) => new URLSearchParams(Object.entries(params).filter(([, v]) => v)).toString();

async function init() {
  const [meta, regions, events, oni] = await Promise.all([
    getJSON("/api/meta"), getJSON("/api/regions"), getJSON("/api/events"), getJSON("/api/oni"),
  ]);
  $("sample-banner").hidden = !meta.using_sample_data;
  $("loaded-at").textContent = `Data loaded: ${new Date(meta.tables[0]?.loaded_at ?? Date.now()).toLocaleString()}`;

  for (const r of regions) $("f-region").add(new Option(r.name, r.code));
  for (const e of [...events].reverse()) $("f-event").add(new Option(e.key.replace("-", "–"), e.key));
  state.events = events;
  state.oni = oni.series;
  renderLatestOni(oni.latest);

  $("f-region").addEventListener("change", (e) => { state.region = e.target.value; refresh(); });
  $("f-event").addEventListener("change", (e) => { state.event = e.target.value; refresh(); });
  $("f-eco").addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-eco]");
    if (!btn) return;
    state.eco = btn.dataset.eco;
    for (const b of $("f-eco").querySelectorAll("button")) b.setAttribute("aria-pressed", String(b === btn));
    renderEco();
  });
  $("download").addEventListener("click", downloadCsv);

  state.risk = await getJSON("/api/regions/risk");
  try {
    const m = await getJSON("/api/model/metrics");
    $("model-name").textContent = m.chosen_model + (m.trained_on_sample_data ? ", sample data" : "");
  } catch { /* no model trained yet: the forecast columns show n/a */ }
  await refresh();
}

async function refresh() {
  const [summary, production] = await Promise.all([
    getJSON(`/api/summary?${qs({ event: state.event, region: state.region })}`),
    getJSON(`/api/production/by-event?${qs({ region: state.region })}`),
  ]);
  state.summary = summary;
  state.production = production;
  renderKpis();
  renderOniChart();
  renderProdChart();
  renderInsight();
  renderRisk();
  renderEco();
}

function renderLatestOni(latest) {
  $("k-oni").textContent = `${latest.anomaly > 0 ? "+" : ""}${latest.anomaly.toFixed(1)}`;
  const pill = $("k-oni-status");
  pill.textContent = latest.status;
  pill.className = "pill" + (latest.status === "El Niño" ? " elnino" : latest.status === "La Niña" ? " lanina" : "");
  $("k-oni-season").textContent = `NOAA CPC · ${latest.season} ${latest.year}`;
}

function renderKpis() {
  const { production_change_pct: p, rice_price_change: price } = state.summary;
  $("k-rainfed").textContent = fmtPct(p.rainfed);
  $("k-irrigated").textContent = fmtPct(p.irrigated);
  $("k-price").textContent = price.change_php === null ? "n/a" : `+₱${price.change_php.toFixed(2)}/kg`;
}

function renderEco() {
  for (const card of document.querySelectorAll("[data-eco-card]")) {
    card.classList.toggle("dim", state.eco !== "both" && card.dataset.ecoCard !== state.eco);
  }
  if (!prodChart) return;
  prodChart.data.datasets.forEach((ds) => { ds.hidden = state.eco !== "both" && ds.key !== state.eco; });
  prodChart.update();
}

// Shades the selected event's years behind the ONI line.
const bandPlugin = {
  id: "eventBand",
  beforeDatasetsDraw(chart, _args, opts) {
    if (opts.from === undefined) return;
    const { ctx, chartArea, scales } = chart;
    const x0 = scales.x.getPixelForValue(opts.from);
    const x1 = scales.x.getPixelForValue(opts.to);
    ctx.save();
    ctx.fillStyle = "#FDE7C9";
    ctx.fillRect(x0, chartArea.top, x1 - x0, chartArea.bottom - chartArea.top);
    ctx.restore();
  },
};

function renderOniChart() {
  const labels = state.oni.map((d) => `${d.season} ${d.year}`);
  const values = state.oni.map((d) => d.anomaly);
  const ev = state.events.find((e) => e.key === state.event);
  const from = state.oni.findIndex((d) => d.year === ev.start_year);
  const to = state.oni.findLastIndex((d) => d.year === ev.end_year);

  if (oniChart) {
    oniChart.options.plugins.eventBand = { from, to };
    oniChart.update();
    return;
  }
  const flat = (v) => labels.map(() => v);
  oniChart = new Chart($("oni-chart"), {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "ONI", data: values, borderColor: "#1C2321", borderWidth: 2, pointRadius: 0, tension: 0.25 },
        { label: "El Niño threshold", data: flat(0.5), borderColor: "#B45309", borderDash: [6, 5], borderWidth: 1.5, pointRadius: 0 },
        { label: "La Niña threshold", data: flat(-0.5), borderColor: "#0F766E", borderDash: [6, 5], borderWidth: 1.5, pointRadius: 0 },
      ],
    },
    options: {
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { display: false },
        eventBand: { from, to },
        tooltip: { filter: (item) => item.datasetIndex === 0, callbacks: { label: (c) => `ONI ${c.parsed.y.toFixed(2)}` } },
      },
      scales: {
        x: { grid: { display: false }, ticks: { maxTicksLimit: 8, callback(v) { return this.getLabelForValue(v).split(" ")[1]; } } },
        y: { suggestedMin: -3, suggestedMax: 3, grid: { color: "#EEF1F0" } },
      },
    },
    plugins: [bandPlugin],
  });
}

function renderProdChart() {
  const labels = state.production.map((d) => d.event.replace("-", "–"));
  const series = [
    { key: "irrigated", label: "Irrigated", backgroundColor: "#0F766E" },
    { key: "rainfed", label: "Rainfed", backgroundColor: "#E8913A" },
  ].map((s) => ({ ...s, data: state.production.map((d) => d[s.key]), borderRadius: 4, maxBarThickness: 44 }));

  if (prodChart) {
    prodChart.data.labels = labels;
    prodChart.data.datasets.forEach((ds, i) => { ds.data = series[i].data; });
    prodChart.update();
    return;
  }
  prodChart = new Chart($("prod-chart"), {
    type: "bar",
    data: { labels, datasets: series },
    options: {
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "top", align: "start", labels: { boxWidth: 12, boxHeight: 12 } },
        tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${fmtPct(c.parsed.y)}` } },
      },
      scales: {
        x: { grid: { display: false } },
        y: { grid: { color: "#EEF1F0" }, ticks: { callback: (v) => `${v}%` } },
      },
    },
  });
}

function renderInsight() {
  const s = state.summary;
  const { rainfed, irrigated } = s.production_change_pct;
  const where = state.region ? $("f-region").selectedOptions[0].text : "the Philippines";
  const ev = s.event.replace("-", "–");
  let text = `No production data for ${where} in the ${ev} event.`;
  if (rainfed !== null && irrigated !== null) {
    const gap = Math.abs(rainfed - irrigated).toFixed(1);
    text = `During the ${ev} El Niño (ONI peak ${s.event_peak_oni.toFixed(1)}), rainfed palay in ${where} changed by ${fmtPct(rainfed)} while irrigated palay changed by ${fmtPct(irrigated)} — a gap of ${gap} percentage points.`;
  }
  $("insight-text").textContent = text;
  const corr = s.oni_vs_rainfed_correlation;
  $("k-corr").textContent = corr.r === null ? "n/a" : `r = ${corr.r.toFixed(2)}`;
  $("k-corr-n").textContent = `${corr.n} event × region pairs`;
}

function renderRisk() {
  const body = $("risk-body");
  body.replaceChildren();
  for (const r of state.risk) {
    const tr = document.createElement("tr");
    if (r.region_code === state.region) tr.className = "selected";
    const cells = [
      [r.region_name],
      [r.level, (td) => { td.textContent = ""; const s = document.createElement("span"); s.className = `risk ${r.level}`; s.textContent = r.level; td.append(s); }],
      [String(r.score), null, "num"],
      [`${r.rainfed_share.toFixed(1)}%`, null, "num"],
      [fmtPct(r.rain_vs_normal), null, "num"],
      [fmtPct(r.past_rainfed_change), null, "num"],
      [r.model_level ?? "n/a", r.model_level ? (td) => { td.textContent = ""; const s = document.createElement("span"); s.className = `risk ${r.model_level}`; s.textContent = r.model_level; td.append(s); } : null],
      [r.model_predicted_change === null ? "n/a" : fmtPct(r.model_predicted_change), null, "num"],
    ];
    for (const [text, custom, cls] of cells) {
      const td = document.createElement("td");
      td.textContent = text;
      if (cls) td.className = cls;
      if (custom) custom(td);
      tr.append(td);
    }
    body.append(tr);
  }
}

function downloadCsv() {
  const rows = [["region_code", "region_name", "risk_level", "score", "rainfed_share_pct", "rain_vs_normal_pct", "past_rainfed_change_pct", "model_level", "model_predicted_change_pct"]];
  for (const r of state.risk) rows.push([r.region_code, r.region_name, r.level, r.score, r.rainfed_share, r.rain_vs_normal, r.past_rainfed_change, r.model_level ?? "", r.model_predicted_change ?? ""]);
  const csv = rows.map((row) => row.map((v) => `"${String(v).replaceAll('"', '""')}"`).join(",")).join("\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
  a.download = "regional_drought_risk.csv";
  a.click();
  URL.revokeObjectURL(a.href);
}

init().catch((err) => {
  console.error(err);
  $("risk-body").innerHTML = "<tr><td colspan='8'>Could not load data. Is the API running and the database loaded?</td></tr>";
});
