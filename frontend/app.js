const API = ""; // тот же ориджин (FastAPI отдаёт и API, и статику)

const statusRu = { OK: "Норма", WARNING: "Внимание", CRITICAL: "Критично" };

let currentFleet = [];
let selectedSerial = null;

async function jget(url, opts) {
  const r = await fetch(API + url, opts);
  if (!r.ok) throw new Error(`${url} -> ${r.status}`);
  return r.json();
}

function el(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html !== undefined) e.innerHTML = html;
  return e;
}

async function loadFleet() {
  const data = await jget("/api/units");
  currentFleet = data.units;
  renderTopbarStats(data);
  renderFleetGrid(data.units);
  if (data.units.length) selectUnit(data.units[0].serial);
}

function renderTopbarStats(data) {
  const wrap = document.getElementById("topbar-stats");
  wrap.innerHTML = "";
  const chips = [
    { label: "Всего изделий", num: data.count, cls: "" },
    { label: "Норма", num: data.ok, cls: "ok" },
    { label: "Внимание", num: data.warning, cls: "warning" },
    { label: "Критично", num: data.critical, cls: "critical" },
  ];
  chips.forEach((c) => {
    const chip = el("div", `stat-chip ${c.cls}`);
    chip.appendChild(el("span", "num", c.num));
    chip.appendChild(el("span", "label", c.label));
    wrap.appendChild(chip);
  });
}

function renderFleetGrid(units) {
  const grid = document.getElementById("fleet-grid");
  grid.innerHTML = "";
  units.forEach((u) => {
    const card = el("div", `unit-card status-${u.health_status}`);
    card.dataset.serial = u.serial;
    const top = el("div", "unit-card-top");
    top.appendChild(el("span", "unit-serial", u.serial));
    top.appendChild(el("span", `unit-badge ${u.health_status}`, statusRu[u.health_status] || u.health_status));
    card.appendChild(top);
    card.appendChild(el("div", "unit-meta", `${u.product}<br/>${u.customer}`));
    const qrow = el("div", "unit-qbar-row");
    const track = el("div", "unit-qbar");
    const fill = el("i");
    fill.style.width = `${u.quality_score}%`;
    track.appendChild(fill);
    qrow.appendChild(track);
    qrow.appendChild(el("span", "unit-qval", `${Math.round(u.quality_score)}`));
    card.appendChild(qrow);
    card.addEventListener("click", () => selectUnit(u.serial));
    grid.appendChild(card);
  });
}

function markSelected(serial) {
  document.querySelectorAll(".unit-card").forEach((c) => {
    c.classList.toggle("selected", c.dataset.serial === serial);
  });
}

async function selectUnit(serial) {
  selectedSerial = serial;
  markSelected(serial);
  const panel = document.getElementById("detail-panel");
  panel.innerHTML = '<div class="empty-state small">Загрузка цифрового паспорта...</div>';

  const [unit, explanation] = await Promise.all([
    jget(`/api/units/${serial}`),
    jget(`/api/units/${serial}/explain`),
  ]);

  panel.innerHTML = "";

  // Passport header
  const head = el("div", "passport-head");
  const img = document.createElement("img");
  img.className = "passport-img";
  img.src = `/api/units/${serial}/qc-image`;
  head.appendChild(img);
  const meta = el("div");
  meta.appendChild(el("div", "passport-title", `${unit.serial}`));
  meta.appendChild(el("div", "passport-sub", `${unit.product}`));
  meta.appendChild(el("div", "passport-sub", `Заказчик: ${unit.customer}`));
  meta.appendChild(el("div", "passport-sub", `Дата производства: ${unit.manufactured_date}`));
  head.appendChild(meta);
  panel.appendChild(head);

  // KPIs
  const kpiRow = el("div", "kpi-row");
  kpiRow.appendChild(kpi("Quality Score (ОТК)", `${unit.qc.quality_score.toFixed(0)} / 100`));
  kpiRow.appendChild(kpi("Класс дефекта", unit.qc.predicted_class));
  kpiRow.appendChild(kpi("AI early-warning", unit.ai_early_warning.probability != null ? `${Math.round(unit.ai_early_warning.probability * 100)}%` : "н/д"));
  kpiRow.appendChild(kpi("Статус C++ детектора", unit.cpp_realtime_analysis.status || "н/д"));
  if (unit.lead_time_days_vs_classic) {
    kpiRow.appendChild(kpi("Выигрыш во времени", `+${unit.lead_time_days_vs_classic} дн.`));
  }
  panel.appendChild(kpiRow);

  // QC probabilities
  panel.appendChild(el("div", "section-title", "Результат выходного контроля (CNN)"));
  panel.appendChild(renderProbBars(unit.qc.probabilities));

  // Telemetry chart
  panel.appendChild(el("div", "section-title", "Телеметрия эксплуатации (СМАРТ ТМ) — температура токоведущих частей, °C"));
  const chartWrap = el("div", "chart-wrap");
  chartWrap.appendChild(lineChartSVG(unit.telemetry.days, unit.telemetry.temperature_c, {
    markDay: unit.telemetry.will_degrade ? unit.telemetry.degrade_start_day : null,
    anomalyDay: unit.cpp_realtime_analysis.anomaly_start_day > 0 ? unit.cpp_realtime_analysis.anomaly_start_day : null,
  }));
  panel.appendChild(chartWrap);

  // RAG explanation
  panel.appendChild(el("div", "section-title", "AI-объяснение (RAG над базой эксплуатационных знаний)"));
  const box = el("div", "explain-box", explanation.explanation);
  panel.appendChild(box);
  panel.appendChild(el("div", "explain-sources", `Источники: ${explanation.sources.join(", ")}`));
}

function kpi(label, num) {
  const k = el("div", "kpi");
  k.appendChild(el("div", "num", num));
  k.appendChild(el("div", "label", label));
  return k;
}

function renderProbBars(probs) {
  const wrap = el("div", "prob-bars");
  Object.entries(probs).sort((a, b) => b[1] - a[1]).forEach(([cls, p]) => {
    const row = el("div", "prob-bar-row");
    row.appendChild(el("span", null, cls));
    const track = el("div", "prob-bar-track");
    const fill = el("div", "prob-bar-fill");
    fill.style.width = `${(p * 100).toFixed(1)}%`;
    track.appendChild(fill);
    row.appendChild(track);
    row.appendChild(el("span", null, `${(p * 100).toFixed(0)}%`));
    wrap.appendChild(row);
  });
  return wrap;
}

function lineChartSVG(days, values, opts = {}) {
  const w = 760, h = 180, padL = 34, padR = 10, padT = 10, padB = 20;
  const minV = Math.min(...values) - 1, maxV = Math.max(...values) + 1;
  const x = (i) => padL + (i / (days.length - 1)) * (w - padL - padR);
  const y = (v) => padT + (1 - (v - minV) / (maxV - minV)) * (h - padT - padB);

  let path = "";
  values.forEach((v, i) => {
    path += (i === 0 ? "M" : "L") + x(i).toFixed(1) + "," + y(v).toFixed(1) + " ";
  });

  const svgns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgns, "svg");
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  svg.setAttribute("width", "100%");
  svg.setAttribute("height", h);

  // grid lines
  for (let i = 0; i <= 3; i++) {
    const gy = padT + (i / 3) * (h - padT - padB);
    const line = document.createElementNS(svgns, "line");
    line.setAttribute("x1", padL); line.setAttribute("x2", w - padR);
    line.setAttribute("y1", gy); line.setAttribute("y2", gy);
    line.setAttribute("stroke", "#24314d"); line.setAttribute("stroke-width", "1");
    svg.appendChild(line);
    const val = maxV - (i / 3) * (maxV - minV);
    const label = document.createElementNS(svgns, "text");
    label.setAttribute("x", 4); label.setAttribute("y", gy + 4);
    label.setAttribute("fill", "#93a2bf"); label.setAttribute("font-size", "10");
    label.textContent = val.toFixed(0);
    svg.appendChild(label);
  }

  // degrade start marker
  if (opts.markDay != null) {
    const lx = x(opts.markDay);
    const line = document.createElementNS(svgns, "line");
    line.setAttribute("x1", lx); line.setAttribute("x2", lx);
    line.setAttribute("y1", padT); line.setAttribute("y2", h - padB);
    line.setAttribute("stroke", "#f5b642"); line.setAttribute("stroke-width", "1.5");
    line.setAttribute("stroke-dasharray", "4 3");
    svg.appendChild(line);
  }
  // anomaly detected marker
  if (opts.anomalyDay != null) {
    const lx = x(opts.anomalyDay);
    const line = document.createElementNS(svgns, "line");
    line.setAttribute("x1", lx); line.setAttribute("x2", lx);
    line.setAttribute("y1", padT); line.setAttribute("y2", h - padB);
    line.setAttribute("stroke", "#ff5c72"); line.setAttribute("stroke-width", "1.5");
    svg.appendChild(line);
  }
  // observation window boundary (day 90) used by AI early-warning
  const ox = x(Math.min(90, days.length - 1));
  const oline = document.createElementNS(svgns, "line");
  oline.setAttribute("x1", ox); oline.setAttribute("x2", ox);
  oline.setAttribute("y1", padT); oline.setAttribute("y2", h - padB);
  oline.setAttribute("stroke", "#3ea6ff"); oline.setAttribute("stroke-width", "1");
  oline.setAttribute("stroke-dasharray", "2 3");
  svg.appendChild(oline);

  const p = document.createElementNS(svgns, "path");
  p.setAttribute("d", path.trim());
  p.setAttribute("fill", "none");
  p.setAttribute("stroke", "#3ea6ff");
  p.setAttribute("stroke-width", "2");
  svg.appendChild(p);

  const legend = el("div", "hint");
  legend.style.marginTop = "6px";
  legend.innerHTML = `<span style="color:#3ea6ff">┊</span> граница окна AI (90 дн.) &nbsp; <span style="color:#f5b642">┊</span> фактическое начало деградации &nbsp; <span style="color:#ff5c72">┃</span> обнаружение C++ детектором`;

  const container = document.createDocumentFragment();
  container.appendChild(svg);
  const wrapDiv = document.createElement("div");
  wrapDiv.appendChild(container);
  wrapDiv.appendChild(legend);
  return wrapDiv;
}

// ---------- QC live demo ----------

async function loadQcSamples() {
  const samples = await jget("/api/qc/sample-images");
  const wrap = document.getElementById("qc-samples");
  wrap.innerHTML = "";
  samples.forEach((s) => {
    const img = document.createElement("img");
    img.className = "qc-sample-thumb";
    img.src = s.path;
    img.title = s.class;
    img.addEventListener("click", () => analyzeImageUrl(s.path));
    wrap.appendChild(img);
  });
}

async function analyzeImageUrl(url) {
  const resp = await fetch(url);
  const blob = await resp.blob();
  await analyzeBlob(blob);
}

async function analyzeBlob(blob) {
  const resultBox = document.getElementById("qc-result");
  resultBox.innerHTML = '<div class="empty-state small">Анализ...</div>';
  const form = new FormData();
  form.append("file", blob, "sample.png");
  const r = await fetch("/api/qc/analyze", { method: "POST", body: form });
  const data = await r.json();
  resultBox.innerHTML = "";
  const kpiRow = el("div", "kpi-row");
  kpiRow.appendChild(kpi("Предсказанный класс", data.predicted_class));
  kpiRow.appendChild(kpi("Уверенность модели", `${(data.confidence * 100).toFixed(1)}%`));
  kpiRow.appendChild(kpi("Quality Score", `${data.quality_score.toFixed(0)} / 100`));
  resultBox.appendChild(kpiRow);
  resultBox.appendChild(renderProbBars(data.probabilities));
}

document.getElementById("qc-file-input").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) analyzeBlob(file);
});

// ---------- C++ benchmark ----------

document.getElementById("bench-btn").addEventListener("click", async () => {
  const box = document.getElementById("bench-result");
  box.innerHTML = "Выполняется...";
  const data = await jget("/api/engine/benchmark?n=2000000");
  box.innerHTML = `
    <div class="big">${Math.round(data.throughput_points_per_sec).toLocaleString("ru-RU")} точек/сек</div>
    <div>${data.benchmark_points.toLocaleString("ru-RU")} измерений телеметрии обработано за ${data.elapsed_seconds.toFixed(3)} сек — движок способен работать на недорогом edge-устройстве рядом со шкафом, без постоянного канала в облако.</div>
  `;
});

loadFleet();
loadQcSamples();
