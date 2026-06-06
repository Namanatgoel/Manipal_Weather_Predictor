/**
 * app.js — Manipal-Climate-RNN Dashboard
 * ──────────────────────────────────────────────────────────────────────────
 * Sources:
 *   inferences.json          → 365-day actual vs predicted (temp + precip)
 *   ../artifacts/climate_analysis.json → OLS regression results
 *
 * Renders:
 *   • 4 metric cards (RMSE/MAE for each target, colour-coded pass/fail)
 *   • Climate trend card (slope statement)
 *   • Chart 1: Temperature Actual vs Predicted (time-series)
 *   • Chart 2: Precipitation Actual vs Predicted (time-series)
 *   • Chart 3: Yearly mean temperature scatter + OLS regression line
 */
"use strict";

/* ── Chart.js global defaults ──────────────────────────────────────────── */
Chart.defaults.color         = "#8b949e";
Chart.defaults.borderColor   = "#30363d";
Chart.defaults.font.family   = "'Inter',system-ui,sans-serif";
Chart.defaults.plugins.tooltip.backgroundColor = "#161b22";
Chart.defaults.plugins.tooltip.borderColor     = "#30363d";
Chart.defaults.plugins.tooltip.borderWidth     = 1;
Chart.defaults.plugins.tooltip.titleColor      = "#e6edf3";
Chart.defaults.plugins.tooltip.bodyColor       = "#8b949e";
Chart.defaults.plugins.tooltip.padding         = 10;
Chart.defaults.plugins.tooltip.mode            = "index";
Chart.defaults.plugins.tooltip.intersect       = false;

const GRID = "#21262d";

/* ── Build time-series Chart.js config ─────────────────────────────────── */
function tsConfig(labels, datasets, yLabel, yUnit) {
  return {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      animation: { duration: 500 },
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: {
          position: "top",
          labels: { usePointStyle: true, pointStyle: "circle",
                    boxWidth: 8, padding: 16, font: { size: 11 } },
        },
        tooltip: {
          callbacks: {
            label: (ctx) =>
              ` ${ctx.dataset.label}: ${Number(ctx.parsed.y).toFixed(3)} ${yUnit}`,
          },
        },
      },
      scales: {
        x: {
          type: "time",
          time: { unit: "month", tooltipFormat: "dd MMM yyyy" },
          grid: { color: GRID },
          ticks: { maxRotation: 0, font: { size: 10 } },
        },
        y: {
          title: { display: true, text: yLabel, color: "#8b949e", font: { size: 10 } },
          grid: { color: GRID },
        },
      },
    },
  };
}

/* ── Build OLS scatter+line config ─────────────────────────────────────── */
function climateConfig(years, means, regLine) {
  const scatter = years.map((yr, i) => ({ x: yr, y: means[i] }));
  const line    = years.map((yr) => ({ x: yr, y: regLine[String(yr)] ?? null }));
  return {
    type: "scatter",
    data: {
      datasets: [
        {
          label: "Yearly Mean Temperature",
          data: scatter, type: "scatter",
          backgroundColor: "#d2a8ff", borderColor: "#d2a8ff",
          pointRadius: 7, pointHoverRadius: 9,
        },
        {
          label: "OLS Regression Line",
          data: line, type: "line",
          borderColor: "#ff7b72", borderWidth: 2.5,
          borderDash: [7, 4], pointRadius: 0, fill: false, tension: 0,
        },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      animation: { duration: 500 },
      plugins: {
        legend: {
          position: "top",
          labels: { usePointStyle: true, pointStyle: "circle", boxWidth: 8, padding: 16 },
        },
        tooltip: {
          callbacks: {
            label: (ctx) =>
              ` ${ctx.dataset.label}: ${Number(ctx.parsed.y).toFixed(4)} °C  (${Math.round(ctx.parsed.x)})`,
          },
        },
      },
      scales: {
        x: {
          title: { display: true, text: "Year", color: "#8b949e" },
          grid: { color: GRID },
          ticks: { stepSize: 1, callback: (v) => String(Math.round(v)) },
        },
        y: {
          title: { display: true, text: "Mean Temperature (°C)", color: "#8b949e" },
          grid: { color: GRID },
        },
      },
    },
  };
}

/* ── Metric card helpers ────────────────────────────────────────────────── */
function setCard(cardId, valueText, pass = null) {
  const card  = document.getElementById(cardId);
  const valEl = card?.querySelector(".metric-value");
  if (valEl) valEl.textContent = valueText;
  if (pass === true)  card?.classList.add("metric-pass");
  if (pass === false) card?.classList.add("metric-fail");
}

/* ── Main ───────────────────────────────────────────────────────────────── */
async function init() {
  /* 1. Fetch inferences.json */
  let inf = null;
  try {
    const r = await fetch("inferences.json");
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    inf = await r.json();
  } catch (err) {
    console.error("Failed to load inferences.json:", err);
    document.querySelectorAll(".metric-value").forEach(el => el.textContent = "N/A");
  }

  /* 2. Fetch climate_analysis.json */
  let climate = null;
  try {
    const r = await fetch("../artifacts/climate_analysis.json");
    if (r.ok) climate = await r.json();
  } catch (_) {}

  /* 3. Metric cards ─────────────────────────────────────────────────────── */
  if (inf) {
    const m = inf.metadata;
    setCard("card-temp-rmse",
            `${m.temperature_rmse.toFixed(4)} °C`,
            m.temperature_rmse <= 0.6);
    const tgt_t = document.getElementById("tgt-temp-rmse");
    if (tgt_t) tgt_t.textContent =
      m.temperature_rmse <= 0.6 ? "✔ PASS — target ≤ 0.60 °C" : "✘ MISS — target ≤ 0.60 °C";

    setCard("card-temp-mae", `${m.temperature_mae.toFixed(4)} °C`);

    setCard("card-precip-rmse",
            `${m.precipitation_rmse.toFixed(4)} mm`,
            m.precipitation_rmse <= 12.5);
    const tgt_p = document.getElementById("tgt-precip-rmse");
    if (tgt_p) tgt_p.textContent =
      m.precipitation_rmse <= 12.5 ? "✔ PASS — target ≤ 12.50 mm" : "✘ MISS — target ≤ 12.50 mm";

    setCard("card-precip-mae", `${m.precipitation_mae.toFixed(4)} mm`);
  }

  if (climate) {
    const slopeEl  = document.getElementById("val-slope");
    const interpEl = document.getElementById("slope-interp");
    if (slopeEl) slopeEl.textContent =
      `${climate.slope_m >= 0 ? "+" : ""}${climate.slope_m.toFixed(6)} °C/yr`;
    if (interpEl) interpEl.textContent = climate.interpretation;
  }

  /* 4. Temperature chart ─────────────────────────────────────────────────── */
  if (inf) {
    const td = inf.temperature;
    const tempCtx = document.getElementById("tempChart")?.getContext("2d");
    if (tempCtx) {
      new Chart(tempCtx, tsConfig(
        td.dates,
        [
          {
            label: "Actual Temperature", data: td.actual,
            borderColor: "#58a6ff", backgroundColor: "#58a6ff18",
            borderWidth: 1.8, pointRadius: 0, fill: true, tension: 0.3,
          },
          {
            label: "Predicted Temperature", data: td.predicted,
            borderColor: "#ff7b72", borderWidth: 1.8,
            pointRadius: 0, borderDash: [5, 3], fill: false, tension: 0.3,
          },
        ],
        "Temperature (°C)", "°C"
      ));
    }

    /* 5. Precipitation chart ─────────────────────────────────────────────── */
    const pd = inf.precipitation;
    const precipCtx = document.getElementById("precipChart")?.getContext("2d");
    if (precipCtx) {
      new Chart(precipCtx, tsConfig(
        pd.dates,
        [
          {
            label: "Actual Precipitation", data: pd.actual,
            borderColor: "#3fb950", backgroundColor: "#3fb95018",
            borderWidth: 1.8, pointRadius: 0, fill: true, tension: 0.3,
          },
          {
            label: "Predicted Precipitation", data: pd.predicted,
            borderColor: "#f0c000", borderWidth: 1.8,
            pointRadius: 0, borderDash: [5, 3], fill: false, tension: 0.3,
          },
        ],
        "Precipitation (mm)", "mm"
      ));
    }
  }

  /* 6. Climate OLS chart ─────────────────────────────────────────────────── */
  const climateCtx = document.getElementById("climateChart")?.getContext("2d");
  if (climateCtx) {
    if (climate) {
      new Chart(climateCtx, climateConfig(
        climate.years, climate.yearly_means, climate.regression_line));
    } else {
      // Fallback: hardcoded from real data run
      const years  = [2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025,2026];
      const means  = [25.6138,25.7156,25.7005,26.0578,26.0458,26.0557,26.6474,26.4337,
                      26.7310,26.8426,26.2959,26.0225,26.7723,26.7831,26.2803,25.9500];
      const slope  = 0.048004, intercept = -70.6499;
      const regLine = {};
      years.forEach(y => { regLine[String(y)] = parseFloat((slope*y+intercept).toFixed(4)); });
      new Chart(climateCtx, climateConfig(years, means, regLine));
    }
  }
}

document.addEventListener("DOMContentLoaded", init);
