const form = document.getElementById("prediction-form");
const statusPill = document.getElementById("status-pill");
const forecastReturn = document.getElementById("forecast-return");
const riskProbability = document.getElementById("risk-probability");
const riskLevel = document.getElementById("risk-level");
const regressionBars = document.getElementById("regression-bars");
const classificationBars = document.getElementById("classification-bars");
const riskDrivers = document.getElementById("risk-drivers");
const leaderboards = document.getElementById("leaderboards");
const sampleTable = document.getElementById("sample-table");

function formPayload() {
  const entries = new FormData(form).entries();
  const payload = {};
  for (const [key, value] of entries) {
    if (["market_regime", "asset_class", "region"].includes(key)) {
      payload[key] = value;
      continue;
    }
    payload[key] = Number(value);
  }
  return payload;
}

function renderBars(container, values, format = (value) => value.toFixed(3)) {
  container.innerHTML = "";
  Object.entries(values).forEach(([label, value]) => {
    const row = document.createElement("div");
    row.className = "bar-row";
    const percentage = Math.max(0, Math.min(1, Math.abs(value)));
    row.innerHTML = `
      <header><strong>${label}</strong><span>${format(value)}</span></header>
      <div class="bar-track"><div class="bar-fill" style="width: ${percentage * 100}%"></div></div>
    `;
    container.appendChild(row);
  });
}

function renderRiskDrivers(items) {
  riskDrivers.innerHTML = "";
  items.forEach((item) => {
    const row = document.createElement("div");
    row.className = "driver-item";
    row.innerHTML = `<strong>${item.label}</strong><span>${item.score.toFixed(3)}</span>`;
    riskDrivers.appendChild(row);
  });
}

function leaderboardCard(title, rows, metricKey) {
  const card = document.createElement("article");
  card.className = "leaderboard-card";
  card.innerHTML = `<h3>${title}</h3>`;
  rows.forEach((row) => {
    const line = document.createElement("div");
    line.className = "leaderboard-row";
    line.innerHTML = `<strong>${row.name}</strong><span>${row.metrics[metricKey].toFixed(3)}</span>`;
    card.appendChild(line);
  });
  return card;
}

function riskPill(level) {
  const normalized = level.toLowerCase();
  return `<span class="risk-pill ${normalized}">${level}</span>`;
}

async function runPrediction() {
  try {
    statusPill.textContent = "Scoring";
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(formPayload()),
    });
    const data = await response.json();
    forecastReturn.textContent = `${data.forecast_return_pct.toFixed(2)}%`;
    riskProbability.textContent = `${(data.risk_probability * 100).toFixed(1)}%`;
    riskLevel.textContent = data.risk_level;
    renderBars(regressionBars, data.regression_model_outputs, (value) => `${(value * 100).toFixed(2)}%`);
    renderBars(classificationBars, data.classification_model_outputs, (value) => `${(value * 100).toFixed(1)}%`);
    renderRiskDrivers(data.top_risk_drivers);
    statusPill.textContent = "Models ready";
  } catch (error) {
    statusPill.textContent = "Prediction failed";
    console.error(error);
  }
}

async function loadSummary() {
  try {
    const response = await fetch("/api/summary");
    const data = await response.json();
    leaderboards.innerHTML = "";
    leaderboards.appendChild(leaderboardCard("Return Forecasting", data.metadata.regression.leaderboard, "r2"));
    leaderboards.appendChild(leaderboardCard("Risk Signal Detection", data.metadata.classification.leaderboard, "roc_auc"));
  } catch (error) {
    console.error(error);
  }
}

async function loadSampleData() {
  try {
    const response = await fetch("/api/sample-data?limit=8");
    const data = await response.json();
    sampleTable.innerHTML = `
      <div class="sample-header">
        <span>Segment</span>
        <span>Region</span>
        <span>Pred Return</span>
        <span>Risk Prob</span>
        <span>Risk Level</span>
      </div>
    `;
    data.forEach((row) => {
      const item = document.createElement("div");
      item.className = "sample-row";
      item.innerHTML = `
        <strong>${row.asset_class} / ${row.market_regime}</strong>
        <span>${row.region}</span>
        <span>${row.predicted_return_pct.toFixed(2)}%</span>
        <span>${(row.risk_probability * 100).toFixed(1)}%</span>
        ${riskPill(row.risk_level)}
      `;
      sampleTable.appendChild(item);
    });
  } catch (error) {
    console.error(error);
  }
}

async function retrainModels() {
  statusPill.textContent = "Retraining";
  const response = await fetch("/api/train", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ rows: 1400, random_seed: 42 }),
  });
  if (!response.ok) {
    statusPill.textContent = "Retrain failed";
    return;
  }
  await Promise.all([loadSummary(), loadSampleData(), runPrediction()]);
  statusPill.textContent = "Artifacts refreshed";
}

document.getElementById("predict-button").addEventListener("click", runPrediction);
document.getElementById("refresh-metrics").addEventListener("click", async () => {
  await Promise.all([loadSummary(), loadSampleData()]);
});
document.getElementById("train-button").addEventListener("click", retrainModels);

window.addEventListener("DOMContentLoaded", async () => {
  await Promise.all([loadSummary(), loadSampleData(), runPrediction()]);
});
