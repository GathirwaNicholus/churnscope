/**
 * charts.js — ChurnScope dashboard chart renderers
 * Requires Chart.js 4.x loaded via CDN
 */

const PALETTE = {
  amber:     "#f0a500",
  amberFade: "rgba(240,165,0,0.18)",
  red:       "#f85149",
  redFade:   "rgba(248,81,73,0.18)",
  green:     "#3fb950",
  greenFade: "rgba(63,185,80,0.18)",
  blue:      "#58a6ff",
  blueFade:  "rgba(88,166,255,0.15)",
  muted:     "#8b949e",
  border:    "#2a3140",
  text:      "#e6edf3",
  surface:   "#1e242d",
};

Chart.defaults.color = PALETTE.muted;
Chart.defaults.borderColor = PALETTE.border;
Chart.defaults.font.family = "'IBM Plex Sans', sans-serif";

function baseOptions(extra = {}) {
  return {
    responsive: true,
    plugins: {
      legend: {
        labels: { color: PALETTE.muted, boxWidth: 12, font: { size: 12 } },
      },
      tooltip: {
        backgroundColor: "#161b22",
        borderColor: PALETTE.border,
        borderWidth: 1,
        titleColor: PALETTE.text,
        bodyColor: PALETTE.muted,
        padding: 10,
      },
    },
    scales: {
      x: {
        grid: { color: PALETTE.border },
        ticks: { color: PALETTE.muted, font: { size: 11 } },
      },
      y: {
        grid: { color: PALETTE.border },
        ticks: { color: PALETTE.muted, font: { size: 11 } },
      },
    },
    ...extra,
  };
}

function initDashboardCharts(data) {

  // 1 ── Tenure distribution (grouped bar)
  new Chart(document.getElementById("tenureChart"), {
    type: "bar",
    data: {
      labels: data.tenure_dist.labels,
      datasets: [
        {
          label: "Churned",
          data: data.tenure_dist.churned,
          backgroundColor: PALETTE.redFade,
          borderColor: PALETTE.red,
          borderWidth: 1,
        },
        {
          label: "Retained",
          data: data.tenure_dist.retained,
          backgroundColor: PALETTE.blueFade,
          borderColor: PALETTE.blue,
          borderWidth: 1,
        },
      ],
    },
    options: {
      ...baseOptions(),
      plugins: {
        ...baseOptions().plugins,
        title: { display: false },
      },
    },
  });

  // 2 ── Churn rate by contract (horizontal bar)
  const contractLabels = Object.keys(data.contract_churn);
  const contractVals   = Object.values(data.contract_churn);
  new Chart(document.getElementById("contractChart"), {
    type: "bar",
    data: {
      labels: contractLabels,
      datasets: [{
        label: "Churn Rate (%)",
        data: contractVals,
        backgroundColor: [PALETTE.redFade, PALETTE.amberFade, PALETTE.greenFade],
        borderColor:      [PALETTE.red, PALETTE.amber, PALETTE.green],
        borderWidth: 1,
      }],
    },
    options: {
      ...baseOptions({ indexAxis: "y" }),
      plugins: { legend: { display: false }, tooltip: baseOptions().plugins.tooltip },
      scales: {
        x: { ...baseOptions().scales.x, min: 0, max: 100,
             ticks: { ...baseOptions().scales.x.ticks, callback: v => v + "%" } },
        y: { ...baseOptions().scales.y },
      },
    },
  });

  // 3 ── Churn rate by internet service
  const intLabels = Object.keys(data.internet_churn);
  const intVals   = Object.values(data.internet_churn);
  new Chart(document.getElementById("internetChart"), {
    type: "bar",
    data: {
      labels: intLabels,
      datasets: [{
        label: "Churn Rate (%)",
        data: intVals,
        backgroundColor: [PALETTE.blueFade, PALETTE.redFade, PALETTE.greenFade],
        borderColor:      [PALETTE.blue, PALETTE.red, PALETTE.green],
        borderWidth: 1,
      }],
    },
    options: {
      ...baseOptions({ indexAxis: "y" }),
      plugins: { legend: { display: false }, tooltip: baseOptions().plugins.tooltip },
      scales: {
        x: { ...baseOptions().scales.x, min: 0, max: 100,
             ticks: { ...baseOptions().scales.x.ticks, callback: v => v + "%" } },
        y: { ...baseOptions().scales.y },
      },
    },
  });

  // 4 ── Monthly charges distribution
  new Chart(document.getElementById("chargeChart"), {
    type: "bar",
    data: {
      labels: data.charge_data.labels,
      datasets: [
        {
          label: "Churned",
          data: data.charge_data.churned,
          backgroundColor: PALETTE.redFade,
          borderColor: PALETTE.red,
          borderWidth: 1,
        },
        {
          label: "Retained",
          data: data.charge_data.retained,
          backgroundColor: PALETTE.blueFade,
          borderColor: PALETTE.blue,
          borderWidth: 1,
        },
      ],
    },
    options: baseOptions(),
  });

  // 5 ── Feature Importances (horizontal bar)
  const featLabels = data.feat_imp.map(f => f[0]);
  const featVals   = data.feat_imp.map(f => +(f[1] * 100).toFixed(2));
  new Chart(document.getElementById("featChart"), {
    type: "bar",
    data: {
      labels: featLabels,
      datasets: [{
        label: "Importance (%)",
        data: featVals,
        backgroundColor: PALETTE.amberFade,
        borderColor: PALETTE.amber,
        borderWidth: 1,
      }],
    },
    options: {
      ...baseOptions({ indexAxis: "y" }),
      plugins: { legend: { display: false }, tooltip: baseOptions().plugins.tooltip },
    },
  });

  // 6 ── Confusion Matrix (rendered as HTML cells)
  const cm = data.cm;
  const cmWrap = document.getElementById("cmWrapper");
  if (cmWrap) {
    const tn = cm[0][0], fp = cm[0][1], fn = cm[1][0], tp = cm[1][1];
    cmWrap.innerHTML = `
      <div class="cm-cell cm-tn">
        <div class="cm-val">${tn}</div>
        <div class="cm-label">True Negative</div>
      </div>
      <div class="cm-cell cm-fp">
        <div class="cm-val">${fp}</div>
        <div class="cm-label">False Positive</div>
      </div>
      <div class="cm-cell cm-fn">
        <div class="cm-val">${fn}</div>
        <div class="cm-label">False Negative</div>
      </div>
      <div class="cm-cell cm-tp">
        <div class="cm-val">${tp}</div>
        <div class="cm-label">True Positive</div>
      </div>
    `;
  }

  // 7 ── ROC Curve
  new Chart(document.getElementById("rocChart"), {
    type: "line",
    data: {
      labels: data.roc.fpr,
      datasets: [
        {
          label: "ROC Curve",
          data: data.roc.tpr,
          borderColor: PALETTE.amber,
          backgroundColor: PALETTE.amberFade,
          fill: true,
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.3,
        },
        {
          label: "Random Classifier",
          data: data.roc.fpr,
          borderColor: PALETTE.muted,
          borderDash: [4, 4],
          borderWidth: 1,
          pointRadius: 0,
          fill: false,
        },
      ],
    },
    options: {
      ...baseOptions(),
      scales: {
        x: { ...baseOptions().scales.x, title: { display: true, text: "False Positive Rate", color: PALETTE.muted } },
        y: { ...baseOptions().scales.y, title: { display: true, text: "True Positive Rate",  color: PALETTE.muted } },
      },
    },
  });
}
