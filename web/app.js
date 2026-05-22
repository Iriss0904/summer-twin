let metadata = null;

const inputBody = document.getElementById("inputTableBody");
const outputBody = document.getElementById("outputTableBody");
const datasetMeta = document.getElementById("datasetMeta");
const runMeta = document.getElementById("runMeta");
const statusBox = document.getElementById("status");
const resultWrap = document.getElementById("resultWrap");
const predictButton = document.getElementById("predictButton");

function fmt(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "";
  return Number(value).toFixed(digits);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderTables() {
  const defaults = metadata.defaults || {};
  const defaultInputs = defaults.inputs || {};
  const defaultOutputs = new Set(defaults.outputs || []);

  inputBody.innerHTML = metadata.summary_columns
    .map((col) => {
      const value = defaultInputs[col.name] ?? "";
      return `
        <tr>
          <td class="metric">${escapeHtml(col.name)}</td>
          <td>${escapeHtml(col.en || col.name)}</td>
          <td class="muted">${escapeHtml(col.unit || "")}</td>
          <td class="range">${fmt(col.min)} - ${fmt(col.max)}</td>
          <td>
            <input
              class="summary-input"
              data-col="${escapeHtml(col.name)}"
              type="number"
              step="any"
              value="${escapeHtml(value)}"
            />
          </td>
        </tr>`;
    })
    .join("");

  outputBody.innerHTML = metadata.summary_columns
    .map((col) => {
      const checked = defaultOutputs.has(col.name) ? "checked" : "";
      return `
        <tr>
          <td>
            <input
              class="output-check"
              data-col="${escapeHtml(col.name)}"
              type="checkbox"
              ${checked}
            />
          </td>
          <td class="metric">${escapeHtml(col.name)}</td>
          <td>${escapeHtml(col.en || col.name)}</td>
          <td class="muted">${escapeHtml(col.unit || "")}</td>
          <td class="range">${fmt(col.min)} - ${fmt(col.max)}</td>
        </tr>`;
    })
    .join("");

  document.querySelectorAll(".summary-input").forEach((el) => {
    el.addEventListener("input", syncOutputAvailability);
  });
  syncOutputAvailability();
}

function collectInputs() {
  const inputs = {};
  document.querySelectorAll(".summary-input").forEach((el) => {
    const raw = el.value.trim();
    if (raw !== "") inputs[el.dataset.col] = Number(raw);
  });
  return inputs;
}

function collectOutputs() {
  return Array.from(document.querySelectorAll(".output-check"))
    .filter((el) => el.checked && !el.disabled)
    .map((el) => el.dataset.col);
}

function syncOutputAvailability() {
  const usedInputs = new Set(Object.keys(collectInputs()));
  document.querySelectorAll(".output-check").forEach((el) => {
    if (usedInputs.has(el.dataset.col)) {
      el.checked = false;
      el.disabled = true;
      el.closest("tr").classList.add("muted");
    } else {
      el.disabled = false;
      el.closest("tr").classList.remove("muted");
    }
  });
}

function setStatus(message, kind = "") {
  statusBox.textContent = message;
  statusBox.className = `status ${kind}`.trim();
}

async function predict() {
  const payload = {
    inputs: collectInputs(),
    outputs: collectOutputs(),
    tolerance: Number(document.getElementById("toleranceInput").value || 1),
    top_k: Number(document.getElementById("topKInput").value || 3),
  };

  predictButton.disabled = true;
  setStatus("Running...");
  runMeta.textContent = "";
  resultWrap.innerHTML = "";

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok || result.error) throw new Error(result.error || "Request failed");
    renderResult(result);
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    predictButton.disabled = false;
  }
}

function renderResult(result) {
  const source =
    result.candidate_source === "hard_matches" ? "hard matches" : "weighted neighbors";
  runMeta.textContent = `${result.mode} - ${source} - hard n=${result.hard_match_count} - candidates=${result.candidate_count}`;

  if (!result.top || result.top.length === 0) {
    setStatus("No result", "warn");
    return;
  }

  setStatus("Done");

  const inputCols = Object.keys(result.inputs);
  const outputCols = result.outputs;

  const head = `
    <thead>
      <tr>
        <th>Rank</th>
        <th>Score</th>
        <th>Case ID</th>
        <th>Input distance</th>
        ${inputCols.map((col) => `<th>${escapeHtml(col)}</th>`).join("")}
        ${outputCols.map((col) => `<th>${escapeHtml(col)}</th>`).join("")}
      </tr>
    </thead>`;

  const body = result.top
    .map((row, index) => {
      const inputCells = inputCols
        .map((col) => `<td>${fmt(row.input_values[col], 3)}</td>`)
        .join("");
      const outputCells = outputCols
        .map((col) => `<td>${fmt(row.outputs[col], 3)}</td>`)
        .join("");
      return `
        <tr>
          <td>${index + 1}</td>
          <td class="score">${fmt(row.confidence_score, 3)}</td>
          <td class="metric">${escapeHtml(row.case_id)}</td>
          <td>${fmt(row.input_distance, 4)}</td>
          ${inputCells}
          ${outputCells}
        </tr>`;
    })
    .join("");

  const notes = result.top
    .filter((row) => row.cluster_id !== undefined)
    .map(
      (row, index) =>
        `Top ${index + 1}: cluster ${row.cluster_id}, size ${row.cluster_size}, transform ${row.transform}`
    );

  resultWrap.innerHTML = `
    <table>${head}<tbody>${body}</tbody></table>
    ${notes.length ? `<div class="result-note">${escapeHtml(notes.join(" - "))}</div>` : ""}
  `;
}

document.getElementById("clearInputsButton").addEventListener("click", () => {
  document.querySelectorAll(".summary-input").forEach((el) => {
    el.value = "";
  });
  syncOutputAvailability();
});

document.getElementById("clearOutputsButton").addEventListener("click", () => {
  document.querySelectorAll(".output-check").forEach((el) => {
    el.checked = false;
  });
});

document.getElementById("selectDefaultButton").addEventListener("click", () => {
  document.querySelectorAll(".output-check").forEach((el) => {
    el.checked = ["sPAP", "dPAP"].includes(el.dataset.col) && !el.disabled;
  });
});

predictButton.addEventListener("click", predict);

fetch("/api/metadata")
  .then((response) => response.json())
  .then((data) => {
    metadata = data;
    datasetMeta.textContent = `${data.row_count} cases - ${data.summary_columns.length} summary columns`;
    document.getElementById("toleranceInput").value = data.defaults.tolerance;
    document.getElementById("topKInput").value = data.defaults.top_k;
    renderTables();
  })
  .catch((error) => {
    datasetMeta.textContent = "Metadata failed";
    setStatus(error.message, "error");
  });
