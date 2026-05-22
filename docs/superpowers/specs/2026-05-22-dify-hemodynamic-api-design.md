# Dify Hemodynamic API — Design Spec

**Date:** 2026-05-22
**Status:** Approved

## Goal

Expose `hemodynamic_predictor.py` as a publicly accessible REST API so that Dify Cloud (cloud.dify.ai) can call it from a workflow. The existing local UI (`app.py`) is preserved unchanged.

## Architecture

Two independent servers sharing the same core files:

```
Local machine (manual use)
└── app.py  (Python http.server, port 8000)
    ├── GET  /              → Web UI
    └── POST /api/predict   → prediction via LangGraph

Render cloud (Dify integration)
└── api_server.py  (FastAPI + uvicorn, port=$PORT)
    ├── GET  /health        → health check
    ├── GET  /api/metadata  → available columns and value ranges
    └── POST /api/predict   → prediction (calls HemodynamicPredictor directly)

Shared, unchanged files
├── hemodynamic_predictor.py
├── hemodynamic_graph.py
└── dataset_40k_cases.csv
```

`api_server.py` calls `HemodynamicPredictor.predict()` directly — no LangGraph layer — keeping the API server simple and stateless.

## API Endpoints

### `POST /api/predict`

Request body:
```json
{
  "inputs": {"SBP": 100, "MAP": 90},
  "outputs": ["sPAP", "dPAP"],
  "tolerance": 1.0,
  "top_k": 3
}
```

Success response (HTTP 200):
```json
{
  "inputs": {"SBP": 100, "MAP": 90},
  "outputs": ["sPAP", "dPAP"],
  "mode": "conditional_kde_real_samples",
  "candidate_count": 25,
  "top": [
    {
      "case_id": "...",
      "confidence_score": 0.82,
      "outputs": {"sPAP": 28.3, "dPAP": 12.1}
    }
  ]
}
```

Error response (HTTP 400):
```json
{"error": "Unknown input summary: XYZ"}
```

### `GET /api/metadata`

Returns all available column names, value ranges, and labels. Used by Dify's LLM node prompt to know the valid field names.

```json
{
  "row_count": 40000,
  "summary_columns": [
    {"name": "SBP", "zh": "收缩压", "unit": "mmHg", "min": 60, "max": 200},
    ...
  ]
}
```

### `GET /health`

```json
{"status": "ok", "rows_loaded": 40000}
```

## New Files

| File | Purpose |
|------|---------|
| `api_server.py` | FastAPI app, the only new source file |
| `requirements.txt` | Python dependencies for Render |
| `render.yaml` | Render deployment config |

### `requirements.txt`
```
fastapi
uvicorn[standard]
```

(`hemodynamic_predictor.py` uses only the Python standard library; `pydantic` is bundled with FastAPI.)

### `render.yaml`
```yaml
services:
  - type: web
    name: hemodynamic-api
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn api_server:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
```

## Render Deployment Steps (manual, one-time)

1. Push repo to GitHub
2. Render → New Web Service → connect GitHub repo
3. Render reads `render.yaml` automatically
4. After deploy, get public URL e.g. `https://hemodynamic-api.onrender.com`
5. Paste URL into Dify HTTP Request node

Note: Render free tier sleeps after 15 min of inactivity; cold start ~30s. Acceptable for demo.

## Dify Workflow Design

```
User natural language input
        ↓
[LLM Node] Query Reconstruction
  Extract known hemodynamic values and prediction targets → output JSON
        ↓
[HTTP Request Node]
  POST https://hemodynamic-api.onrender.com/api/predict
  Body: JSON from previous LLM node
        ↓
[LLM Node] Result Formatter (optional)
  Translate prediction results into natural language
        ↓
Output to user
```

### Query Reconstruction prompt template

```
你是一个心血管参数提取助手。
从用户输入中提取已知的血流动力学指标值，以及用户想预测的指标名称。

可用指标名称（必须严格使用这些名称）：
SBP, DBP, MAP, PP, HR, CO, SV, SVR, PVR, sPAP, dPAP, mPAP, PCWP, CVP, ...
（完整列表见 GET /api/metadata）

输出格式（只输出 JSON，不要任何解释）：
{
  "inputs": {"指标名": 数值, ...},
  "outputs": ["指标名", ...],
  "tolerance": 1.0,
  "top_k": 3
}

用户输入：{{#sys.query#}}
```

### HTTP Request node config

- Method: `POST`
- URL: `https://hemodynamic-api.onrender.com/api/predict`
- Headers: `Content-Type: application/json`
- Body: `{{#query_reconstruction.text#}}`

## Out of Scope

- Authentication / API keys (not needed for demo)
- Rate limiting
- Persistent logging
- Multi-tool agent (future: waveform generation, PVLoop extraction)
