# Dify Hemodynamic API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `api_server.py` (FastAPI) that exposes `HemodynamicPredictor` as a public REST API, deployable to Render, callable from Dify Cloud via HTTP Request node.

**Architecture:** A standalone FastAPI app that wraps `HemodynamicPredictor` directly (no LangGraph). The existing `app.py` (local UI) is untouched. Both files share `hemodynamic_predictor.py` as the core engine.

**Tech Stack:** FastAPI, uvicorn, pytest, httpx (test client), Render (deployment)

---

## File Map

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `api_server.py` | FastAPI app: 3 endpoints, error handler, predictor singleton |
| Create | `requirements.txt` | Production deps: fastapi + uvicorn |
| Create | `render.yaml` | Render Web Service config |
| Create | `tests/conftest.py` | Add project root to sys.path for imports |
| Create | `tests/test_api_server.py` | Endpoint tests using TestClient |

**Do not modify:** `app.py`, `hemodynamic_predictor.py`, `hemodynamic_graph.py`

---

## Task 1: Requirements and test infrastructure

**Files:**
- Create: `requirements.txt`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create `requirements.txt`**

```
fastapi
uvicorn[standard]
```

- [ ] **Step 2: Install production deps**

```bash
pip install fastapi "uvicorn[standard]"
```

Expected: installs without errors. `python -c "import fastapi; import uvicorn; print('ok')"` prints `ok`.

- [ ] **Step 3: Install test deps**

```bash
pip install pytest httpx
```

Expected: installs without errors.

- [ ] **Step 4: Create `tests/conftest.py`**

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
```

- [ ] **Step 5: Commit**

```bash
git add requirements.txt tests/conftest.py
git commit -m "chore: add requirements and test infrastructure"
```

---

## Task 2: Health endpoint (TDD)

**Files:**
- Create: `api_server.py` (skeleton + health endpoint)
- Modify: `tests/test_api_server.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_api_server.py`:

```python
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from api_server import app
    with TestClient(app) as c:
        yield c


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert isinstance(data["rows_loaded"], int)
    assert data["rows_loaded"] > 0
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
pytest tests/test_api_server.py::test_health_returns_ok -v
```

Expected: `ModuleNotFoundError: No module named 'api_server'`

- [ ] **Step 3: Create `api_server.py` with health endpoint**

```python
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from hemodynamic_predictor import HemodynamicPredictor


_predictor: HemodynamicPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor
    _predictor = HemodynamicPredictor()
    yield
    _predictor = None


app = FastAPI(title="Hemodynamic Predictor API", lifespan=lifespan)


@app.exception_handler(HTTPException)
async def _http_exc(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


def _get() -> HemodynamicPredictor:
    if _predictor is None:
        raise RuntimeError("Predictor not initialized")
    return _predictor


class PredictRequest(BaseModel):
    inputs: dict[str, float] = Field(
        description="Known hemodynamic summary values, e.g. {'SBP': 100, 'MAP': 90}"
    )
    outputs: list[str] = Field(
        description="Summary columns to predict, e.g. ['sPAP', 'dPAP']"
    )
    tolerance: float = Field(default=1.0, ge=0.0)
    top_k: int = Field(default=3, ge=1, le=10)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "rows_loaded": len(_get().rows)}
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
pytest tests/test_api_server.py::test_health_returns_ok -v
```

Expected: `PASSED` (loading 40k rows takes ~2–5 seconds first time)

- [ ] **Step 5: Commit**

```bash
git add api_server.py tests/test_api_server.py
git commit -m "feat: add FastAPI skeleton and health endpoint"
```

---

## Task 3: Metadata endpoint (TDD)

**Files:**
- Modify: `api_server.py` (add `/api/metadata`)
- Modify: `tests/test_api_server.py`

- [ ] **Step 1: Write failing test** — add to `tests/test_api_server.py`:

```python
def test_metadata_returns_columns(client):
    response = client.get("/api/metadata")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["row_count"], int)
    assert data["row_count"] > 0
    cols = [c["name"] for c in data["summary_columns"]]
    assert "SBP" in cols
    assert "sPAP" in cols
    assert "dPAP" in cols
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
pytest tests/test_api_server.py::test_metadata_returns_columns -v
```

Expected: `FAILED` — 404 Not Found

- [ ] **Step 3: Add metadata endpoint to `api_server.py`** — add after the `health()` function:

```python
@app.get("/api/metadata")
def api_metadata() -> dict[str, Any]:
    return _get().metadata()
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
pytest tests/test_api_server.py::test_metadata_returns_columns -v
```

Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
git add api_server.py tests/test_api_server.py
git commit -m "feat: add /api/metadata endpoint"
```

---

## Task 4: Predict endpoint — success case (TDD)

**Files:**
- Modify: `api_server.py` (add `POST /api/predict`)
- Modify: `tests/test_api_server.py`

- [ ] **Step 1: Write failing test** — add to `tests/test_api_server.py`:

```python
def test_predict_returns_top_cases(client):
    response = client.post("/api/predict", json={
        "inputs": {"SBP": 100, "MAP": 90},
        "outputs": ["sPAP", "dPAP"],
        "tolerance": 1.0,
        "top_k": 3,
    })
    assert response.status_code == 200
    data = response.json()
    assert data["inputs"] == {"SBP": 100.0, "MAP": 90.0}
    assert "sPAP" in data["outputs"]
    assert "dPAP" in data["outputs"]
    assert isinstance(data["candidate_count"], int)
    assert data["candidate_count"] > 0
    assert len(data["top"]) <= 3
    for case in data["top"]:
        assert "sPAP" in case["outputs"]
        assert "dPAP" in case["outputs"]
        assert 0.0 <= case["confidence_score"] <= 1.0


def test_predict_top_k_respected(client):
    response = client.post("/api/predict", json={
        "inputs": {"SBP": 100, "MAP": 90},
        "outputs": ["sPAP", "dPAP"],
        "tolerance": 5.0,
        "top_k": 2,
    })
    assert response.status_code == 200
    assert len(response.json()["top"]) <= 2
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_api_server.py::test_predict_returns_top_cases tests/test_api_server.py::test_predict_top_k_respected -v
```

Expected: both `FAILED` — 404 or 405

- [ ] **Step 3: Add predict endpoint to `api_server.py`** — add after `api_metadata()`:

```python
@app.post("/api/predict")
def api_predict(req: PredictRequest) -> dict[str, Any]:
    try:
        return _get().predict(
            inputs=req.inputs,
            outputs=req.outputs,
            tolerance=req.tolerance,
            top_k=req.top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_api_server.py::test_predict_returns_top_cases tests/test_api_server.py::test_predict_top_k_respected -v
```

Expected: both `PASSED`

- [ ] **Step 5: Commit**

```bash
git add api_server.py tests/test_api_server.py
git commit -m "feat: add POST /api/predict endpoint"
```

---

## Task 5: Predict endpoint — error cases (TDD)

**Files:**
- Modify: `tests/test_api_server.py`

(No changes to `api_server.py` needed — `HTTPException` + the exception handler already return `{"error": "..."}`)

- [ ] **Step 1: Write failing tests** — add to `tests/test_api_server.py`:

```python
def test_predict_unknown_input_returns_400(client):
    response = client.post("/api/predict", json={
        "inputs": {"UNKNOWN_FIELD_XYZ": 100},
        "outputs": ["sPAP"],
    })
    assert response.status_code == 400
    assert "error" in response.json()


def test_predict_output_equals_input_returns_400(client):
    response = client.post("/api/predict", json={
        "inputs": {"SBP": 100},
        "outputs": ["SBP"],
    })
    assert response.status_code == 400
    assert "error" in response.json()


def test_predict_no_inputs_returns_422(client):
    # FastAPI validates the request schema before our code runs
    response = client.post("/api/predict", json={
        "inputs": {},
        "outputs": ["sPAP"],
    })
    # HemodynamicPredictor raises ValueError for empty inputs → 400
    assert response.status_code == 400
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_api_server.py::test_predict_unknown_input_returns_400 tests/test_api_server.py::test_predict_output_equals_input_returns_400 tests/test_api_server.py::test_predict_no_inputs_returns_422 -v
```

Expected: failing (wrong status codes or missing `error` key)

- [ ] **Step 3: Run tests to confirm they pass**

```bash
pytest tests/test_api_server.py -v
```

Expected: all 7 tests `PASSED` — the exception handler added in Task 2 already handles this correctly.

If any test fails, check that the `_http_exc` exception handler is registered on `app` (it should be from Task 2, Step 3). No code changes needed.

- [ ] **Step 4: Commit**

```bash
git add tests/test_api_server.py
git commit -m "test: add error case coverage for /api/predict"
```

---

## Task 6: Render deployment config

**Files:**
- Create: `render.yaml`

- [ ] **Step 1: Create `render.yaml`**

```yaml
services:
  - type: web
    name: hemodynamic-api
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn api_server:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: "3.11.0"
```

- [ ] **Step 2: Smoke-test the server locally**

```bash
uvicorn api_server:app --host 127.0.0.1 --port 8001
```

In a second terminal:

```bash
curl http://127.0.0.1:8001/health
# Expected: {"status":"ok","rows_loaded":40000}

curl http://127.0.0.1:8001/api/metadata | python -m json.tool | head -20
# Expected: JSON with row_count and summary_columns list

curl -X POST http://127.0.0.1:8001/api/predict \
  -H "Content-Type: application/json" \
  -d '{"inputs":{"SBP":100,"MAP":90},"outputs":["sPAP","dPAP"],"tolerance":1.0,"top_k":3}'
# Expected: JSON with "top" array containing sPAP and dPAP values
```

- [ ] **Step 3: Stop the server (Ctrl+C) and commit**

```bash
git add render.yaml
git commit -m "feat: add Render deployment config"
```

---

## Task 7: Push to GitHub and deploy on Render

- [ ] **Step 1: Create GitHub repo and push** (if not already done)

```bash
git remote add origin https://github.com/<your-username>/summer-twin.git
git push -u origin main
```

- [ ] **Step 2: Deploy on Render**

1. Go to https://render.com → New → Web Service
2. Connect your GitHub repo
3. Render auto-detects `render.yaml` — confirm settings and click **Deploy**
4. Wait for build + deploy (~3–5 min). Watch the build log for `Loaded 40000 cases`.

- [ ] **Step 3: Verify deployed endpoints**

Replace `<your-app>` with your actual Render subdomain:

```bash
curl https://<your-app>.onrender.com/health
# Expected: {"status":"ok","rows_loaded":40000}

curl -X POST https://<your-app>.onrender.com/api/predict \
  -H "Content-Type: application/json" \
  -d '{"inputs":{"SBP":100,"MAP":90},"outputs":["sPAP","dPAP"],"tolerance":1.0,"top_k":3}'
# Expected: JSON with top predictions
```

- [ ] **Step 4: Configure Dify workflow**

In your Dify workflow (https://cloud.dify.ai/app/30618799-274e-46bc-a1a6-24c7f3c7a5de/workflow):

**LLM Node — Query Reconstruction:**

System prompt:
```
你是一个心血管参数提取助手。从用户输入中提取已知的血流动力学指标值和需要预测的指标名称。

严格使用以下指标名称（区分大小写）：
SBP, DBP, MAP, PP, sPAP, dPAP, mPAP, EDV_lv, ESV_lv, SV_lv, EF_lv, LVSP, LVEDP, SW_lv,
EDV_rv, ESV_rv, SV_rv, EF_rv, RVSP, RVDP, CO, HR, RAP, LAP, PCWP, SVR, PVR,
mean_Psvn, mean_Ppvn, RF_mi, RF_ao, RF_ti, RF_po

只输出如下 JSON，不要任何解释文字：
{
  "inputs": {"指标名": 数值},
  "outputs": ["指标名"],
  "tolerance": 1.0,
  "top_k": 3
}
```

User message: `{{#sys.query#}}`

**HTTP Request Node:**
- Method: `POST`
- URL: `https://<your-app>.onrender.com/api/predict`
- Headers: `Content-Type: application/json`
- Body (raw): `{{#llm_node.text#}}`

- [ ] **Step 5: Test Dify workflow end-to-end**

Input: `我的SBP是100，MAP是90，帮我预测一下sPAP和dPAP`

Expected: HTTP Request node returns JSON with `top` array containing sPAP and dPAP predictions.
