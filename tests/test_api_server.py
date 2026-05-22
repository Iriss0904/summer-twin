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
