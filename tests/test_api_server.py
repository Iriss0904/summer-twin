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
