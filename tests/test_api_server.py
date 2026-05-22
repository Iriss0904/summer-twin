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
