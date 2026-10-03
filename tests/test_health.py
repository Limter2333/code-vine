import pytest
from fastapi.testclient import TestClient

from code_vine.main import app


@pytest.fixture()
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert isinstance(body["app"], str)
    assert body["app"]


def test_root_lists_endpoints(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "Welcome to code-vine"
    assert body["health"] == "/health"
