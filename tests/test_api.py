from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_forecast_endpoint(monkeypatch):
    class DummyModel:
        def get_forecast(self, steps):
            class Pred:
                predicted_mean = [0.1] * steps
            return Pred()

    monkeypatch.setenv('PORTFOLIO_CSV', '')
    monkeypatch.setattr('src.models.utils.load_model', lambda name: DummyModel())
    response = client.post("/forecast", json={"horizon": 3})
    assert response.status_code == 200
    data = response.json()
    assert data['horizon'] == 3
    assert len(data['forecast']) == 3