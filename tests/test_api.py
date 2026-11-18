from fastapi.testclient import TestClient
from app import app

client = TestClient(app)
SAMPLE = {"material":"ABS", "nozzle_temperature":245, "bed_temperature":90, "printing_speed":70,
          "ambient_temperature":23, "fan_speed":60, "base_area":1500, "layer_height":.2}

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model"] == "WarpingPredictionModel"

def test_prediction_endpoint():
    response = client.post("/predict", json=SAMPLE)
    assert response.status_code == 200
    result = response.json()
    assert result["prediction"] in {"Warping Risk", "No Warping Risk"}
    assert 0 <= result["warping_probability"] <= 1
    assert result["risk_level"] in {"Low", "Medium", "High"}

def test_invalid_input():
    invalid = {**SAMPLE, "fan_speed": 120}
    assert client.post("/predict", json=invalid).status_code == 422
