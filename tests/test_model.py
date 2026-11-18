from pathlib import Path
import joblib
import pandas as pd
from src.feature_engineering import engineer_features

def test_saved_model_predicts():
    path = Path("models/warping_model.joblib")
    assert path.exists(), "Run python -m src.train before prediction tests"
    job = pd.DataFrame([{"material":"ABS", "nozzle_temperature":245, "bed_temperature":90,
        "printing_speed":70, "ambient_temperature":23, "fan_speed":60, "base_area":1500, "layer_height":.2}])
    model = joblib.load(path); features = engineer_features(job)
    assert model.predict(features)[0] in {"Yes", "No"}
    assert 0 <= model.predict_proba(features)[0].sum() <= 1.000001
