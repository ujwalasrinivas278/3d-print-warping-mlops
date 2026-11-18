import pandas as pd
from src.feature_engineering import engineer_features
from src.preprocessing import build_preprocessor

def test_engineered_values_and_reuse():
    row = pd.DataFrame([{"material":"ABS", "nozzle_temperature":245, "bed_temperature":90,
        "printing_speed":70, "ambient_temperature":23, "fan_speed":60, "base_area":1500, "layer_height":.2}])
    first = engineer_features(row); second = engineer_features(row)
    assert first.loc[0, "nozzle_bed_temperature_difference"] == 155
    assert first.loc[0, "cooling_stress_index"] > 0
    assert first.equals(second)

def test_preprocessor_imputes_and_encodes():
    jobs = pd.DataFrame([
        {"material":"ABS", "nozzle_temperature":245, "bed_temperature":90, "printing_speed":70,
         "ambient_temperature":23, "fan_speed":60, "base_area":1500, "layer_height":.2},
        {"material":None, "nozzle_temperature":None, "bed_temperature":80, "printing_speed":55,
         "ambient_temperature":22, "fan_speed":40, "base_area":1200, "layer_height":.2},
    ])
    transformed = build_preprocessor().fit_transform(engineer_features(jobs))
    assert transformed.shape[0] == 2
    assert transformed.shape[1] > 8
