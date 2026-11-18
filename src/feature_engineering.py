"""Shared training and serving feature logic."""
import numpy as np
import pandas as pd

# Relative shrinkage tendency coefficients: PLA 0.22, PETG 0.55, ABS 1.00.
SHRINKAGE_COEFFICIENTS = {"PLA": 0.22, "PETG": 0.55, "ABS": 1.00}
ENGINEERED_FEATURES = ["nozzle_bed_temperature_difference", "cooling_stress_index"]

def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    missing = set(["material", "nozzle_temperature", "bed_temperature", "ambient_temperature", "fan_speed"]) - set(data.columns)
    if missing:
        raise ValueError(f"Missing required feature columns: {sorted(missing)}")
    result = data.copy()
    result["nozzle_bed_temperature_difference"] = result.nozzle_temperature - result.bed_temperature
    tendency = result.material.map(SHRINKAGE_COEFFICIENTS).fillna(0.55)
    # Dimensionless score: combines thermal gradient, cooling, material, and ambient offset.
    result["cooling_stress_index"] = (result["nozzle_bed_temperature_difference"].clip(lower=0) / 150) * \
        (1 + result.fan_speed.clip(0, 100) / 100) * tendency * (1 + (23 - result.ambient_temperature).clip(lower=-10, upper=15) / 50)
    return result
