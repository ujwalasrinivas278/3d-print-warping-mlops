"""Schema and range checks for generated, stored, and incoming job data."""
import pandas as pd
from .data_generation import FEATURES, MATERIALS

RANGES = {"nozzle_temperature": (150, 300), "bed_temperature": (20, 130), "printing_speed": (5, 150),
          "ambient_temperature": (0, 50), "fan_speed": (0, 100), "base_area": (50, 10000), "layer_height": (.05, .5)}

def validate_dataset(df: pd.DataFrame, min_records: int = 1) -> None:
    missing_cols = set(FEATURES + ["warping_defect"]) - set(df.columns)
    if missing_cols: raise ValueError(f"Missing columns: {sorted(missing_cols)}")
    if len(df) < min_records: raise ValueError(f"Expected at least {min_records} records, got {len(df)}")
    if df["material"].isna().any(): raise ValueError("material contains missing values")
    invalid = set(df.material.dropna().unique()) - set(MATERIALS)
    if invalid: raise ValueError(f"Invalid materials: {sorted(invalid)}")
    for col, (low, high) in RANGES.items():
        values = pd.to_numeric(df[col], errors="coerce")
        if values.isna().any(): raise ValueError(f"{col} contains missing or non-numeric values")
        if ((values < low) | (values > high)).any(): raise ValueError(f"{col} must be between {low} and {high}")
    if not set(df.warping_defect.dropna().unique()).issubset({"Yes", "No"}):
        raise ValueError("warping_defect must contain only Yes/No")
