"""Persist reusable engineered datasets and lightweight version metadata."""
from pathlib import Path
import json
import pandas as pd
from .feature_engineering import engineer_features

def build_feature_store(data_dir="data", store_dir="feature_store"):
    data_root, store_root = Path(data_dir), Path(store_dir); store_root.mkdir(parents=True, exist_ok=True)
    for version in (1, 2):
        source = pd.read_csv(data_root / f"dataset_v{version}.csv")
        engineered = engineer_features(source)
        engineered.to_csv(store_root / f"features_v{version}.csv", index=False)
    (store_root / "metadata.json").write_text(json.dumps({"feature_logic_version": "1.0", "engineered_features": ["nozzle_bed_temperature_difference", "cooling_stress_index"], "source_versions": ["v1", "v2"]}, indent=2), encoding="utf-8")

def load_features(version: int, store_dir="feature_store") -> pd.DataFrame:
    return pd.read_csv(Path(store_dir) / f"features_v{version}.csv")

if __name__ == "__main__":
    build_feature_store()
