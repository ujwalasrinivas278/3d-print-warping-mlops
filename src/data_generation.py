"""Generate deterministic synthetic FDM job data for two dataset versions."""
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = ["material", "nozzle_temperature", "bed_temperature", "printing_speed",
            "ambient_temperature", "fan_speed", "base_area", "layer_height"]
MATERIALS = ["PLA", "ABS", "PETG"]

def generate_dataset(version: int, n_records: int = 500, seed: int = 42) -> pd.DataFrame:
    if version not in (1, 2):
        raise ValueError("version must be 1 or 2")
    if n_records < 250:
        raise ValueError("Each dataset version must have at least 250 records")
    rng = np.random.default_rng(seed + version)
    material = rng.choice(MATERIALS, n_records, p=[.45, .30, .25])
    nozzle_base = np.select([material == "PLA", material == "ABS"], [205, 245], 235)
    bed_base = np.select([material == "PLA", material == "ABS"], [60, 95], 75)
    nozzle = np.clip(rng.normal(nozzle_base, 8), 185, 265).round(1)
    bed = np.clip(rng.normal(bed_base - (8 if version == 2 else 0), 9), 35, 110).round(1)
    speed = np.clip(rng.normal(52 + (18 if version == 2 else 0), 12), 25, 110).round(1)
    ambient = np.clip(rng.normal(23, 4, n_records), 12, 35).round(1)
    fan = np.clip(rng.normal(np.select([material == "PLA", material == "ABS"], [75, 25], 45), 15), 0, 100).round(1)
    area = np.clip(rng.lognormal(np.log(1300), .55, n_records), 250, 5000).round(1)
    layer = rng.choice([.12, .16, .20, .24, .28], n_records, p=[.1,.2,.4,.2,.1])
    shrink = np.select([material == "PLA", material == "ABS"], [.22, 1.0], .55)
    temp_stress = np.maximum((nozzle - bed - 120) / 65, 0)
    speed_stress = np.maximum((speed - 50) / 45, 0)
    cooling = np.maximum((fan / 100) * (nozzle - bed) / 150, 0)
    geometry = np.maximum((area - 1000) / 2500, 0) + np.maximum((layer - .2) / .15, 0)
    logit = -2.1 + 1.35 * shrink + 1.05 * temp_stress + .8 * speed_stress + .65 * cooling + .55 * geometry + .035 * (ambient - 23)
    probability = 1 / (1 + np.exp(-logit))
    defect = np.where(rng.random(n_records) < probability, "Yes", "No")
    return pd.DataFrame({"material": material, "nozzle_temperature": nozzle, "bed_temperature": bed,
        "printing_speed": speed, "ambient_temperature": ambient, "fan_speed": fan,
        "base_area": area, "layer_height": layer, "warping_defect": defect})

def save_datasets(data_dir: str | Path = "data", n_records: int = 500) -> dict:
    root = Path(data_dir); root.mkdir(parents=True, exist_ok=True)
    result = {}
    for version in (1, 2):
        df = generate_dataset(version, n_records)
        path = root / f"dataset_v{version}.csv"; df.to_csv(path, index=False)
        result[version] = df
    metadata = {"synthetic": True, "purpose": "educational experimentation only; not for manufacturing decisions",
        "features": FEATURES, "target": "warping_defect", "versions": {}}
    for v, df in result.items():
        metadata["versions"][f"v{v}"] = {"records": len(df), "target_distribution": df.warping_defect.value_counts().to_dict(),
            "material_distribution": df.material.value_counts().to_dict()}
    import json
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return result

if __name__ == "__main__":
    save_datasets()
