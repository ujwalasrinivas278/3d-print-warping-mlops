from pathlib import Path
import pandas as pd
from src.data_generation import generate_dataset, save_datasets
from src.data_validation import validate_dataset

def test_generation_is_reproducible_and_sized(tmp_path):
    assert generate_dataset(1).equals(generate_dataset(1))
    assert len(generate_dataset(2)) >= 250
    saved = save_datasets(tmp_path)
    assert len(saved[1]) >= 250 and len(saved[2]) >= 250
    assert (tmp_path / "dataset_v1.csv").exists()

def test_validation_rejects_invalid_material():
    df = generate_dataset(1); df.loc[0, "material"] = "NYLON"
    try:
        validate_dataset(df)
    except ValueError as exc:
        assert "Invalid materials" in str(exc)
    else:
        raise AssertionError("Invalid material should fail validation")

def test_committed_datasets_valid():
    for version in (1, 2):
        path = Path("data") / f"dataset_v{version}.csv"
        if path.exists(): validate_dataset(pd.read_csv(path), min_records=250)
