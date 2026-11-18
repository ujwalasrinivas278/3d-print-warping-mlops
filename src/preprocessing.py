"""Leakage-safe sklearn preprocessing fitted only on the training partition."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL = ["material"]
NUMERICAL = ["nozzle_temperature", "bed_temperature", "printing_speed", "ambient_temperature",
             "fan_speed", "base_area", "layer_height", "nozzle_bed_temperature_difference", "cooling_stress_index"]

def build_preprocessor() -> ColumnTransformer:
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    numerical = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    return ColumnTransformer([("categorical", categorical, CATEGORICAL), ("numerical", numerical, NUMERICAL)])
