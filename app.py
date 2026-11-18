"""FastAPI service for pre-print warping risk predictions."""
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from src.feature_engineering import engineer_features
from src.data_validation import RANGES

MODEL_PATH = Path("models/warping_model.joblib")
app = FastAPI(title="3D Printing Warping Prediction", version="1.0.0")
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

class PrintJob(BaseModel):
    material: str
    nozzle_temperature: float = Field(..., ge=150, le=300)
    bed_temperature: float = Field(..., ge=20, le=130)
    printing_speed: float = Field(..., ge=5, le=150)
    ambient_temperature: float = Field(..., ge=0, le=50)
    fan_speed: float = Field(..., ge=0, le=100)
    base_area: float = Field(..., ge=50, le=10000)
    layer_height: float = Field(..., ge=.05, le=.5)

    @field_validator("material")
    @classmethod
    def check_material(cls, value):
        if value not in {"PLA", "ABS", "PETG"}: raise ValueError("material must be PLA, ABS, or PETG")
        return value

@app.get("/health")
def health():
    return {"status": "healthy" if model is not None else "model_not_loaded", "model": "WarpingPredictionModel"}

@app.post("/predict")
def predict(job: PrintJob):
    if model is None: raise HTTPException(status_code=503, detail="Model is not trained. Run python -m src.train first.")
    features = engineer_features(pd.DataFrame([job.model_dump()]))
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    classes = list(model.classes_); probability = float(probabilities[classes.index("Yes")])
    risk = "High" if probability >= .70 else "Medium" if probability >= .40 else "Low"
    return {"prediction": "Warping Risk" if prediction == "Yes" else "No Warping Risk",
            "print_conditions_acceptable": bool(prediction == "No"), "warping_probability": round(probability, 4), "risk_level": risk}
