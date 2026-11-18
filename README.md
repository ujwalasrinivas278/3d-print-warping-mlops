# Predicting Warping Defects in 3D-Printed Components

An educational end-to-end MLOps project that estimates the chance of FDM print warping before a job starts. All records and labels are synthetic. They support software and ML workflow practice only and must not be used for manufacturing decisions.

## Problem and objectives

Unsuitable temperature, speed, cooling, material, and geometry settings can contribute to warping. This project creates versioned sample data, validates it, engineers reusable features, trains and compares classifiers, records experiments in MLflow, and serves predictions through FastAPI. The API returns a predicted class, warping probability, risk level, and whether conditions are acceptable.

## Technologies and architecture

Python, Pandas, NumPy, scikit-learn, Matplotlib, MLflow, FastAPI, Uvicorn, Joblib, Pytest, and GitHub Actions.

```text
Synthetic data v1/v2 -> validation -> feature engineering -> feature store
                                                |                  |
                                                +-> train/test split -> sklearn pipelines
                                                                      |       |
                                                                   MLflow   Joblib model
                                                                             |
                                                                         FastAPI
```

## Dataset and versions

`data/dataset_v1.csv` and `data/dataset_v2.csv` contain 500 deterministic records each. Features are material (PLA, ABS, PETG), nozzle and bed temperatures, printing speed, ambient temperature, fan speed, base area, and layer height; target is `warping_defect` (Yes/No). V2 increases the speed distribution and reduces the bed temperature distribution. `data/metadata.json` records counts and distributions. Run `python -m src.data_comparison` to calculate and plot mean speed, bed temperature, defect counts/rates, and material distribution (`report/dataset_comparison.json`, `report/dataset_versions.png`). The synthetic target probabilities are constructed from documented assumptions, not measured lab outcomes.

## Validation, preprocessing, and features

`src.data_validation` checks required columns, missing values, allowed materials, target labels, and numeric ranges. The serving schema uses the same limits. `src.preprocessing` uses a ColumnTransformer: most-frequent imputation and one-hot encoding for material, median imputation and standard scaling for numeric features. It is inside each sklearn Pipeline and is fitted only after the stratified train/test split, avoiding leakage.

`src.feature_engineering` adds `nozzle_bed_temperature_difference = nozzle - bed` and a dimensionless cooling stress index combining the temperature difference, fan fraction, material shrinkage coefficient, and ambient offset. Documented relative coefficients are PLA 0.22, PETG 0.55, ABS 1.00. The exact same function is used for saved feature data and API records. Feature Store CSVs in `feature_store/` are versioned by source dataset, with logic metadata.

## Training and evaluation

Run `python -m src.train`. Both Logistic Regression and Random Forest use the same reproducible 75/25 stratified split (seed 42), preprocessing, and held-out evaluation. Metrics and plots are calculated from execution, not constants. The selection rule prioritizes recall for the defective (`Yes`) class, then F1 and accuracy as tie-breakers. Recall matters because a false negative can let a defect-prone job proceed; precision also matters because excessive warnings can make the system less useful. This trade-off is only demonstrated on synthetic data. Model and preprocessing are saved together in `models/warping_model.joblib`.

Metrics: accuracy, warping-class precision, recall, F1, confusion matrices, and classification reports. Outputs include `report/confusion_*.png`, `report/model_comparison.png`, and `models/model_metadata.json`.

## MLflow tracking and registry

Training uses experiment `3D_Printing_Warping_Prediction` with local SQLite tracking (`mlflow.db`), logs metrics, parameters, confusion matrix/report, dataset artifact, and sklearn model artifact for each candidate, then registers the selected model as `WarpingPredictionModel`. SQLite supports the local Model Registry used here. Browse the experiment with:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000`. Run training before starting the UI. Each training run can add a new model version.

## Install and run (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.data_generation
python -m src.data_comparison
python -m src.feature_store
python -m src.train
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000/docs` for interactive API documentation. In another terminal, use:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
$job = @{ material='ABS'; nozzle_temperature=245; bed_temperature=90; printing_speed=70; ambient_temperature=23; fan_speed=60; base_area=1500; layer_height=0.2 } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType 'application/json' -Body $job
```

`GET /health` reports model availability. `POST /predict` validates input and returns `prediction`, `print_conditions_acceptable`, `warping_probability`, and `risk_level` (`Low` < .40, `Medium` .40-.70, `High` >= .70). Risk cutoffs are communication bands, not calibrated decision thresholds.

## Tests and CI

After training, run `python -m pytest -q`. Tests cover generation, size, validation, feature calculations/reuse, model prediction, API health and prediction, and invalid inputs. GitHub Actions in `.github/workflows/mlops-ci.yml` regenerates data, validates, engineers features, trains, tests, and compiles Python on pushes and pull requests. Local checks do not imply that the workflow ran on GitHub; cloud execution requires pushing this folder to a GitHub repository.

## Limitations and next steps

Synthetic probability rules cannot represent printer-specific physics, measurement noise, material batches, or real base rates. There is no model calibration, external validation, drift monitoring, security layer, or production deployment. Future work should use consented lab data, validate per printer/material, tune thresholds against failure costs, calibrate probabilities, and add monitoring and artifact governance. `report/implementation_report.pdf` is the implementation report; screenshot evidence is marked pending unless actually captured during a demonstration.

## Execution record for this workspace

Dataset generation, feature-store creation, and CSV range validation completed locally using the available bundled Python runtime. Actual generated comparison: V1 has mean speed 59.90, mean bed temperature 74.42, and 209 defects (41.8%); V2 has mean speed 68.50, mean bed temperature 65.02, and 260 defects (52.0%). Both versions have 500 records. The current environment did not provide scikit-learn, Matplotlib, MLflow, FastAPI, Uvicorn, Joblib, or Pytest; package installation attempts did not complete. Therefore local model training, plots, MLflow execution/registration, API tests, and Pytest are pending. The report marks unavailable metrics and evidence as pending rather than estimating them. The GitHub workflow is defined, but it has not run on GitHub.
