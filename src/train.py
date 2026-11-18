"""Train, compare, track and save the selected end-to-end sklearn pipeline."""
import json
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from .data_validation import validate_dataset
from .evaluate import calculate_metrics, save_evaluation_plots
from .feature_store import load_features
from .preprocessing import build_preprocessor

SEED = 42

def train(dataset_version=1, feature_dir="feature_store", models_dir="models", report_dir="report"):
    df = load_features(dataset_version, feature_dir); validate_dataset(df)
    X = df.drop(columns="warping_defect"); y = df["warping_defect"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=SEED, stratify=y)
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=250, max_depth=10, min_samples_leaf=2,
            class_weight="balanced", random_state=SEED, n_jobs=-1),
    }
    results, predictions, fitted = {}, {}, {}
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("3D_Printing_Warping_Prediction")
    for name, estimator in candidates.items():
        pipeline = Pipeline([("preprocessor", build_preprocessor()), ("classifier", estimator)])
        pipeline.fit(X_train, y_train); scores = calculate_metrics(pipeline, X_test, y_test)
        predictions[name] = pipeline.predict(X_test); results[name] = scores; fitted[name] = pipeline
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_param("model_name", name); mlflow.log_param("dataset_version", f"v{dataset_version}")
            mlflow.log_params({f"model_{k}": v for k, v in estimator.get_params(deep=False).items() if isinstance(v, (str, int, float, bool)) or v is None})
            for metric in ("accuracy", "precision", "recall", "f1"): mlflow.log_metric(metric, scores[metric])
            mlflow.log_dict({"confusion_matrix_labels": ["No", "Yes"], "matrix": scores["confusion_matrix"], "classification_report": scores["classification_report"]}, "evaluation.json")
            mlflow.log_artifact(Path("data") / f"dataset_v{dataset_version}.csv", artifact_path="dataset")
            mlflow.sklearn.log_model(pipeline, artifact_path="model")
            scores["run_id"] = run.info.run_id
    # Prioritize defect recall; use F1 and then accuracy to break ties.
    selected = max(results, key=lambda key: (results[key]["recall"], results[key]["f1"], results[key]["accuracy"]))
    output = Path(models_dir); output.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[selected], output / "warping_model.joblib")
    (output / "model_metadata.json").write_text(json.dumps({"registered_name": "WarpingPredictionModel", "selected_model": selected,
        "dataset_version": f"v{dataset_version}", "positive_class": "Yes", "thresholds": {"medium": .40, "high": .70},
        "metrics": {k: {m: v[m] for m in ("accuracy", "precision", "recall", "f1")} for k,v in results.items()}}, indent=2), encoding="utf-8")
    save_evaluation_plots(results, y_test, predictions, report_dir)
    # Register only the selected candidate in the local SQLite Model Registry.
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    with mlflow.start_run(run_name=f"Registry - {selected}") as run:
        mlflow.log_param("selected_model", selected); mlflow.log_param("dataset_version", f"v{dataset_version}")
        mlflow.log_metrics({m: results[selected][m] for m in ("accuracy", "precision", "recall", "f1")})
        mlflow.sklearn.log_model(fitted[selected], artifact_path="selected_model", registered_model_name="WarpingPredictionModel")
    return results, selected

if __name__ == "__main__":
    scores, winner = train()
    print(json.dumps({"selected_model": winner, "metrics": {k: {m: v[m] for m in ("accuracy", "precision", "recall", "f1")} for k,v in scores.items()}}, indent=2))
