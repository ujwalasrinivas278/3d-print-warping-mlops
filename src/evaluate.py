"""Metrics and plots for model comparison."""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, ConfusionMatrixDisplay)

def calculate_metrics(model, X_test, y_test):
    pred = model.predict(X_test)
    return {"accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, pos_label="Yes", zero_division=0),
            "recall": recall_score(y_test, pred, pos_label="Yes", zero_division=0),
            "f1": f1_score(y_test, pred, pos_label="Yes", zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, pred, labels=["No", "Yes"]).tolist(),
            "classification_report": classification_report(y_test, pred, labels=["No", "Yes"], zero_division=0, output_dict=True)}

def save_evaluation_plots(results, y_test, predictions, output_dir="report"):
    root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
    for name, pred in predictions.items():
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(y_test, pred, labels=["No", "Yes"], display_labels=["No defect", "Warping"], cmap="Blues", ax=ax, colorbar=False)
        ax.set_title(f"{name} confusion matrix"); fig.tight_layout(); fig.savefig(root / f"confusion_{name.lower().replace(' ', '_')}.png", dpi=160); plt.close(fig)
    frame = pd.DataFrame(results).T[["accuracy", "precision", "recall", "f1"]]
    ax = frame.plot(kind="bar", ylim=(0, 1), figsize=(8, 5), ylabel="Score", title="Holdout model comparison")
    ax.legend(loc="lower right"); ax.figure.tight_layout(); ax.figure.savefig(root / "model_comparison.png", dpi=160); plt.close(ax.figure)
