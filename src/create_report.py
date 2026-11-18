"""Build a compact implementation report from actual run artifacts and metrics."""
import json
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                Image, PageBreak, Preformatted, KeepTogether)

def create_report(output="report/implementation_report.pdf"):
    root=Path(".")
    metadata_path=root/"models/model_metadata.json"
    metadata=json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else None
    comparison_path=root/"report/dataset_comparison.json"
    if comparison_path.exists():
        versions=json.loads(comparison_path.read_text(encoding="utf-8"))
    else:
        import pandas as pd
        versions={}
        for v in (1,2):
            df=pd.read_csv(root/"data"/f"dataset_v{v}.csv")
            versions[f"v{v}"]={"records":len(df),"mean_printing_speed":round(float(df.printing_speed.mean()),3),
                "mean_bed_temperature":round(float(df.bed_temperature.mean()),3),"defect_count":int((df.warping_defect=="Yes").sum()),
                "defect_percentage":round(float((df.warping_defect=="Yes").mean()*100),2),"materials":df.material.value_counts().to_dict()}
        comparison_path.write_text(json.dumps(versions,indent=2),encoding="utf-8")
    styles=getSampleStyleSheet(); styles.add(ParagraphStyle(name="CoverTitle", parent=styles["Title"], alignment=TA_CENTER, fontSize=24, leading=30, textColor=colors.HexColor("#16324F")))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=10))
    doc=SimpleDocTemplate(output, pagesize=letter, rightMargin=.65*inch,leftMargin=.65*inch,topMargin=.65*inch,bottomMargin=.65*inch)
    story=[]
    story += [Spacer(1,1.1*inch),Paragraph("MLOps Project",styles["CoverTitle"]),Paragraph("Predicting Warping Defects in 3D-Printed Components",styles["Heading1"]),Spacer(1,.25*inch),Paragraph("Implementation Report",styles["Heading2"]),Spacer(1,.4*inch),Paragraph("Prepared for a college MLOps laboratory submission",styles["BodyText"]),Spacer(1,.3*inch),Paragraph("Generated: 18 November 2026",styles["BodyText"]),Spacer(1,2*inch),Paragraph("Dataset and labels are synthetic and intended only for educational experimentation. They are not validated for real manufacturing decisions.",styles["Italic"]),PageBreak()]
    def section(title, body): story.extend([Paragraph(title,styles["Heading1"]),Paragraph(body,styles["BodyText"]),Spacer(1,8)])
    section("Abstract", "This project implements a reproducible training and serving workflow for classifying synthetic FDM print jobs by warping risk. It creates two data versions, validates and engineers features, compares Logistic Regression and Random Forest, records results in MLflow, saves a complete sklearn pipeline, and exposes predictions via FastAPI.")
    section("Problem statement and objectives", "Warping can be influenced by material, temperatures, speed, cooling, ambient conditions, and geometry. The objective is to demonstrate how these pre-print features can flow through an educational ML lifecycle, with defect recall prioritized in model selection.")
    section("Technologies and architecture", "Python, Pandas, NumPy, scikit-learn, Matplotlib, MLflow, FastAPI, Uvicorn, Joblib, Pytest, and GitHub Actions. Data is generated and validated, features are stored, a train-only fitted preprocessing pipeline feeds classifiers, MLflow tracks experiments, and FastAPI serves the saved pipeline.")
    story.append(Preformatted("Synthetic CSVs -> validation -> engineering -> feature store -> train/test split\n                                                           -> sklearn models -> MLflow + Joblib -> FastAPI",styles["Code"]))
    section("Dataset generation and versions", "Each CSV contains 500 records and eight inputs: material, nozzle temperature, bed temperature, printing speed, ambient temperature, fan speed, base area, and layer height. The Yes/No target is sampled from an illustrative material- and condition-dependent probability formula. V2 increases printing speeds and reduces bed temperatures to represent more challenging synthetic conditions.")
    rows=[["Measure","Version 1","Version 2"],["Records",versions["v1"]["records"],versions["v2"]["records"]],["Mean speed",versions["v1"]["mean_printing_speed"],versions["v2"]["mean_printing_speed"]],["Mean bed temp",versions["v1"]["mean_bed_temperature"],versions["v2"]["mean_bed_temperature"]],["Defect count",versions["v1"]["defect_count"],versions["v2"]["defect_count"]],["Defect %",versions["v1"]["defect_percentage"],versions["v2"]["defect_percentage"]]]
    story.append(Table(rows, repeatRows=1, style=TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#16324F")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),.4,colors.grey),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#EEF3F7")]),("PADDING",(0,0),(-1,-1),6)])))
    story.append(Paragraph("Material counts: " + str({v: versions[v]["materials"] for v in versions}),styles["Small"]))
    if (root/"report/dataset_versions.png").exists():
        story.append(Image("report/dataset_versions.png",width=7.0*inch,height=2.25*inch))
    else:
        story.append(Paragraph("Dataset visualization: pending Matplotlib installation. Tabulated comparison above was calculated directly from both CSV files.",styles["Small"]))
    section("Validation and preprocessing", "Validation rejects missing required values, unknown materials, invalid ranges, and unsupported labels. The ColumnTransformer imputes categorical values with the mode and numeric values with the median, one-hot encodes material, and scales numeric features. It is fitted inside a pipeline only on the training partition; the held-out partition is not used during fitting.")
    section("Feature engineering and feature store", "The feature store preserves engineered copies for each dataset version. The nozzle-bed difference is nozzle temperature minus bed temperature. The cooling stress index combines nonnegative temperature difference, fan fraction, material coefficients (PLA 0.22, PETG 0.55, ABS 1.00), and bounded ambient offset. Training and API serving call the same function.")
    section("Models and evaluation", "The implementation uses a 75/25 stratified split with seed 42. Logistic Regression and a 250-tree Random Forest are configured with class balancing. Selection prioritizes recall for Yes (defect), then F1 and accuracy. Recall matters because a missed defective print is costly; precision remains relevant because false alarms also have a cost. Metrics are only included after the holdout evaluation is actually run.")
    table=[["Model","Accuracy","Precision","Recall","F1"]]
    if metadata:
        for name, vals in metadata["metrics"].items(): table.append([name]+[f"{vals[k]:.3f}" for k in ("accuracy","precision","recall","f1")])
    else:
        table.extend([["Logistic Regression","Pending","Pending","Pending","Pending"],["Random Forest","Pending","Pending","Pending","Pending"]])
    story.append(Table(table,repeatRows=1,style=TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#16324F")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),.4,colors.grey),("PADDING",(0,0),(-1,-1),6)])))
    story.append(Paragraph(f"Selected model: {metadata['selected_model']} (positive class recall-first rule)." if metadata else "Training was not executed in the current environment because the required scikit-learn and MLflow packages could not be installed here. No metrics are reported as results.",styles["BodyText"]))
    for image in ["confusion_logistic_regression.png","confusion_random_forest.png","model_comparison.png"]:
        path=root/"report"/image
        if path.exists(): story.append(Image(str(path),width=5.8*inch,height=3.4*inch))
    section("MLflow and model registration", "Training code targets experiment 3D_Printing_Warping_Prediction and local SQLite tracking. Both candidate runs are configured to log their parameters, dataset version, holdout metrics, confusion matrix/report, dataset, and model artifact; selected model registration is configured as WarpingPredictionModel. Execution evidence is pending. Once dependencies are installed and training completes, start the UI with: mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000.")
    section("FastAPI deployment and prediction", "GET /health reports API and model availability. POST /predict validates inputs, applies the shared feature engineering function, calls the saved model and returns predicted class, probability, risk band, and acceptability. Run uvicorn app:app --reload and use /docs to try requests. Risk cutoffs (.40 and .70) are display bands, not optimized operational thresholds.")
    section("CI/CD, tests, and evidence", "The GitHub Actions workflow is configured to install dependencies, regenerate and validate data, create features, train, run Pytest, and compile modules on push and pull request. Dataset generation and validation executed locally. Model, API, and Pytest execution remain pending because dependencies are unavailable here. No GitHub cloud run is claimed. Screenshot section: pending evidence. Capture genuine screenshots after running the MLflow UI and API locally; no tool screenshots are fabricated here.")
    section("Conclusion and future enhancements", "The project demonstrates data versioning, validation, leakage-safe preprocessing, shared feature logic, model comparison, experiment tracking, registry, API serving, and automated CI checks. Its major limitation is synthetic data. Next steps include controlled lab data collection, printer-specific validation, calibrated probabilities, cost-based threshold setting, drift monitoring, access controls, and deployment hardening.")
    doc.build(story)

if __name__ == "__main__": create_report()
