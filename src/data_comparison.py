"""Dataset version summaries and comparison visualizations."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd

def compare(data_dir="data", report_dir="report"):
    root = Path(data_dir); out = Path(report_dir); out.mkdir(parents=True, exist_ok=True)
    versions = {f"v{v}": pd.read_csv(root / f"dataset_v{v}.csv") for v in (1, 2)}
    summary = {}
    for name, df in versions.items():
        summary[name] = {"records": len(df), "mean_printing_speed": round(float(df.printing_speed.mean()), 3),
            "mean_bed_temperature": round(float(df.bed_temperature.mean()), 3), "defect_count": int((df.warping_defect == "Yes").sum()),
            "defect_percentage": round(float((df.warping_defect == "Yes").mean() * 100), 2),
            "materials": df.material.value_counts().to_dict()}
    (out / "dataset_comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, col in enumerate(["printing_speed", "bed_temperature"]):
        axes[i].boxplot([versions[v][col] for v in ("v1", "v2")], labels=["V1", "V2"])
        axes[i].set_title(col.replace("_", " ").title()); axes[i].set_ylabel("Value")
    materials = sorted(set().union(*(set(df.material.unique()) for df in versions.values())))
    width=.35; x=range(len(materials))
    for idx, version in enumerate(("v1", "v2")):
        counts=versions[version].material.value_counts()
        axes[2].bar([a+(idx-.5)*width for a in x], [counts.get(m,0) for m in materials], width, label=version.upper())
    axes[2].set_xticks(list(x), materials); axes[2].set_title("Material distribution"); axes[2].legend()
    fig.tight_layout(); fig.savefig(out / "dataset_versions.png", dpi=160); plt.close(fig)
    return summary

if __name__ == "__main__":
    print(json.dumps(compare(), indent=2))
