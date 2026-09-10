from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tscp.config import load_config
from tscp.experiments import _dataset_display_name


MODEL_LABELS = {
    "ridge": "Ridge Regression",
    "gradient_boosting": "Gradient Boosting",
    "dnn": "DNNs",
    "mlp": "DNNs",
    "random_forest": "Random Forest",
    "logistic": "Logistic Regression",
    "extra_trees": "Extra Trees",
}


def _load_run(value: str) -> pd.DataFrame:
    path = Path(value).resolve()
    metrics = path / "metrics.csv" if path.is_dir() else path
    config_path = metrics.parent / "config.resolved.yaml"
    if not metrics.is_file():
        raise FileNotFoundError(f"Cannot find metrics file: {metrics}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Cannot find resolved config beside metrics: {config_path}")
    config = load_config(config_path)
    if config.get("experiment", {}).get("type") != "model_ablation":
        raise ValueError(f"Not a model_ablation run: {metrics}")
    frame = pd.read_csv(metrics)
    required = {
        "panel", "x", "dataset", "model", "seed", "coverage",
        "corrected_bound", "average_size", "budget",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{metrics} is missing columns: {sorted(missing)}")
    frame["task"] = str(config["task"])
    frame["source"] = str(metrics)
    return frame


def _choose_dataset(frame: pd.DataFrame, task: str, requested: str | None) -> str:
    available = sorted(frame.loc[frame.task == task, "dataset"].astype(str).unique())
    if requested is not None:
        if requested not in available:
            raise ValueError(f"Dataset {requested!r} not found for {task}; available={available}")
        return requested
    if len(available) != 1:
        raise ValueError(
            f"Expected exactly one {task} dataset, found {available}; select one with "
            f"--{task}-dataset."
        )
    return available[0]


def main() -> Path:
    parser = argparse.ArgumentParser(
        description="Combine separately tuned model-ablation runs into a 2x2 figure."
    )
    parser.add_argument("--inputs", nargs="+", required=True, help="Run directories or metrics.csv files.")
    parser.add_argument("--output-dir", default="outputs/model_ablation_2x2")
    parser.add_argument("--regression-dataset")
    parser.add_argument("--classification-dataset")
    parser.add_argument("--figsize", nargs=2, type=float, default=(12.0, 8.0))
    parser.add_argument("--font-size", type=float, default=11.0)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    frame = pd.concat([_load_run(value) for value in args.inputs], ignore_index=True)
    duplicate_keys = ["task", "dataset", "model", "seed", "panel", "x"]
    duplicates = frame.duplicated(duplicate_keys, keep=False)
    if duplicates.any():
        collided = frame.loc[duplicates, duplicate_keys + ["source"]]
        raise ValueError(
            "Multiple supplied runs contain the same model/seed/calibration point. "
            "Keep only the chosen tuned run for each model.\n" + collided.to_string(index=False)
        )

    regression_dataset = _choose_dataset(frame, "regression", args.regression_dataset)
    classification_dataset = _choose_dataset(frame, "classification", args.classification_dataset)
    selected = frame[
        ((frame.task == "regression") & (frame.dataset == regression_dataset))
        | ((frame.task == "classification") & (frame.dataset == classification_dataset))
    ].copy()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output / "model_ablation_2x2_input_points.csv", index=False)

    coverage_summary = selected[selected.panel == "coverage"].groupby(
        ["task", "dataset", "model", "panel", "x"], as_index=False,
    ).agg(
        empirical_coverage=("coverage", "mean"),
        theoretical_coverage=("corrected_bound", "mean"),
        outer_seeds=("seed", "nunique"),
    )
    size_summary = selected[selected.panel == "size"].groupby(
        ["task", "dataset", "model", "panel", "x"], as_index=False,
    ).agg(
        average_size=("average_size", "mean"),
        prechosen_size=("budget", "mean"),
        outer_seeds=("seed", "nunique"),
    )
    summary = pd.concat([coverage_summary, size_summary], ignore_index=True, sort=False)

    preferred = ["ridge", "logistic", "gradient_boosting", "dnn", "mlp", "random_forest", "extra_trees"]
    models = list(dict.fromkeys(preferred + sorted(summary.model.astype(str).unique())))
    models = [model for model in models if model in set(summary.model.astype(str))]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_by_model = {model: colors[index % len(colors)] for index, model in enumerate(models)}

    plt.rcParams.update({"font.size": args.font_size})
    fig, axes = plt.subplots(2, 2, figsize=tuple(args.figsize), squeeze=False)
    columns = [("regression", regression_dataset), ("classification", classification_dataset)]
    for col, (task, dataset) in enumerate(columns):
        part = summary[(summary.task == task) & (summary.dataset == dataset)]
        coverage_part = part[part.panel == "coverage"]
        size_part = part[part.panel == "size"]
        for model, model_part in coverage_part.groupby("model"):
            model_part = model_part.sort_values("x")
            color = color_by_model[str(model)]
            axes[0, col].plot(
                model_part.x, model_part.empirical_coverage,
                color=color, marker="o", label=MODEL_LABELS.get(str(model), str(model)),
            )
            axes[0, col].plot(
                model_part.x, model_part.theoretical_coverage,
                color=color, linestyle="--", label="_nolegend_",
            )
            model_size = size_part[size_part.model == model].sort_values("x")
            axes[1, col].plot(
                model_size.x, model_size.average_size,
                color=color, marker="o", label=MODEL_LABELS.get(str(model), str(model)),
            )
        budget_curve = size_part.groupby("x", as_index=False).prechosen_size.mean()
        axes[1, col].plot(
            budget_curve.x, budget_curve.prechosen_size,
            color="black", linestyle="--", label="Pre-chosen set size",
        )
        axes[0, col].set_title(_dataset_display_name(dataset).replace("CaliforniaHousing", "California Housing"))
        axes[0, col].set_xlabel("Number of test samples")
        axes[1, col].set_xlabel(r"Total calibration size $2n$")
        if col == 0:
            axes[0, col].set_ylabel("Coverage")
            axes[1, col].set_ylabel("Average set size")
        for ax in axes[:, col]:
            ax.grid(alpha=0.25)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

    model_handles = [
        Line2D([], [], color=color_by_model[model], marker="o", label=MODEL_LABELS.get(model, model))
        for model in models
    ]
    top_handles = model_handles + [
        Line2D([], [], color="black", linestyle="-", label="Empirical"),
        Line2D([], [], color="black", linestyle="--", label="Theoretical"),
    ]
    bottom_handles = [Line2D([], [], color="black", linestyle="--", label="Pre-chosen set size")] + model_handles
    axes[0, 1].legend(handles=top_handles, loc="center left", bbox_to_anchor=(1.04, 0.5), frameon=False)
    axes[1, 1].legend(handles=bottom_handles, loc="center left", bbox_to_anchor=(1.04, 0.5), frameon=False)

    summary.to_csv(output / "model_ablation_2x2_points.csv", index=False)
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    fig.savefig(output / "model_ablation_2x2.pdf", bbox_inches="tight")
    fig.savefig(output / "model_ablation_2x2.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(output)
    return output


if __name__ == "__main__":
    main()
