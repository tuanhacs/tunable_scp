from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tscp.config import add_plot_arguments, apply_overrides, apply_plot_arguments, load_config
from tscp.experiments import make_figures, seeds_for_dataset


def _points_path(value: str) -> Path:
    path = Path(value).resolve()
    if path.is_file():
        return path
    if path.is_dir():
        preferred = path / "budget_ablation_points.csv"
        fallback = path / "metrics.csv"
        if preferred.is_file():
            return preferred
        if fallback.is_file():
            return fallback
    raise FileNotFoundError(
        f"Cannot find budget_ablation_points.csv or metrics.csv from {path}."
    )


def main() -> Path:
    parser = argparse.ArgumentParser(
        description="Re-match and replot an existing budget-ablation point cloud."
    )
    parser.add_argument(
        "--input", required=True,
        help="Run directory, budget_ablation_points.csv, or metrics.csv.",
    )
    parser.add_argument(
        "--config",
        help="Config used for matching; defaults to config.resolved.yaml beside the input CSV.",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory; defaults to the directory containing the input CSV.",
    )
    parser.add_argument(
        "--set", action="append", default=[], dest="overrides",
        help="Override a dotted config key before matching and plotting.",
    )
    add_plot_arguments(parser)
    args = parser.parse_args()

    points_path = _points_path(args.input)
    config_path = (
        Path(args.config).resolve()
        if args.config
        else points_path.parent / "config.resolved.yaml"
    )
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Cannot find config {config_path}; provide one with --config."
        )
    config = apply_overrides(load_config(config_path), args.overrides)
    config = apply_plot_arguments(config, args)
    if config.get("experiment", {}).get("type") != "budget_ablation":
        raise ValueError(f"Not a budget_ablation config: {config_path}")

    frame = pd.read_csv(points_path)
    required = {
        "dataset", "budget_type", "budget_title", "seed", "batch",
        "method", "coverage", "average_size", "average_budget",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{points_path} is missing columns: {sorted(missing)}")

    selected_parts = []
    for dataset in config["datasets"]:
        allowed_seeds = seeds_for_dataset(config, dataset)
        part = frame[
            (frame.dataset == dataset)
            & frame.seed.astype(int).isin(allowed_seeds)
        ]
        if part.empty:
            raise ValueError(
                f"The input contains no points for dataset={dataset!r}, "
                f"seeds={allowed_seeds}."
            )
        selected_parts.append(part)
    frame = pd.concat(selected_parts, ignore_index=True)

    output = Path(args.output_dir).resolve() if args.output_dir else points_path.parent
    output.mkdir(parents=True, exist_ok=True)
    make_figures(frame, config, output)
    print(output)
    return output


if __name__ == "__main__":
    main()
