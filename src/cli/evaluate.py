"""Recompute publication metrics from a row-level annotation table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.evaluation import evaluate_dataframe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="CSV containing truth, prediction, x, and y")
    parser.add_argument("output", type=Path, help="Destination one-row metrics CSV")
    parser.add_argument("--truth", required=True, help="Ground-truth column")
    parser.add_argument("--prediction", required=True, help="Predicted-domain column")
    parser.add_argument("--x", default="x", help="Spatial x-coordinate column")
    parser.add_argument("--y", default="y", help="Spatial y-coordinate column")
    parser.add_argument("--dataset", default="", help="Dataset label stored in output")
    parser.add_argument("--sample", default="", help="Sample label stored in output")
    parser.add_argument("--model", default="", help="Model label stored in output")
    parser.add_argument("--replicate", default="", help="Replicate label stored in output")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    frame = pd.read_csv(args.input)
    metrics = evaluate_dataframe(frame, args.truth, args.prediction, args.x, args.y)
    row = {
        "data_type": args.dataset,
        "data_name": args.sample,
        "model_name": args.model,
        "replicate": args.replicate,
        **metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(args.output, index=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
