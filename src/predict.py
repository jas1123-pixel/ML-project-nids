"""Score a CSV of connection records.

Usage:
    python -m src.predict data/sample_traffic.csv --out reports/predictions.csv
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd

from .data import COLUMNS
from .detector import HybridDetector

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="CSV with the 41 NSL-KDD feature columns")
    ap.add_argument("--model", default=str(ROOT / "models" / "nids_hybrid.joblib"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "predictions.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    det = HybridDetector.load(args.model)
    res = det.predict_frame(df[COLUMNS])
    out = pd.concat([df.reset_index(drop=True), res.reset_index(drop=True)], axis=1)
    out.to_csv(args.out, index=False)

    print(res["verdict"].value_counts().to_string())
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
