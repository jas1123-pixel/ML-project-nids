"""Train, compare and evaluate the intrusion-detection models.

Usage:
    python -m src.train                       # synthetic NSL-KDD-like data
    python -m src.train --data data/KDDTrain+.txt   # real NSL-KDD
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from .data import CLASSES, COLUMNS, generate_synthetic, load_dataset
from .detector import MODEL_NAMES, HybridDetector, build_pipeline

ROOT = Path(__file__).resolve().parent.parent
REPORTS, MODELS, DATA = ROOT / "reports", ROOT / "models", ROOT / "data"


def binary_metrics(y_true, y_pred_attack) -> dict:
    """Detection rate (recall on attacks) and false-alarm rate (on normal traffic)."""
    is_attack = np.asarray(y_true) != "normal"
    flagged = np.asarray(y_pred_attack).astype(bool)
    tp, fn = (flagged & is_attack).sum(), (~flagged & is_attack).sum()
    fp, tn = (flagged & ~is_attack).sum(), (~flagged & ~is_attack).sum()
    return {
        "detection_rate": float(tp / max(tp + fn, 1)),
        "false_alarm_rate": float(fp / max(fp + tn, 1)),
        "precision": float(tp / max(tp + fp, 1)),
    }


def compare_models(Xtr, ytr, Xte, yte) -> pd.DataFrame:
    rows = []
    for name in MODEL_NAMES:
        t0 = time.time()
        pipe = build_pipeline(name).fit(Xtr, ytr)
        train_s = time.time() - t0
        t0 = time.time()
        pred = pipe.predict(Xte)
        infer_ms = (time.time() - t0) / len(Xte) * 1000
        rows.append({
            "model": name,
            "accuracy": accuracy_score(yte, pred),
            "macro_f1": f1_score(yte, pred, average="macro"),
            "weighted_f1": f1_score(yte, pred, average="weighted"),
            "train_seconds": round(train_s, 1),
            "ms_per_connection": round(infer_ms, 4),
        })
        print(f"  {name:24s} acc={rows[-1]['accuracy']:.4f}  macroF1={rows[-1]['macro_f1']:.4f}")
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)


def plot_confusion(y_true, y_pred, path):
    labels = [c for c in CLASSES if c in set(y_true) | set(y_pred)]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (row-normalised)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]}\n{cmn[i, j]:.0%}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_importance(detector, Xte, yte, path, n=3000, top=15):
    idx = np.random.default_rng(0).choice(len(Xte), size=min(n, len(Xte)), replace=False)
    Xs, ys = Xte.iloc[idx], yte.iloc[idx]
    res = permutation_importance(detector.pipeline_, Xs, ys, scoring="f1_macro",
                                 n_repeats=3, random_state=0, n_jobs=1)
    imp = pd.Series(res.importances_mean, index=Xs.columns).sort_values(ascending=False)
    imp.to_csv(REPORTS / "feature_importance.csv", header=["importance"])
    top_imp = imp.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top_imp.index, top_imp.values, color="#2b6cb0")
    ax.set_xlabel("Drop in macro-F1 when feature is shuffled")
    ax.set_title(f"Top {top} features (permutation importance)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def zero_day_experiment(Xtr, ytr, Xte, yte) -> pd.DataFrame:
    """Hide one attack family from training; can we still catch it?

    supervised_only : classifier trained without that family (it cannot name it)
    hybrid          : classifier OR anomaly layer raises an alert
    """
    rows = []
    for held in [c for c in CLASSES if c != "normal" and c in set(ytr)]:
        keep = ytr != held
        det = HybridDetector("random_forest").fit(Xtr[keep], ytr[keep])
        mask = yte == held
        out = det.predict_frame(Xte[mask])
        sup_flag = out["prediction"] != "normal"
        hyb_flag = sup_flag | out["is_anomaly"]
        normal_out = det.predict_frame(Xte[yte == "normal"])
        fa_sup = float((normal_out["prediction"] != "normal").mean())
        fa_hyb = float(((normal_out["prediction"] != "normal") | normal_out["is_anomaly"]).mean())
        rows.append({
            "unseen_attack_family": held,
            "test_samples": int(mask.sum()),
            "detected_by_supervised": float(sup_flag.mean()),
            "detected_by_hybrid": float(hyb_flag.mean()),
            "false_alarm_supervised": fa_sup,
            "false_alarm_hybrid": fa_hyb,
        })
        print(f"  unseen={held:6s} supervised={rows[-1]['detected_by_supervised']:.2%} "
              f"hybrid={rows[-1]['detected_by_hybrid']:.2%}")
    return pd.DataFrame(rows)


def plot_zero_day(df, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df["detected_by_supervised"], 0.4, label="Supervised only", color="#a0aec0")
    ax.bar(x + 0.2, df["detected_by_hybrid"], 0.4, label="Hybrid (+ anomaly layer)", color="#2b6cb0")
    ax.set_xticks(x, df["unseen_attack_family"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Detection rate on never-seen attack family")
    ax.set_title("Zero-day experiment")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="path to NSL-KDD file (optional)")
    ap.add_argument("--n-samples", type=int, default=30000, help="synthetic sample count")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-zero-day", action="store_true")
    args = ap.parse_args()

    for d in (REPORTS, MODELS, DATA):
        d.mkdir(exist_ok=True)

    df, source = load_dataset(args.data, args.n_samples, args.seed)
    print(f"Data source : {source}\nClass counts:\n{df['label'].value_counts().to_string()}\n")

    X, y = df[COLUMNS], df["label"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=args.seed)

    print("Comparing models...")
    comp = compare_models(Xtr, ytr, Xte, yte)
    comp.to_csv(REPORTS / "model_comparison.csv", index=False)
    best = comp.loc[0, "model"]
    print(f"\nBest model by macro-F1: {best}\nTraining final hybrid detector...")

    det = HybridDetector(best, seed=args.seed).fit(Xtr, ytr)
    out = det.predict_frame(Xte)
    pred = out["prediction"]

    report = classification_report(yte, pred, output_dict=True, zero_division=0)
    supervised_alert = pred != "normal"
    hybrid_alert = supervised_alert | out["is_anomaly"]
    auc = roc_auc_score(yte != "normal", out["attack_probability"])
    metrics = {
        "data_source": source,
        "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
        "best_model": best,
        "accuracy": float(accuracy_score(yte, pred)),
        "macro_f1": float(f1_score(yte, pred, average="macro")),
        "binary_roc_auc": float(auc),
        "supervised_binary": binary_metrics(yte, supervised_alert),
        "hybrid_binary": binary_metrics(yte, hybrid_alert),
        "per_class": {k: v for k, v in report.items() if k in CLASSES},
    }

    plot_confusion(yte, pred, REPORTS / "confusion_matrix.png")
    plot_importance(det, Xte, yte, REPORTS / "feature_importance.png")

    if not args.skip_zero_day:
        print("\nZero-day experiment...")
        zd = zero_day_experiment(Xtr, ytr, Xte, yte)
        zd.to_csv(REPORTS / "zero_day.csv", index=False)
        plot_zero_day(zd, REPORTS / "zero_day.png")
        metrics["zero_day"] = zd.to_dict(orient="records")

    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    det.save(MODELS / "nids_hybrid.joblib")

    # A labelled sample file for the dashboard's upload / replay demo
    sample = Xte.copy()
    sample["label"] = yte
    sample.sample(n=min(1500, len(sample)), random_state=1).to_csv(DATA / "sample_traffic.csv", index=False)

    print("\n=== RESULTS ===")
    print(classification_report(yte, pred, zero_division=0, digits=3))
    print(f"Binary ROC-AUC          : {auc:.4f}")
    print(f"Supervised  det/false   : {metrics['supervised_binary']['detection_rate']:.3f} / "
          f"{metrics['supervised_binary']['false_alarm_rate']:.3f}")
    print(f"Hybrid      det/false   : {metrics['hybrid_binary']['detection_rate']:.3f} / "
          f"{metrics['hybrid_binary']['false_alarm_rate']:.3f}")
    print(f"\nSaved model  -> {MODELS / 'nids_hybrid.joblib'}\nSaved report -> {REPORTS}")


if __name__ == "__main__":
    main()
