"""Hybrid intrusion detector.

Layer 1 (supervised)  : classifies a connection as normal / dos / probe / r2l / u2r.
Layer 2 (unsupervised): Isolation Forest trained on *normal* traffic only. It flags
                        connections that look unlike anything benign, which is how
                        the system can catch attacks it was never trained on
                        (zero-day / novel behaviour).
"""
from __future__ import annotations

import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

from .data import CATEGORICAL, COLUMNS, NUMERIC

HEAVY_TAILED = ["duration", "src_bytes", "dst_bytes", "total_bytes", "bytes_ratio"]
ENGINEERED = ["total_bytes", "bytes_ratio", "error_rate_sum", "host_pressure"]
NUMERIC_ALL = NUMERIC + ENGINEERED


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Domain-driven features that make attacks easier to separate."""
    out = df[COLUMNS].copy()
    out["total_bytes"] = out["src_bytes"] + out["dst_bytes"]
    out["bytes_ratio"] = (out["src_bytes"] + 1) / (out["dst_bytes"] + 1)
    out["error_rate_sum"] = out["serror_rate"] + out["rerror_rate"]
    out["host_pressure"] = out["count"] * out["dst_host_count"] / 256.0
    for c in HEAVY_TAILED:
        out[c] = np.log1p(out[c])
    return out


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", StandardScaler(), NUMERIC_ALL),
    ])


def make_classifier(name: str, seed: int = 42):
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                                n_jobs=-1, random_state=seed),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                                                 class_weight="balanced",
                                                                 random_state=seed),
        "mlp": MLPClassifier(hidden_layer_sizes=(96, 48), max_iter=60, random_state=seed),
    }
    return models[name]


MODEL_NAMES = ["logistic_regression", "random_forest", "hist_gradient_boosting", "mlp"]


class _Features:
    """Tiny sklearn-style step so raw dataframes can go straight into a Pipeline."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return add_features(X)

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def get_params(self, deep=True):
        return {}

    def set_params(self, **p):
        return self

    def __sklearn_is_fitted__(self):
        return True


def build_pipeline(name: str, seed: int = 42) -> Pipeline:
    return Pipeline([("features", _Features()), ("prep", build_preprocessor()),
                     ("clf", make_classifier(name, seed))])


class HybridDetector:
    def __init__(self, model_name: str = "random_forest", anomaly_quantile: float = 0.03,
                 seed: int = 42):
        self.model_name = model_name
        self.anomaly_quantile = anomaly_quantile
        self.seed = seed

    # ------------------------------------------------------------------ fit
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HybridDetector":
        self.pipeline_ = build_pipeline(self.model_name, self.seed).fit(X, y)
        self.classes_ = list(self.pipeline_.classes_)

        # Anomaly layer: normal traffic only (uses the fitted feature/prep steps)
        prep = Pipeline(self.pipeline_.steps[:-1])
        Z = prep.transform(X[y.values == "normal"])
        self.iforest_ = IsolationForest(n_estimators=200, max_samples=512,
                                        random_state=self.seed, n_jobs=-1).fit(Z)
        scores = self.iforest_.score_samples(Z)
        # Threshold: flag the ~3% least "normal-looking" benign training samples and worse
        self.anomaly_threshold_ = float(np.quantile(scores, self.anomaly_quantile))
        self._prep_ = prep
        return self

    # -------------------------------------------------------------- predict
    def anomaly_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Isolation-Forest anomaly score: higher = less like normal traffic."""
        s = -self.iforest_.score_samples(self._prep_.transform(X))
        return s

    def predict_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        proba = self.pipeline_.predict_proba(X)
        pred = np.array(self.classes_)[proba.argmax(axis=1)]
        conf = proba.max(axis=1)
        anom = self.anomaly_scores(X)
        is_anom = anom > -self.anomaly_threshold_

        verdict = np.where(
            pred != "normal", "ALERT: " + pd.Series(pred).str.upper().values,
            np.where(is_anom, "SUSPICIOUS: possible unknown attack", "Normal"))
        out = pd.DataFrame({
            "prediction": pred,
            "confidence": conf.round(4),
            "attack_probability": (1 - proba[:, self.classes_.index("normal")]).round(4),
            "anomaly_score": anom.round(4),
            "is_anomaly": is_anom,
            "verdict": verdict,
        }, index=X.index)
        return out

    def predict_supervised(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline_.predict(X)

    # ----------------------------------------------------------- persistence
    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "HybridDetector":
        return joblib.load(path)
