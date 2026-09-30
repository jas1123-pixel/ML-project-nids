"""Streamlit dashboard for the Intelligent Network Intrusion Detection System.

Run:  streamlit run app.py
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.data import COLUMNS, generate_synthetic
from src.detector import HybridDetector

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "nids_hybrid.joblib"
REPORTS = ROOT / "reports"

st.set_page_config(page_title="AI Network Intrusion Detection", page_icon="🛡️", layout="wide")


@st.cache_resource
def load_model():
    return HybridDetector.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error("No trained model found. Run `python -m src.train` first.")
    st.stop()

detector = load_model()

SCENARIOS = {
    "Normal office traffic": {"normal": 1.0},
    "Mixed traffic (realistic)": {"normal": 0.80, "dos": 0.10, "probe": 0.06, "r2l": 0.03, "u2r": 0.01},
    "DoS flood attack": {"normal": 0.35, "dos": 0.65},
    "Port scan / recon": {"normal": 0.50, "probe": 0.50},
    "Password guessing / FTP abuse (R2L)": {"normal": 0.70, "r2l": 0.30},
    "Privilege escalation (U2R)": {"normal": 0.80, "u2r": 0.20},
}

st.title("🛡️ Intelligent Network Intrusion Detection")
st.caption("Hybrid ML detector: supervised attack classifier + Isolation-Forest anomaly layer "
           "for unseen (zero-day) behaviour.")

tab_live, tab_upload, tab_report = st.tabs(["📡 Live monitor", "📂 Analyze a CSV", "📊 Model report"])

# --------------------------------------------------------------------------- #
with tab_live:
    c1, c2, c3 = st.columns([2, 1, 1])
    scenario = c1.selectbox("Traffic scenario", list(SCENARIOS))
    batch = c2.slider("Connections per batch", 50, 1000, 300, step=50)
    c3.write("")
    go = c3.button("▶ Capture next batch", type="primary", use_container_width=True)
    if c3.button("Reset log", use_container_width=True):
        st.session_state.pop("log", None)

    if go:
        seed = int(np.random.default_rng().integers(0, 1_000_000))
        raw = generate_synthetic(batch, seed=seed, mix=SCENARIOS[scenario])
        truth = raw.pop("label")
        res = detector.predict_frame(raw[COLUMNS])
        chunk = pd.concat([raw.reset_index(drop=True), res.reset_index(drop=True)], axis=1)
        chunk["true_label"] = truth.values
        prev = st.session_state.get("log", pd.DataFrame())
        st.session_state["log"] = pd.concat([prev, chunk], ignore_index=True)
        st.session_state["last"] = chunk

    if "last" not in st.session_state:
        st.info("Pick a scenario and press **Capture next batch** to simulate incoming traffic.")
    else:
        last, log = st.session_state["last"], st.session_state["log"]
        is_alert = last["verdict"].str.startswith("ALERT")
        is_susp = last["verdict"].str.startswith("SUSPICIOUS")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Connections (this batch)", len(last))
        m2.metric("🔴 Alerts", int(is_alert.sum()))
        m3.metric("🟠 Suspicious (unknown)", int(is_susp.sum()))
        m4.metric("Session total scanned", len(log))

        left, right = st.columns(2)
        with left:
            st.subheader("Verdicts in this batch")
            st.bar_chart(last["prediction"].value_counts())
        with right:
            st.subheader("Alert trend (session)")
            trend = log.assign(batch=np.arange(len(log)) // max(batch, 1))
            trend["alert"] = ~trend["verdict"].eq("Normal")
            st.line_chart(trend.groupby("batch")["alert"].sum())

        st.subheader("Flagged connections (highest risk first)")
        flagged = last[is_alert | is_susp].sort_values("attack_probability", ascending=False)
        show = ["verdict", "attack_probability", "anomaly_score", "protocol_type", "service",
                "flag", "src_bytes", "dst_bytes", "count", "serror_rate", "true_label"]
        st.dataframe(flagged[show].head(200), use_container_width=True, hide_index=True)

        truth_attack = last["true_label"] != "normal"
        caught = (is_alert | is_susp)
        if truth_attack.any():
            st.success(f"Ground-truth check: caught {int((caught & truth_attack).sum())} of "
                       f"{int(truth_attack.sum())} simulated attacks; "
                       f"{int((caught & ~truth_attack).sum())} false alarms.")

# --------------------------------------------------------------------------- #
with tab_upload:
    st.write("Upload a CSV containing the 41 NSL-KDD feature columns "
             "(try `data/sample_traffic.csv`).")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        df = pd.read_csv(up)
        missing = [c for c in COLUMNS if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            res = detector.predict_frame(df[COLUMNS])
            out = pd.concat([df.reset_index(drop=True), res.reset_index(drop=True)], axis=1)
            a, b, c = st.columns(3)
            a.metric("Rows scanned", len(out))
            b.metric("Alerts", int(out["verdict"].str.startswith("ALERT").sum()))
            c.metric("Suspicious", int(out["verdict"].str.startswith("SUSPICIOUS").sum()))
            st.bar_chart(out["verdict"].value_counts())
            st.dataframe(out.sort_values("attack_probability", ascending=False).head(500),
                         use_container_width=True, hide_index=True)
            st.download_button("⬇ Download predictions", out.to_csv(index=False).encode(),
                               "predictions.csv", "text/csv")

# --------------------------------------------------------------------------- #
with tab_report:
    mfile = REPORTS / "metrics.json"
    if not mfile.exists():
        st.warning("Run `python -m src.train` to generate the report.")
    else:
        m = json.loads(mfile.read_text())
        st.caption(f"Data: {m['data_source']} · train {m['n_train']:,} / test {m['n_test']:,}")
        a, b, c, d = st.columns(4)
        a.metric("Best model", m["best_model"])
        b.metric("Accuracy", f"{m['accuracy']:.2%}")
        c.metric("Macro-F1", f"{m['macro_f1']:.3f}")
        d.metric("Binary ROC-AUC", f"{m['binary_roc_auc']:.3f}")

        e, f = st.columns(2)
        sb, hb = m["supervised_binary"], m["hybrid_binary"]
        e.metric("Hybrid detection rate", f"{hb['detection_rate']:.2%}",
                 f"{hb['detection_rate'] - sb['detection_rate']:+.2%} vs supervised")
        f.metric("Hybrid false-alarm rate", f"{hb['false_alarm_rate']:.2%}",
                 f"{hb['false_alarm_rate'] - sb['false_alarm_rate']:+.2%} vs supervised",
                 delta_color="inverse")

        st.subheader("Model comparison")
        st.dataframe(pd.read_csv(REPORTS / "model_comparison.csv"), hide_index=True,
                     use_container_width=True)
        p1, p2 = st.columns(2)
        if (REPORTS / "confusion_matrix.png").exists():
            p1.image(str(REPORTS / "confusion_matrix.png"))
        if (REPORTS / "feature_importance.png").exists():
            p2.image(str(REPORTS / "feature_importance.png"))
        if (REPORTS / "zero_day.png").exists():
            st.subheader("Zero-day experiment")
            st.caption("One attack family is removed from training; the hybrid's anomaly layer "
                       "recovers much of what a purely supervised model misses.")
            z1, z2 = st.columns([1, 1])
            z1.image(str(REPORTS / "zero_day.png"))
            z2.dataframe(pd.read_csv(REPORTS / "zero_day.csv"), hide_index=True)
