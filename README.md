# 🛡️ Intelligent Network Intrusion Detection (Hybrid ML NIDS)

A machine-learning intrusion detection system that classifies network connections as
**normal / DoS / Probe / R2L / U2R** *and* flags **never-seen (zero-day) behaviour**
using an anomaly-detection layer, with a live dashboard for demos.

## Why this is more than "train a classifier"

| Problem with classic signature/ML IDS | What this project does |
|---|---|
| Only detects attacks it has seen | **Hybrid design**: Isolation Forest trained on *normal traffic only* raises "SUSPICIOUS: possible unknown attack" |
| Class imbalance (R2L/U2R are rare) | Balanced class weights + macro-F1 model selection |
| Raw features hide attack patterns | Engineered features (bytes ratio, total bytes, error-rate sum, host pressure) + log scaling |
| "Black box" alerts | Permutation feature importance, confidence + anomaly score for every alert |
| Accuracy hides false alarms | Reports detection rate **and** false-alarm rate |
| Models are never shown working | Streamlit SOC-style dashboard with attack scenario simulator |

## Architecture

```
 connection record (41 NSL-KDD features)
            │
   feature engineering + preprocessing (one-hot, log, scaling)
            │
   ┌────────┴──────────────────────────┐
   │ Layer 1: supervised classifier    │ → normal / dos / probe / r2l / u2r + confidence
   │ (best of LR, RF, HistGB, MLP)     │
   ├───────────────────────────────────┤
   │ Layer 2: Isolation Forest         │ → anomaly score (trained on normal only)
   └────────┬──────────────────────────┘
            │
   verdict: ALERT: <TYPE>  |  SUSPICIOUS: possible unknown attack  |  Normal
```

## Quick start

```bash
pip install -r requirements.txt
python -m src.train              # trains, compares models, writes models/ + reports/
streamlit run app.py             # launches the dashboard
python -m src.predict data/sample_traffic.csv    # CLI scoring
```

Training takes about a minute on a laptop CPU (`--skip-zero-day` makes it faster).

### Using the real NSL-KDD dataset (recommended for final submission)

The project runs out-of-the-box on a built-in **synthetic NSL-KDD-style generator**
(same 41 features, same 5 classes) so it works offline. For your final numbers, download
NSL-KDD (`KDDTrain+.txt`) and run:

```bash
python -m src.train --data data/KDDTrain+.txt
```

The loader maps the 39 specific attack names (neptune, smurf, nmap, guess_passwd,
buffer_overflow, …) to the 4 attack families automatically. The dashboard simulator
keeps using synthetic traffic; the CSV upload tab works with real rows.

## Results (synthetic data, 30k connections, 25% held-out test)

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Random Forest (selected) | 95.9% | 0.905 |
| Hist. Gradient Boosting | 95.7% | 0.902 |
| MLP | 95.3% | 0.893 |
| Logistic Regression | 95.0% | 0.877 |

Final hybrid detector: **91.5% attack detection at 2.8% false alarms** (binary view), ROC-AUC 0.955.

**Zero-day experiment** (one attack family removed from training, then tested):

| Unseen family | Supervised only | Hybrid |
|---|---|---|
| DoS | 87.2% | 90.2% |
| Probe | 5.5% | **95.2%** |
| R2L | 24.9% | 37.6% |
| U2R | 19.3% | 42.7% |

The anomaly layer costs about 3 points of false alarms and recovers most of the
reconnaissance traffic the classifier could not name. R2L/U2R remain hard, which matches
the intrusion-detection literature: those attacks look almost like legitimate sessions.

> **Be upfront in your presentation:** these figures come from synthetic data that
> imitates NSL-KDD. Re-run on the real dataset and quote those numbers. Expect lower
> R2L/U2R recall there, which is a good talking point about the limits of
> connection-level features.

## Project structure

```
app.py                  Streamlit dashboard (live monitor, CSV analysis, model report)
src/data.py             NSL-KDD schema, loader, synthetic traffic generator
src/detector.py         Feature engineering, model zoo, HybridDetector
src/train.py            Training, comparison, evaluation, zero-day experiment, plots
src/predict.py          CLI batch scoring
models/                 Saved detector (created by training)
reports/                metrics.json, confusion matrix, feature importance, zero-day plots
data/sample_traffic.csv Labelled demo file for the upload tab
```

## Demo script (2–3 minutes)

1. **Problem**: signature IDS miss new attacks; plain ML misses what it wasn't trained on.
2. **Dashboard → Live monitor**: run *Mixed traffic*, then *DoS flood* and *Port scan* and show alerts with risk scores.
3. **Model report tab**: model comparison, confusion matrix, top features (e.g. `serror_rate`, `count`, `same_srv_rate`).
4. **Zero-day chart**: "We hid port scans from training; the classifier caught 5%, our hybrid caught 95%."
5. **Limits and next steps**: real-time packet capture (CICFlowMeter → this model), CICIDS2017, SHAP explanations, drift monitoring and periodic retraining.

## Ideas to extend

- Swap in **CICIDS2017 / UNSW-NB15** (map columns in `src/data.py`).
- Add a deep autoencoder as a second anomaly detector.
- Feed live flows from Zeek/CICFlowMeter into `HybridDetector.predict_frame`.
- Add SHAP explanations per alert.
