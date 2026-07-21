# 🛠️ Predictive Maintenance — Failure Risk Predictor

Predicts whether an industrial machine will fail in the **next 24 hours** using sensor telemetry, error logs, and maintenance history. Trained on a full year of hourly data from 100 machines (Azure AI predictive-maintenance sample dataset). Random Forest classifier, ROC AUC ≈ 0.99 on a genuinely held-out future time period.

**Live demo:** https://machine-failure-predictor-12345.streamlit.app _(update with your actual URL)_

---

## What's in this repo

| File | Purpose |
|---|---|
| `app.py` | The Streamlit web app |
| `model.pkl` | Trained Random Forest model + feature names (loaded by `app.py`) |
| `demo_data.csv` | Compact sample of real machine snapshots the app lets you browse |
| `requirements.txt` | Python packages needed to run the app |
| `runtime.txt` | Pins the Python version (3.11) so dependencies install from prebuilt wheels |
| `predictive_maintenance_notebook.ipynb` | Full EDA + modeling walkthrough — the portfolio piece |
| `.gitignore` | Excludes local/build artifacts from version control |

---

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. Confirm both tabs work ("Explore a real machine" and "Build your own scenario") before deploying.

---

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub as a **public** repository (required for the free tier).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app** → **Deploy a public app from GitHub**.
4. Select this repo, branch `main`, and set **Main file path** to `app.py`.
5. Click **Deploy**. First build takes 2–5 minutes.

Any time you push new commits, the deployed app auto-updates.

### Known deployment gotchas

- **Python version:** Streamlit Cloud may default to a very new Python release that doesn't have prebuilt wheels for some pinned packages (e.g. `pillow`), causing a build failure trying to compile from source. Fix: keep `runtime.txt` in the repo root with `3.11` inside it, and do a full **Reboot app** (not just rerun) after adding it.
- **File naming:** `app.py` expects the demo data file to be named exactly `demo_data.csv` at the repo root — not `.xls`, not a different name. A mismatch here throws `FileNotFoundError` at startup.
- **Model/library version match:** `model.pkl` was trained with a specific `scikit-learn` version. If `requirements.txt` installs a different one, you may see a version-mismatch warning at load time — worth spot-checking predictions still look sane after any dependency update.

---

## How to talk about this project in interviews

- **Problem framing:** predictive maintenance is a real, high-value ML use case (unplanned downtime is expensive) — shows translating a business problem into a supervised learning task.
- **Time-series discipline:** the train/test split is time-based, not random, preventing leakage — a common mistake interviewers probe for.
- **Class imbalance handling:** only ~2% positive rate; handled with class weighting and training-set undersampling, not naive accuracy.
- **End-to-end ownership:** raw data → feature engineering → model → deployed, publicly accessible product, not just a notebook.
