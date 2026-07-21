# 🛠️ Predictive Maintenance — Failure Risk Predictor

Predicts whether an industrial machine will fail in the **next 24 hours** using sensor telemetry, error logs, and maintenance history. Trained on a full year of hourly data from 100 machines (Azure AI predictive-maintenance sample dataset). Random Forest classifier, ROC AUC ≈ 0.99 on a genuinely held-out future time period.

**Live demo:** : https://machine-failure-predictor-12.streamlit.app

---

## What's in this folder

| File | Purpose |
|---|---|
| `PdM_*.csv` | Raw source data (telemetry, errors, failures, maintenance, machines) |
| `build_features.py` | Turns raw data into an hourly feature table with rolling stats + labels |
| `train_model.py` | Trains the Random Forest and saves `model.pkl` |
| `predictive_maintenance_notebook.ipynb` | Full EDA + modeling walkthrough with charts and explanations — this is your portfolio piece |
| `app.py` | The Streamlit web app that gets deployed |
| `demo_data.csv` | Compact sample of real machine snapshots the app lets you browse |
| `model.pkl` | The trained model (loaded by `app.py`) |
| `requirements.txt` | Python packages needed to run the app |

---

## Part 1 — Run it locally first

```bash
pip install -r requirements.txt
streamlit run app.py
```

This opens the app at `http://localhost:8501`. Confirm both tabs work ("Explore a real machine" and "Build your own scenario") before deploying.

If you ever want to rebuild the model from scratch:
```bash
python build_features.py   # rebuilds features.csv from raw CSVs (~2 min)
python train_model.py      # retrains and overwrites model.pkl
```

---

## Part 2 — Put the project on GitHub

1. Create a **free GitHub account** at github.com if you don't have one.
2. Create a **new repository** (e.g. `predictive-maintenance-app`). Make it **public** — Streamlit Community Cloud's free tier deploys from public repos.
3. Upload these files to the repo (drag-and-drop on GitHub's web UI works fine, or use git):
   - `app.py`
   - `model.pkl`
   - `demo_data.csv`
   - `requirements.txt`
   - `predictive_maintenance_notebook.ipynb` (optional, but great for recruiters to see)
   - `README.md`

   **Do NOT upload the raw `PdM_*.csv` files or `features.csv`** — they're large and the app doesn't need them at runtime (only `demo_data.csv` and `model.pkl` are used by `app.py`).

   Using git from your terminal instead:
   ```bash
   git init
   git add app.py model.pkl demo_data.csv requirements.txt README.md predictive_maintenance_notebook.ipynb
   git commit -m "Predictive maintenance app"
   git branch -M main
   git remote add origin https://github.com/<your-username>/predictive-maintenance-app.git
   git push -u origin main
   ```

---

## Part 3 — Deploy for free on Streamlit Community Cloud

This gives you a public URL like `https://predictive-maintenance-app.streamlit.app` that anyone can open — no server, no credit card.

1. Go to **share.streamlit.io** and sign in with your GitHub account.
2. Click **"Create app"** → **"Deploy a public app from GitHub"**.
3. Select your repository, branch (`main`), and set **Main file path** to `app.py`.
4. Click **Deploy**. It installs `requirements.txt` and launches the app — first deploy takes 2–5 minutes.
5. You'll get a shareable public URL. Put it in this README and on your resume/LinkedIn/portfolio site.

**Whenever you push new commits to GitHub, the deployed app auto-updates** — no redeploy step needed.

### If deployment fails
- Check the app logs in the Streamlit Cloud dashboard (usually a missing file or package version mismatch).
- Most common cause: `model.pkl` wasn't uploaded, or was uploaded via Git LFS by accident (GitHub's web upload has a 25MB per-file limit for drag-and-drop; `model.pkl` here is ~5MB so it's fine, but if you rebuild a bigger model, use `git` directly instead of the web uploader).

---

## Part 4 — Other free deployment options (if you want alternatives)

- **Hugging Face Spaces** (also free, supports Streamlit/Gradio) — good alternative if you want it alongside other ML demos.
- **Render.com free web service** — more general-purpose, works for Flask/FastAPI apps too if you later build an API instead of a Streamlit UI.
- Streamlit Community Cloud is recommended here because it's purpose-built for exactly this kind of app and requires zero server config.

---

## How to talk about this project in interviews

- **Problem framing:** predictive maintenance is a real, high-value ML use case (unplanned downtime is expensive) — this shows you can translate a business problem into a supervised learning task.
- **Time-series discipline:** the train/test split is time-based, not random, which prevents leakage — mention this explicitly, it's a common mistake interviewers probe for.
- **Class imbalance handling:** only ~2% positive rate; handled with class weighting + training-set undersampling, not naive accuracy.
- **End-to-end ownership:** raw data → feature engineering → model → deployed, publicly accessible product, not just a notebook.
