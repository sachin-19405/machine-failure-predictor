import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Predictive Maintenance AI", page_icon="🛠️", layout="wide")

# ---------- Load model + data (cached so it only loads once) ----------
@st.cache_resource
def load_model():
    bundle = joblib.load("model.pkl")
    return bundle["model"], bundle["feature_names"]

@st.cache_data
def load_demo_data(feature_names):
    df = pd.read_csv("demo_data.csv", parse_dates=["datetime"])
    dummies = pd.get_dummies(df["model"], prefix="model")
    for col in feature_names:
        if col.startswith("model_") and col not in dummies.columns:
            dummies[col] = 0
    df = pd.concat([df, dummies], axis=1)
    return df

model, feature_names = load_model()
demo = load_demo_data(feature_names)

st.title("🛠️ Predictive Maintenance — Failure Risk Predictor")
st.caption(
    "Predicts the probability that an industrial machine will experience a "
    "component failure in the **next 24 hours**, based on sensor telemetry, "
    "recent error codes, and maintenance history. Trained on a year of hourly "
    "data from 100 machines (Random Forest, ROC AUC ≈ 0.99)."
)

tab1, tab2 = st.tabs(["🔍 Explore a real machine", "🎛️ Build your own scenario"])

# ---------------------------------------------------------------
# TAB 1 — browse real historical snapshots
# ---------------------------------------------------------------
with tab1:
    st.subheader("Pick a machine and a date")
    col1, col2 = st.columns(2)
    machine_id = col1.selectbox("Machine ID", sorted(demo["machineID"].unique()))

    m_df = demo[demo["machineID"] == machine_id].sort_values("datetime")
    dates = m_df["datetime"].dt.date.tolist()
    picked_date = col2.select_slider("Date", options=dates, value=dates[len(dates) // 2])

    row = m_df[m_df["datetime"].dt.date == picked_date].iloc[0]

    X_row = pd.DataFrame([row[feature_names]])
    prob = model.predict_proba(X_row)[0, 1]

    st.divider()
    c1, c2, c3 = st.columns([1, 1, 1.4])
    c1.metric("Predicted failure risk (next 24h)", f"{prob*100:.1f}%")
    actual = "⚠️ Failure occurred" if row["fail_next_24h"] == 1 else "✅ No failure"
    c2.metric("Actual outcome (ground truth)", actual)
    c3.metric("Machine profile", f"{row['model']}, age {int(row['age'])} yrs")

    st.progress(min(float(prob), 1.0))

    with st.expander("Show raw sensor readings for this snapshot"):
        st.dataframe(row[["volt", "rotate", "pressure", "vibration"] +
                         [c for c in feature_names if "days_since_maint" in c]].to_frame("value"))

# ---------------------------------------------------------------
# TAB 2 — manual what-if scenario
# ---------------------------------------------------------------
with tab2:
    st.subheader("Set sensor values and see the predicted risk update live")
    defaults = demo[feature_names].median(numeric_only=True)

    colA, colB = st.columns(2)
    with colA:
        volt = st.slider("Voltage (mean, 24h)", 150.0, 250.0, float(defaults["volt_mean_24h"]))
        rotate = st.slider("Rotation speed (mean, 24h)", 300.0, 550.0, float(defaults["rotate_mean_24h"]))
        pressure = st.slider("Pressure (mean, 24h)", 70.0, 200.0, float(defaults["pressure_mean_24h"]))
        vibration = st.slider("Vibration (mean, 24h)", 30.0, 80.0, float(defaults["vibration_mean_24h"]))
    with colB:
        errors_24h = st.slider("Error codes logged in last 24h", 0, 10, 0)
        days_since_maint = st.slider("Days since last maintenance (any component)", 0, 60, 10)
        age = st.slider("Machine age (years)", 0, 20, 8)
        model_choice = st.selectbox("Machine model", ["model1", "model2", "model3", "model4"])

    scenario = defaults.copy()
    scenario["volt_mean_24h"] = volt
    scenario["volt_mean_3h"] = volt
    scenario["rotate_mean_24h"] = rotate
    scenario["rotate_mean_3h"] = rotate
    scenario["pressure_mean_24h"] = pressure
    scenario["pressure_mean_3h"] = pressure
    scenario["vibration_mean_24h"] = vibration
    scenario["vibration_mean_3h"] = vibration
    scenario["age"] = age
    for c in feature_names:
        if "days_since_maint" in c:
            scenario[c] = days_since_maint
        if c.startswith("error") and c.endswith("_count_24h"):
            scenario[c] = 0
    if errors_24h > 0 and "error1_flag_count_24h" in feature_names:
        scenario["error1_flag_count_24h"] = errors_24h

    for c in feature_names:
        if c.startswith("model_"):
            scenario[c] = 0
    model_col = f"model_{model_choice}"
    if model_col in feature_names:
        scenario[model_col] = 1

    X_scenario = pd.DataFrame([scenario[feature_names]])
    prob_scenario = model.predict_proba(X_scenario)[0, 1]

    st.divider()
    risk_label = "🔴 High risk" if prob_scenario > 0.5 else ("🟡 Moderate risk" if prob_scenario > 0.15 else "🟢 Low risk")
    st.metric("Predicted failure risk (next 24h)", f"{prob_scenario*100:.1f}%", risk_label)
    st.progress(min(float(prob_scenario), 1.0))
    st.caption(
        "Try raising vibration/pressure and error counts while pushing maintenance "
        "recency up — you should see the risk climb, matching real degradation patterns."
    )

st.divider()
st.caption(
    "Data: Azure AI Predictive Maintenance sample dataset (100 machines, hourly telemetry, "
    "1 year). Model: Random Forest classifier. Built as a portfolio project — not for "
    "real industrial use."
)
