
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

st.set_page_config(page_title="Digital Twin — Health Risk Predictor", layout="wide")

# ---- Load model ----
MODEL_PATHS = [
    "saved_models/final_model.pkl",
    "saved_models/best_rf_pipeline.pkl",
    "saved_models/best_xgb_pipeline.pkl",
    "best_model.pkl"
]

def load_model():
    for p in MODEL_PATHS:
        if os.path.exists(p):
            try:
                m = joblib.load(p)
                return m, p
            except Exception as e:
                st.warning(f"Found model at {p} but failed to load: {e}")
    return None, None

model, model_path = load_model()

st.title("🩺 Digital Twin — Health Risk Predictor (Enhanced)")
st.markdown("Enter vitals manually, upload a CSV for batch predictions, view feature importances, and download results.")

col_main, col_side = st.columns([3,1])

with col_side:
    st.subheader("Model status")
    if model is None:
        st.error("No trained model found. Run the notebook cells to save a model in 'saved_models/'.")
    else:
        st.success(f"Loaded model from: {model_path}")
        st.write(type(model))

# ---- Sidebar manual inputs ----
st.sidebar.header("Manual Patient Vitals")
hr = st.sidebar.slider("Heart Rate (bpm)", 30, 200, 80)
rr = st.sidebar.slider("Respiratory Rate (breaths/min)", 8, 40, 16)
temp = st.sidebar.slider("Body Temperature (°C)", 34.0, 42.0, 36.6, step=0.1)
spo2 = st.sidebar.slider("Oxygen Saturation (%)", 60, 100, 97)
sys_bp = st.sidebar.slider("Systolic BP (mmHg)", 70, 220, 120)
dia_bp = st.sidebar.slider("Diastolic BP (mmHg)", 40, 140, 80)
age = st.sidebar.number_input("Age (years)", 0, 120, 35)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
weight = st.sidebar.number_input("Weight (kg)", 20.0, 200.0, 70.0)
height = st.sidebar.number_input("Height (m)", 1.0, 2.5, 1.7)

derived_hrv = max(0.1, 100.0 / max(1, hr))
pulse_pressure = sys_bp - dia_bp
bmi = round(weight / (height**2), 2)
map_val = (2*dia_bp + sys_bp) / 3.0

input_data = {
    "Heart Rate": hr,
    "Respiratory Rate": rr,
    "Body Temperature": temp,
    "Oxygen Saturation": spo2,
    "Systolic Blood Pressure": sys_bp,
    "Diastolic Blood Pressure": dia_bp,
    "Age": age,
    "Gender": gender,
    "Weight (kg)": weight,
    "Height (m)": height,
    "Derived_HRV": derived_hrv,
    "Derived_Pulse_Pressure": pulse_pressure,
    "Derived_BMI": bmi,
    "Derived_MAP": map_val
}
input_df = pd.DataFrame([input_data])

with col_main:
    st.subheader("Input")
    st.dataframe(input_df)

    # Predict single
    st.markdown("### Prediction (single)")
    if model is not None:
        try:
            pred = model.predict(input_df)[0]
            # guess mapping (update if your mapping differs)
            if isinstance(pred, (np.integer, int)):
                mapping = {0: "High Risk", 1: "Low Risk"}
                st.metric("Predicted Risk", mapping.get(int(pred), str(pred)))
            else:
                st.metric("Predicted Risk", str(pred))
            try:
                probs = model.predict_proba(input_df)[0]
                if len(probs) == 2:
                    st.write(f"High Risk probability: {probs[0]:.3f} — Low Risk: {probs[1]:.3f}")
                else:
                    st.write("Class probabilities:", probs)
            except Exception:
                st.info("Probabilities not available for this model.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.info("No model loaded.")

    st.markdown('---')

    # ---- Batch predictions ----
    st.subheader("Batch Predictions (CSV upload)")
    st.write("Upload a CSV with the same column names as the dataset. Required columns (example):")
    example_cols = list(input_df.columns)
    st.write(example_cols)

    uploaded = st.file_uploader("Upload CSV for batch prediction", type=["csv"])
    if uploaded is not None:
        try:
            df_upload = pd.read_csv(uploaded)
            st.write("Preview of uploaded data:")
            st.dataframe(df_upload.head())

            # Check for missing expected columns
            missing = [c for c in example_cols if c not in df_upload.columns]
            if len(missing) > 0:
                st.warning(f"Uploaded CSV is missing expected columns: {missing}. Attempting to continue but results may be incorrect.")
            if model is not None:
                # Predict (pipeline should handle preprocessing)
                preds = model.predict(df_upload)
                out = df_upload.copy()
                out["Predicted_Risk"] = preds
                try:
                    probs = model.predict_proba(df_upload)
                    if probs.shape[1] == 2:
                        out["Prob_HighRisk"] = probs[:,0]
                        out["Prob_LowRisk"] = probs[:,1]
                except Exception:
                    pass

                st.write("Sample predictions:")
                st.dataframe(out.head())

                # Download results
                csv_bytes = out.to_csv(index=False).encode("utf-8")
                st.download_button("Download predictions (CSV)", data=csv_bytes, file_name="predictions.csv")
            else:
                st.info("Load a model to run batch predictions.")
        except Exception as e:
            st.error(f"Failed to read or predict on uploaded CSV: {e}")

    st.markdown('---')

    # ---- Feature importances ----
    st.subheader("Feature Importances")
    fi_displayed = False
    if model is not None:
        try:
            clf = None
            if hasattr(model, "named_steps"):
                if "classifier" in model.named_steps:
                    clf = model.named_steps["classifier"]
                elif "clf" in model.named_steps:
                    clf = model.named_steps["clf"]
            elif hasattr(model, "feature_importances_"):
                clf = model

            if clf is not None and hasattr(clf, "feature_importances_"):
                importances = clf.feature_importances_
                feat_names = example_cols.copy()  # hope they align; this matches training notebook
                n = min(len(importances), len(feat_names))
                fi = pd.Series(importances[:n], index=feat_names[:n]).sort_values(ascending=False)
                st.bar_chart(fi.head(10))
                st.write("Top feature importances:")
                st.dataframe(fi.head(15).reset_index().rename(columns={"index": "feature", 0: "importance"}))
                fi_displayed = True
            else:
                st.info("Feature importances not available for this model (non-tree model).")
        except Exception as e:
            st.error(f"Could not compute feature importances: {e}")
    if not fi_displayed:
        st.write("No feature importance plot available.")

    st.markdown('---')

    # ---- Simple trajectory simulator (batch) ----
    st.subheader("Trajectory Simulator (batch preview)")
    steps = st.number_input("Steps", min_value=2, max_value=50, value=8)
    hr_delta = st.number_input("HR change per step", min_value=-10, max_value=20, value=1)
    # create a small batch from input_df across steps
    traj = []
    hr_now = hr
    for i in range(steps):
        r = input_data.copy()
        r["Heart Rate"] = hr_now
        traj.append(r)
        hr_now += hr_delta
    traj_df = pd.DataFrame(traj)
    st.write("Trajectory preview:")
    st.dataframe(traj_df.head())

    if model is not None:
        try:
            preds_traj = model.predict(traj_df)
            mapping = {0: "High Risk", 1: "Low Risk"}
            labels = [mapping.get(int(p), str(p)) for p in preds_traj]
            st.line_chart(pd.DataFrame({
                "Heart Rate": traj_df["Heart Rate"],
                "Risk (encoded)": pd.Series(labels).astype('category').cat.codes
            }))
            st.table(pd.DataFrame({"Step": range(1, len(labels)+1), "Heart Rate": traj_df["Heart Rate"], "Predicted Risk": labels}))
        except Exception as e:
            st.error(f"Trajectory prediction failed: {e}")

with col_side:
    st.markdown("### Help & Notes")
    st.write("- Ensure uploaded CSV uses the same column names as displayed above.")
    st.write("- The model mapping of numeric labels to text (High/Low) assumes 0=High Risk, 1=Low Risk. Adjust mapping if your encoding differs.")
    st.write("- SHAP explanations are not shown in this simplified app. Install `shap` and extend the app if needed.")
    st.write("- Validate with clinical experts before real-world use.")

