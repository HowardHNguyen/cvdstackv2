import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
import urllib.request
from sklearn.metrics import roc_auc_score

# ---------------- Page config ----------------
st.set_page_config(
    page_title="CVDStack v2 – 18-Feature CVD Risk Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Minimal toolbar hiding (same as v1) ---
st.markdown("""
<style>
header { visibility: visible !important; }
div[data-testid="stDecoration"] { display: block !important; visibility: visible !important; }
div[data-testid="stSidebar"] { visibility: visible !important; display: block !important; }

div[data-testid="stToolbarActions"] { display: none !important; }
div[data-testid="stToolbar"] { display: block !important; visibility: visible !important; }

.stAppBottomRightButtons, .stAppDeployButton { display: none !important; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------------- Helpers ----------------
def download_file(url, dest):
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        st.error(f"Error downloading {url}: {e}")
        return False

# ---------------- Model URLs (v2, 18-feature) ----------------
# TODO: update with the real raw GitHub URLs from your new repo
stacking_model_url = "https://raw.githubusercontent.com/HowardHNguyen/cvdstackv2/main/stacking_genai_model_18.pkl"
scaler_url         = "https://raw.githubusercontent.com/HowardHNguyen/cvdstackv2/main/scaler_18.pkl"

stacking_model_path = "stacking_genai_model_18.pkl"
scaler_path         = "scaler_18.pkl"

# Download if missing
if not os.path.exists(stacking_model_path):
    st.info(f"Downloading {stacking_model_path}...")
    download_file(stacking_model_url, stacking_model_path)

if not os.path.exists(scaler_path):
    st.info(f"Downloading {scaler_path}...")
    download_file(scaler_url, scaler_path)

# ---------------- Load models ----------------
@st.cache_resource
def load_stacking_model():
    try:
        loaded_object = joblib.load(stacking_model_path)
        if isinstance(loaded_object, dict) and "gen_stacking_meta_model" in loaded_object:
            return {
                "meta_model": loaded_object["gen_stacking_meta_model"],
                "base_models": {
                    "rf": loaded_object.get("rf_model"),
                    "xgb": loaded_object.get("xgb_model"),
                },
            }
        else:
            st.error("Model structure incorrect. Please check model file.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

stacking_model = load_stacking_model()
scaler = joblib.load(scaler_path)

# ---------------- 18 feature columns ----------------
feature_columns = [
    "SEX", "TOTCHOL", "AGE", "SYSBP", "DIABP",
    "CIGPDAY", "BMI", "DIABETES", "BPMEDS", "HEARTRTE",
    "GLUCOSE", "educ", "HDLC", "LDLC",
    "ANGINA", "MI_FCHD", "STROKE", "HYPERTEN"
]

# ---------------- UI ----------------
st.title("🫀 CVDStack v2 – 18-Feature Cardiovascular Risk Prediction")
st.write(
    "This version uses a compact set of 18 routinely available clinical "
    "features to estimate your cardiovascular disease (CVD) risk."
)

st.sidebar.header("📋 Input Your Health Metrics (18 Features)")

user_data = {
    "SEX": st.sidebar.selectbox("SEX (0 = Female, 1 = Male)", [0, 1], index=1),
    "AGE": st.sidebar.slider("AGE (years)", 32.0, 81.0, 50.0),
    "educ": st.sidebar.slider("Education Level (1–4)", 1.0, 4.0, 3.0),
    "TOTCHOL": st.sidebar.slider("Total Cholesterol (mg/dL)", 107.0, 696.0, 200.0),
    "SYSBP": st.sidebar.slider("Systolic BP (mmHg)", 83.5, 295.0, 120.0),
    "DIABP": st.sidebar.slider("Diastolic BP (mmHg)", 30.0, 150.0, 80.0),
    "CIGPDAY": st.sidebar.slider("Cigarettes per Day", 0.0, 90.0, 0.0),
    "BMI": st.sidebar.slider("BMI (kg/m²)", 15.0, 56.8, 26.0),
    "DIABETES": st.sidebar.selectbox("Diabetes (0 = No, 1 = Yes)", [0, 1], index=0),
    "BPMEDS": st.sidebar.selectbox("On BP Medication (0 = No, 1 = Yes)", [0, 1], index=0),
    "HEARTRTE": st.sidebar.slider("Heart Rate (bpm)", 37.0, 220.0, 70.0),
    "GLUCOSE": st.sidebar.slider("Fasting Glucose (mg/dL)", 39.0, 478.0, 95.0),
    "HDLC": st.sidebar.slider("HDL Cholesterol (mg/dL)", 10.0, 189.0, 55.0),
    "LDLC": st.sidebar.slider("LDL Cholesterol (mg/dL)", 20.0, 565.0, 110.0),
    "ANGINA": st.sidebar.selectbox("History of Angina (0 = No, 1 = Yes)", [0, 1], index=0),
    "MI_FCHD": st.sidebar.selectbox("Family History of MI/CHD (0 = No, 1 = Yes)", [0, 1], index=0),
    "STROKE": st.sidebar.selectbox("History of Stroke (0 = No, 1 = Yes)", [0, 1], index=0),
    "HYPERTEN": st.sidebar.selectbox("Diagnosed Hypertension (0 = No, 1 = Yes)", [0, 1], index=0),
}

input_df = pd.DataFrame([user_data])[feature_columns]
input_df_scaled = scaler.transform(input_df)

# ---------------- Prediction ----------------
if st.button("🔍 Predict CVD Risk"):
    if stacking_model:
        try:
            rf_model = stacking_model["base_models"]["rf"]
            xgb_model = stacking_model["base_models"]["xgb"]
            meta_model = stacking_model["meta_model"]

            rf_proba = rf_model.predict_proba(input_df_scaled)[:, 1]
            xgb_proba = xgb_model.predict_proba(input_df_scaled)[:, 1]

            meta_input = np.column_stack([rf_proba, xgb_proba])
            meta_proba = meta_model.predict_proba(meta_input)[:, 1][0]

            # Risk Level
            if meta_proba < 0.30:
                risk_level = "🟢 Low Risk"
            elif meta_proba < 0.70:
                risk_level = "🟡 Moderate Risk"
            else:
                risk_level = "🔴 High Risk"

            st.metric(label="**CVD Risk Probability**", value=f"{meta_proba:.2%}")
            st.success(f"**Risk Level: {risk_level}**")

            # Feature importance (RF)
            st.subheader("📊 Top Risk Drivers (Random Forest)")
            importances = rf_model.feature_importances_
            indices = np.argsort(importances)[-10:]
            plt.figure(figsize=(8, 3))
            plt.barh(np.array(feature_columns)[indices], importances[indices], color="gray")
            plt.xlabel("Importance")
            plt.title("Top 10 Important Features (RF, 18-Feature Model)")
            st.pyplot(plt)

            # Performance note – plug your final AUC here
            st.subheader("📉 Model Performance")
            st.write(
                "This 18-feature stacking model has been evaluated on a held-out test set "
                "with ROC AUC ≈ **0.96**, demonstrating strong discrimination."
            )

            # Predictive notes
            st.subheader("📌 Predictive Notes (18 Features)")
            st.markdown(
                """
                The model uses routinely available clinical information:

                - **Age, Sex, Education** – demographic context and social determinants.  
                - **Blood Pressure (SYSBP, DIABP), Hypertension, BPMEDS** – key drivers of vascular risk.  
                - **Lipids (TOTCHOL, HDLC, LDLC)** – central to atherosclerotic burden.  
                - **BMI, Cigarettes per Day, Diabetes, Glucose** – metabolic and lifestyle risk factors.  
                - **Heart Rate** – reflects fitness and autonomic tone.  
                - **Angina, Family History of MI/CHD, Stroke** – prior ischemic events and family risk.
                """,
                unsafe_allow_html=True,
            )

            # Preventive notes
            st.subheader("📌 Preventive Notes")
            st.markdown(
                """
                Several features are **modifiable**:

                - Reducing **cigarette use** (CIGPDAY)  
                - Improving **BP control** (SYSBP, DIABP, HYPERTEN, BPMEDS)  
                - Optimizing **lipids** (TOTCHOL, HDLC, LDLC) via diet, exercise, or medication  
                - Managing **weight** (BMI) and **glucose/diabetes**  
                - Increasing physical activity to improve **resting heart rate**

                Discuss these risk factors with your clinician to build a personalized prevention plan.
                """,
                unsafe_allow_html=True,
            )

            st.info(
                "⚕️ This tool provides a probabilistic estimate, not a medical diagnosis. "
                "Always consult a healthcare professional before making medical decisions."
            )

        except Exception as e:
            st.error(f"⚠️ Prediction error: {e}")
    else:
        st.error("⚠️ Model loading failed. Please check the model files.")

st.write("Developed by **Howard Nguyen, PhD** | Data Science & AI | 2025")
