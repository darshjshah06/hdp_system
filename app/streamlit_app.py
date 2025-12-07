import os
import sys

# Allow imports from project root (hdp_system/)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import pandas as pd
import streamlit as st
from sqlalchemy import text
from src.hdp_system.db import get_engine


# ================================
# Page Config + CSS
# ================================
st.set_page_config(page_title="Heart Disease Prediction System ❤️", layout="wide")

# Custom CSS (UNCHANGED UI)
st.markdown(
    """
    <style>
        /* Title */
        .main-title {
            text-align: center;
            font-size: 3.5rem;
            color: #ff4d6d;
            font-weight: 900;
            margin-bottom: 0.3rem;
        }

        .subtitle-text {
            text-align: center;
            font-size: 1.2rem;
            color: #444444;
            margin-top: -10px;
            margin-bottom: 1.2rem;
        }

        /* Risk glow badge */
        .risk-badge {
            font-size: 1.8rem;
            font-weight: 700;
            padding: 10px 22px;
            border-radius: 12px;
            display: inline-block;
            margin-top: 5px;
            color: white;
        }
        .low-risk {
            background: #3cb371;
            box-shadow: 0 0 15px #3cb371aa;
        }
        .moderate-risk {
            background: #ffa500;
            box-shadow: 0 0 15px #ffa500aa;
        }
        .high-risk {
            background: #ff4d6d;
            box-shadow: 0 0 15px #ff4d6daa;
        }

        /* Recommendations card */
        .recommendation-card {
            border-radius: 12px;
            padding: 18px 20px;
            background: #f0f4ff;
            border: 1px solid #b6c8ff;
            margin-top: 12px;
        }

        /* Predict button */
        .stButton>button {
            background-color: #ff4d6d !important;
            color: white !important;
            padding: 14px 26px;
            font-size: 1.3rem;
            border-radius: 12px;
            font-weight: 700;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ================================
# Model + Feature Configuration
# ================================
MODEL_PATH = "models/best_model.pkl"

FEATURE_COLS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

ALL_MODEL_COLS = ["age_group"] + FEATURE_COLS


# ================================
# Helper Functions
# ================================
def compute_age_group(age: int) -> str:
    if age <= 54:
        return "40_54"
    elif age <= 69:
        return "55_69"
    else:
        return "70_plus"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def log_prediction_to_db(input_data, prediction, probability):
    engine = get_engine()
    cols = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak", "slope",
        "ca", "thal", "prediction", "probability",
    ]
    values = {col: input_data[col] for col in cols[:-2]}
    values["prediction"] = prediction
    values["probability"] = probability

    sql = text(
        f"INSERT INTO predictions_log ({', '.join(cols)}) "
        f"VALUES ({', '.join([f':{c}' for c in cols])})"
    )
    with engine.begin() as conn:
        conn.execute(sql, values)


# ================================
# Improved Clinical Recommendations
# ================================
def generate_recommendations(data, pred, proba):
    recs = []

    # ============ Accurate Clinical Thresholds ============

    # Cholesterol
    if data["chol"] >= 300:
        recs.append("Total cholesterol is very high — lifestyle change and a clinical lipid panel review are recommended.")
    elif data["chol"] >= 240:
        recs.append("Cholesterol is above the high-risk threshold — a heart-healthy diet and consistent exercise could help.")

    # Resting BP
    if data["trestbps"] >= 160:
        recs.append("Resting blood pressure is severely elevated — follow-up is strongly recommended.")
    elif data["trestbps"] >= 140:
        recs.append("Resting blood pressure is above normal; regular monitoring and reduction of salt/stress may help.")

    # ST Depression
    if data["oldpeak"] >= 3.0:
        recs.append("Marked ST depression may indicate ischemic strain — consider professional evaluation.")
    elif data["oldpeak"] >= 1.5:
        recs.append("ST depression is elevated above baseline — worth monitoring.")

    # Max heart rate (more accurate)
    if data["thalach"] <= 90:
        recs.append("Max heart rate is unusually low; this can indicate chronotropic incompetence — evaluation recommended.")
    elif data["thalach"] <= 120:
        recs.append("Max heart rate is below expected range for most adults; consider discussing with a clinician.")

    # Exercise-induced angina
    if data["exang"] == 1:
        recs.append("Exercise-induced chest pain reported — avoid strenuous activity until medically cleared.")

    # Chest pain type
    if data["cp"] == 3:
        recs.append("Chest pain type is consistent with symptomatic angina — monitoring advised.")
    elif data["cp"] == 2:
        recs.append("Atypical chest pain noted — track symptoms and discuss if persistent.")

    # Age
    if data["age"] >= 70:
        recs.append("Adults over 70 benefit significantly from routine cardiology checkups.")

    # Add high-risk–specific recommendation
    if pred == 2:  # high risk
        recs.insert(0, "Given the elevated predicted risk, a professional cardiac evaluation is recommended soon.")

    if not recs:
        return ["No significant abnormalities detected — continue maintaining healthy habits."]

    return recs


# ================================
# Streamlit App UI (UNCHANGED)
# ================================
def main():
    st.markdown("<div class='main-title'>Heart Disease Prediction System ❤️</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        "<div class='subtitle-text'>Enter patient information to estimate heart disease risk and receive simple wellness suggestions.</div>",
        unsafe_allow_html=True,
    )

    st.subheader("Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 18, 100, 50)
        sex = st.selectbox("Sex (1 = male, 0 = female)", [0, 1])
        cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
        trestbps = st.number_input("Resting Blood Pressure", 80, 220, 130)
        chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 250)
        fbs = st.selectbox("Fasting Blood Sugar > 120? (1 = yes, 0 = no)", [0, 1])

    with col2:
        restecg = st.selectbox("Resting ECG (0–2)", [0, 1, 2])
        thalach = st.number_input("Max Heart Rate", 60, 220, 150)
        exang = st.selectbox("Exercise-Induced Angina? (1 = yes, 0 = no)", [0, 1])
        oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0, step=0.1)
        slope = st.selectbox("Slope (0–2)", [0, 1, 2])
        ca = st.selectbox("Major Vessels (0–3)", [0, 1, 2, 3])
        thal = st.selectbox("Thal (1–3)", [1, 2, 3])

    age_group = compute_age_group(age)

    input_data = {
        "age_group": age_group,
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }

    if st.button("Predict Heart Disease Risk ❤️"):
        model = load_model()
        X_input = pd.DataFrame([input_data], columns=ALL_MODEL_COLS)

        proba = float(model.predict_proba(X_input)[0, 1])

        # ------------------ NEW RISK LEVEL -------------------
        if proba < 0.33:
            pred = 0     # low
        elif proba < 0.66:
            pred = 1     # moderate
        else:
            pred = 2     # high

        # ---------------- Prediction Section ----------------
        st.markdown("---")
        st.subheader("Prediction")

        if pred == 0:
            st.markdown("<div class='risk-badge low-risk'>Low Risk of Heart Disease</div>", unsafe_allow_html=True)
        elif pred == 1:
            st.markdown("<div class='risk-badge moderate-risk'>Moderate Risk of Heart Disease</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='risk-badge high-risk'>High Risk of Heart Disease</div>", unsafe_allow_html=True)

        st.markdown(f"### Estimated Risk Probability: `{proba:.3f}`")

        log_prediction_to_db(input_data, pred, proba)

        # ---------------- Recommendations ----------------
        st.markdown("---")
        st.subheader("Personalized Suggestions")

        recs = generate_recommendations(input_data, pred, proba)
        for r in recs:
            st.write(f"- {r}")

        # ---------------- Disclaimer ----------------
        st.markdown("---")
        st.markdown(
            """
            ### Disclaimer  
            This tool is for educational and demonstration purposes only.  
            Do not rely on it for medical decisions or diagnosis.  
            For real clinical concerns, consult a licensed healthcare professional.  
            — Created by Darsh J. Shah
            """
        )


if __name__ == "__main__":
    main()
