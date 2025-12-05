import os
import sys

# Allow imports from project root (hdp_system/)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import pandas as pd
import streamlit as st
from sqlalchemy import text

from src.hdp_system.db import get_engine



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

# Model requires age_group + numeric features
ALL_MODEL_COLS = ["age_group"] + FEATURE_COLS



# Helper Functions
# ================================

def compute_age_group(age: int) -> str:
    """Replicates SQL-defined age bins."""
    if age <= 54:
        return "40_54"
    elif age <= 69:
        return "55_69"
    else:
        return "70_plus"


@st.cache_resource
def load_model():
    """Load the trained ML pipeline."""
    return joblib.load(MODEL_PATH)


def log_prediction_to_db(input_data: dict, prediction: int, probability: float):
    """Insert a prediction into the predictions_log table."""
    engine = get_engine()

    cols = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak", "slope",
        "ca", "thal", "prediction", "probability",
    ]

    values = {col: input_data[col] for col in cols[:-2]}
    values["prediction"] = prediction
    values["probability"] = probability

    sql = text(f"""
        INSERT INTO predictions_log
        ({", ".join(cols)})
        VALUES ({", ".join([f":{c}" for c in cols])})
    """)

    try:
        with engine.begin() as conn:
            conn.execute(sql, values)
    except Exception as e:
        st.warning(f"⚠️ Could not log prediction: {e}")


def generate_recommendations(data, pred, proba):
    """Provide personalized wellness suggestions."""
    recs = []

    # Cholesterol
    if data["chol"] > 240:
        recs.append("• Your cholesterol level is elevated. Consider a lipid panel review and dietary adjustments.")

    # Blood pressure
    if data["trestbps"] > 140:
        recs.append("• Your resting blood pressure is high. Daily walking and reduced salt intake may help.")

    # ST depression
    if data["oldpeak"] > 2.0:
        recs.append("• Elevated ST depression may indicate strain. Follow-up testing could be beneficial.")

    # Age
    if data["age"] > 60:
        recs.append("• Adults over 60 benefit greatly from annual cardiovascular screenings.")

    # Exercise-induced angina
    if data["exang"] == 1:
        recs.append("• You reported exercise-induced chest discomfort. Consider moderated activity and medical evaluation.")

    # Chest pain types
    if data["cp"] in [2, 3]:
        recs.append("• Your chest pain type may warrant further assessment depending on symptom severity.")

    # General wrapping
    if pred == 1:
        recs.insert(0, "• Because your predicted risk is elevated, a formal check-up is recommended.")

    if not recs:
        return ["• No major red flags detected from your inputs — maintain a healthy lifestyle and regular check-ups."]

    return recs

# Streamlit App UI
# ================================

def main():
    st.title("❤️ Heart Disease Prediction System (HDP)")
    st.write(
        "Enter patient information to estimate heart disease risk. "
        "This model was built using clinical features and trained through a reproducible ML pipeline."
    )

    # Input Form
    # -----------------------
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

    # Construct full input record
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

    # Prediction
    # -----------------------
    if st.button("Predict"):
        model = load_model()

        X_input = pd.DataFrame([input_data], columns=ALL_MODEL_COLS)

        proba = float(model.predict_proba(X_input)[0, 1])
        pred = int(proba >= 0.5)

        st.subheader("Prediction Result")
        st.write(f"**Estimated Risk Probability:** `{proba:.3f}`")

        if pred == 1:
            st.error("Model Prediction: **HIGH RISK** of Heart Disease (1)")
        else:
            st.success("Model Prediction: **LOW RISK** of Heart Disease (0)")

        # Log prediction
        log_prediction_to_db(input_data, pred, proba)

        # Clinical Recommendations
        # -----------------------
        st.subheader("🩺 Personalized Recommendations")

        recs = generate_recommendations(input_data, pred, proba)
        for r in recs:
            st.write(r)

        # Disclaimer
        # -----------------------
        st.markdown("---")
        st.markdown(
            """
            ### ⚠️ Disclaimer  
            Please do not rely solely on this tool for any medical decision-making.  
            If you have genuine concerns or symptoms, seek professional medical help immediately.  
            This project was created for **educational and demonstration purposes only**.  

            **— By the creator, Darsh J. Shah**
            """
        )


if __name__ == "__main__":
    main()
