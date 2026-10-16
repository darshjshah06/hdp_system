"""Prediction logic shared by the web app: input validation, risk level,
clinical recommendations, and logging to predictions_log."""
import os
from functools import lru_cache

import joblib
import pandas as pd
from sqlalchemy import text

from .db import get_engine

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(ROOT, "models", "best_model.pkl")

# Allowed range for each input: (min, max, is_integer).
FIELDS = {
    "age": (18, 100, True),
    "sex": (0, 1, True),
    "cp": (0, 3, True),
    "trestbps": (80, 220, True),
    "chol": (100, 600, True),
    "fbs": (0, 1, True),
    "restecg": (0, 2, True),
    "thalach": (60, 220, True),
    "exang": (0, 1, True),
    "oldpeak": (0.0, 10.0, False),
    "slope": (0, 2, True),
    "ca": (0, 3, True),
    "thal": (1, 3, True),
}
FEATURE_COLS = list(FIELDS)
ALL_MODEL_COLS = ["age_group"] + FEATURE_COLS
LEVELS = ["low", "moderate", "high"]


def compute_age_group(age: int) -> str:
    if age <= 54:
        return "40_54"
    elif age <= 69:
        return "55_69"
    else:
        return "70_plus"


@lru_cache(maxsize=1)
def load_model():
    return joblib.load(MODEL_PATH)


def validate(raw: dict) -> dict:
    """Coerce and range-check the request body. Raises ValueError with a
    message fit to show the user."""
    data = {}
    for key, (lo, hi, is_int) in FIELDS.items():
        if key not in raw:
            raise ValueError(f"Missing field: {key}")
        try:
            value = float(raw[key])
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be a number")
        if is_int:
            if value != int(value):
                raise ValueError(f"{key} must be a whole number")
            value = int(value)
        if not lo <= value <= hi:
            raise ValueError(f"{key} must be between {lo} and {hi}")
        data[key] = value
    return data


def generate_recommendations(data, pred):
    recs = []

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

    # Max heart rate
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

    if pred == 2:  # high risk
        recs.insert(0, "Given the elevated predicted risk, a professional cardiac evaluation is recommended soon.")

    if not recs:
        return ["No significant abnormalities detected — continue maintaining healthy habits."]

    return recs


def log_prediction_to_db(data, prediction, probability):
    cols = FEATURE_COLS + ["prediction", "probability"]
    values = {**{c: data[c] for c in FEATURE_COLS}, "prediction": prediction, "probability": probability}
    sql = text(
        f"INSERT INTO predictions_log ({', '.join(cols)}) "
        f"VALUES ({', '.join(f':{c}' for c in cols)})"
    )
    with get_engine().begin() as conn:
        conn.execute(sql, values)


def predict(raw: dict) -> dict:
    data = validate(raw)
    row = {"age_group": compute_age_group(data["age"]), **data}
    X = pd.DataFrame([row], columns=ALL_MODEL_COLS)
    proba = float(load_model().predict_proba(X)[0, 1])

    if proba < 0.33:
        pred = 0
    elif proba < 0.66:
        pred = 1
    else:
        pred = 2

    # A database outage shouldn't block the prediction itself.
    try:
        log_prediction_to_db(data, pred, proba)
        logged = True
    except Exception as e:
        print(f"predictions_log insert failed: {e}")
        logged = False

    return {
        "probability": proba,
        "level": LEVELS[pred],
        "recommendations": generate_recommendations(data, pred),
        "logged": logged,
    }
