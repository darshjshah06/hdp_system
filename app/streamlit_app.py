def load_prediction_history():
    engine = get_engine()
    query = """
        SELECT id, created_at, age, sex, cp, trestbps, chol, fbs,
               restecg, thalach, exang, oldpeak, slope, ca, thal,
               prediction, probability
        FROM predictions_log
        ORDER BY created_at DESC;
    """
    with engine.begin() as conn:
        df = pd.read_sql(query, conn)
    return df


def main():
    st.title("Heart Disease Prediction System (HDP)")
    tab1, tab2 = st.tabs(["🔮 Make Prediction", "📊 Prediction History"])

    # ---------------------------
    # TAB 1 — PREDICTION
    # ---------------------------
    with tab1:
        st.write("Enter patient information to predict heart disease risk.")

        age = st.number_input("Age", min_value=18, max_value=100, value=50)
        sex = st.selectbox("Sex (1 = male, 0 = female)", [0, 1], index=1)
        cp = st.selectbox("Chest pain type (0–3)", [0, 1, 2, 3], index=0)
        trestbps = st.number_input("Resting blood pressure", min_value=80, max_value=220, value=130)
        chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=250)
        fbs = st.selectbox("Fasting blood sugar > 120?", [0, 1])
        restecg = st.selectbox("Rest ECG", [0, 1, 2])
        thalach = st.number_input("Max heart rate", min_value=60, max_value=220, value=150)
        exang = st.selectbox("Exercise-induced angina?", [0, 1])
        oldpeak = st.number_input("ST depression", min_value=0.0, max_value=10.0, value=1.0)
        slope = st.selectbox("Slope", [0, 1, 2], index=1)
        ca = st.selectbox("Major vessels (0–3)", [0, 1, 2, 3])
        thal = st.selectbox("Thal", [1, 2, 3], index=1)

        input_data = {
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
            "chol": chol, "fbs": fbs, "restecg": restecg,
            "thalach": thalach, "exang": exang, "oldpeak": oldpeak,
            "slope": slope, "ca": ca, "thal": thal
        }

        if st.button("Predict"):
            model = load_model()
            X_input = pd.DataFrame([input_data])

            proba = float(model.predict_proba(X_input)[0, 1])
            pred = int(proba >= 0.5)

            st.subheader("Prediction")
            st.write(f"Risk probability: **{proba:.3f}**")
            st.error("HIGH RISK (1)") if pred == 1 else st.success("LOW RISK (0)")

            log_prediction_to_db(input_data, pred, proba)
            st.caption("Prediction saved to database.")

    # ---------------------------
    # TAB 2 — HISTORY
    # ---------------------------
    with tab2:
        st.subheader("📊 Prediction History from Database")

        try:
            df = load_prediction_history()
            st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(f"Could not load history: {e}")

