❤️ Heart Disease Prediction System

A complete end-to-end machine learning pipeline for predicting heart disease using:

PostgreSQL (data storage)
SQL feature engineering
Python ML pipeline (Scikit-Learn + XGBoost)
Streamlit dashboard (interactive predictions)
Model persistence (Joblib)
Prediction logging (stored in database)

Built to demonstrate real-world MLOps structure: ingestion → cleaning → feature engineering → training → evaluation → deployment UI.

🚀 Features:

End-to-end SQL + Python ML pipeline
Raw data → Cleaned → Features → Models → Predictions.
Trains + compares 3 models
Logistic Regression
Random Forest
XGBoost
Automatically saves the best model.
Streamlit web app
Fill in patient information → get prediction + probability.
Prediction Logging
Every prediction is saved to PostgreSQL for tracking.
Modular folder structure
Easy to extend, deploy, or improve.

Project Structure:

hdp_system/
│
├── app/
│   └── streamlit_app.py        # Main UI
│
├── data/
│   └── raw/heart.csv           # Dataset
│
├── models/
│   └── best_model.pkl          # Saved model
│
├── sql/
│   ├── schema.sql              # Database schema
│   ├── cleaning.sql            # Cleaning logic
│   └── feature_engineering.sql # Feature creation
│
├── src/hdp_system/
│   ├── data_ingestion.py       # Runs SQL pipeline
│   ├── train_models.py         # Trains ML models
│   └── evaluate.py             # Performance metrics
│
├── .env                        # DATABASE_URL here
├── requirements.txt
└── README.md                   # This file


