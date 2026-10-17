❤️ Heart Disease Prediction System

A complete end-to-end machine learning pipeline for predicting heart disease using:
Loads & preprocesses heart disease clinical data
Stores cleaned data and engineered features in PostgreSQL
Trains ML models (Logistic Regression, RandomForest, XGBoost)
Selects & saves the best model
Provides a dark, Apple-style web app for real-time prediction
Logs prediction history into the database
Generates personalized clinical recommendations
Includes disclaimer and polished UI






🚀 Features:

End-to-End ML Workflow
Ingestion → Cleaning → Feature Engineering → Model Training → Evaluation → Deployment
Database-Backed System
PostgreSQL stores raw, cleaned, engineered data

Prediction events logged in predictions_log
Web Application
Dark, Apple-style UI with a hot pink accent
Real-time predictions
Personalized recommendation engine
Disclaimer for safe usage
Machine Learning
Models trained:
Logistic Regression
Random Forest
XGBoost
Best model automatically selected by AUC
Full sklearn pipeline with preprocessing




hdp_system/
│
├── app/
│   ├── server.py                 # Web server + /api/predict
│   └── static/                   # UI: index.html, style.css, app.js
│
├── src/hdp_system/
│   ├── data_ingestion.py         # Load → clean → engineer SQL features
│   ├── train_models.py           # Train + evaluate + save best model
│   ├── evaluate.py               # Extra metrics + plot
│   ├── predict.py                # Validation, risk level, recommendations, logging
│   ├── db.py                     # DB connection helper
│   └── config.py                 # Paths & env management
│
├── sql/
│   ├── schema.sql                # Database schema
│   ├── cleaning.sql              # Clean & normalize raw data
│   └── feature_engineering.sql   # Age-group + feature calculations
│
├── models/
│   └── best_model.pkl            # Saved ML pipeline
│
├── data/
│   └── raw/
│        └── heart.csv            # Input dataset
│
├── requirements.txt
└── README.md





🔨 Installation:

git clone <https>
cd hdp_system

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

CREATE DATABASE hdp_db;

psql -d hdp_db -f sql/schema.sql

python3 -m src.hdp_system.data_ingestion

python3 -m src.hdp_system.train_models

python3 -m src.hdp_system.evaluate

python3 app/server.py               # then open http://localhost:8501





✨ Recommendation Engine

The app generates personalized suggestions based on:

Cholesterol
Blood pressure
Age
ST depression
Angina
Chest pain type

These appear dynamically under prediction results.



⚠️ Disclaimer

Please do not rely solely on this tool for medical decision-making.
If you have health concerns, consult a medical professional.
This project is for educational purposes only.

— By the creator, Darsh J. Shah





📦 Technologies Used

Python 3.10+
HTML/CSS/JavaScript (no framework)
PostgretQL
SQLAlchemy
Scikit-Learn
XGBoost
Pandas
NumPy



🔁 Daily Auto-Commit

tools/autocommit/ commits and pushes once a day (19:00, via a macOS launchd agent). The commit message is picked at random from 5 short messages (never the same twice in a row). If nothing changed, it makes an empty commit so there is still one every day.

./tools/autocommit/install.sh             # install / reinstall
./tools/autocommit/install.sh uninstall   # remove
./tools/autocommit/autocommit.sh          # run once by hand

Log: ~/Library/Logs/hdp-system-autocommit.log. Edit the MESSAGES list in autocommit.sh to change the wording.
