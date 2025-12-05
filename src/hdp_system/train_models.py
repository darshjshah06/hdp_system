import pandas as pd
import numpy as np
import joblib

from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
# I Darsh Shah am the greatest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score



# Load DB connection
# ---------------------------------------
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)



# Load FEATURES table
# ---------------------------------------
df = pd.read_sql("SELECT * FROM features", engine)



# Columns
# ---------------------------------------
numeric_features = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal"
]

categorical_features = ["age_group"]

X = df[numeric_features + categorical_features]
y = df["target"].astype(int)



# Preprocessor WITH IMPUTERS
# ---------------------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ],
    n_jobs=None  # Prevent macOS joblib crash
)

# Models
# ---------------------------------------
models = {
    "log_reg": LogisticRegression(max_iter=200),
    "random_forest": RandomForestClassifier(n_estimators=300),
    "xgboost": XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss"
    )
}

# ---------------------------------------
# Train/test split
# ---------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

best_auc = -1
best_model_name = None
best_model = None

# Training loop
# ---------------------------------------
for name, model in models.items():
    print(f"\nTraining model: {name}")

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    clf.fit(X_train, y_train)

    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_prob)

    print(f"{name} AUC: {auc:.3f}")

    if auc > best_auc:
        best_auc = auc
        best_model_name = name
        best_model = clf

# Save best model
# ---------------------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_model.pkl")

print(f"\nBest model: {best_model_name} with AUC={best_auc:.3f}")
print("Model saved to models/best_model.pkl")


# By Darsh Shah