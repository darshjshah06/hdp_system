import numpy as np
import pandas as pd
from sqlalchemy import text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

from .db import get_engine
from .config import (
    TARGET_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
    MODEL_DIR,
    SCALER_PATH,
    ENCODER_PATH,
)

def fetch_features():
    engine = get_engine()
    query = "SELECT * FROM features"
    df = pd.read_sql(text(query), engine)
    return df

def build_preprocessor(df: pd.DataFrame):
    X = df.drop(columns=["id", TARGET_COLUMN])
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor, numeric_features, categorical_features

def prepare_data():
    df = fetch_features()
    X = df.drop(columns=["id", TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    preprocessor, num_cols, cat_cols = build_preprocessor(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Fit preprocessor on training data
    preprocessor.fit(X_train)

    # Save pieces for later use in app
    # Note: ColumnTransformer is saved within the model pipeline, but we can also keep info here
    # If you want separate scaler/encoder, you could extract them; for simplicity we save preprocessor via joblib in training.
    return X_train, X_test, y_train, y_test, preprocessor

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, preprocessor = prepare_data()
    print("Preprocessing complete.")
