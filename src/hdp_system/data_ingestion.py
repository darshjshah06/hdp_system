import pandas as pd
from sqlalchemy import text
from .db import get_engine

def load_csv_to_patients_raw(csv_path: str):
    """
    Load the heart disease CSV into the patients_raw table.
    """
    df = pd.read_csv(csv_path)
    engine = get_engine()

    # Ensure column names match schema.sql
    df.to_sql("patients_raw", engine, if_exists="append", index=False)

def run_sql_script(path: str):
    engine = get_engine()
    with open(path, "r") as f:
        sql = f.read()
    with engine.begin() as conn:
        conn.execute(text(sql))

if __name__ == "__main__":
    # 1. Load CSV to patients_raw
    load_csv_to_patients_raw("data/raw/heart.csv")

    # 2. Run cleaning and feature engineering
    run_sql_script("sql/cleaning.sql")
    run_sql_script("sql/feature_engineering.sql")
    print("Data ingestion and feature engineering completed.")
