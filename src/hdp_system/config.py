import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://username:password@localhost:5432/hdp_db"
)

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "target"
MODEL_DIR = "models"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.joblib")
BACKGROUND_DATA_PATH = os.path.join(MODEL_DIR, "background_data.npy")
