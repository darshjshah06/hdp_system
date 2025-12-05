import numpy as np
import joblib
import shap

from .config import BEST_MODEL_PATH, BACKGROUND_DATA_PATH

def get_shap_explainer():
    model = joblib.load(BEST_MODEL_PATH)
    background = np.load(BACKGROUND_DATA_PATH)
    explainer = shap.KernelExplainer(model.predict_proba, background)
    return explainer, model

def explain_instance(input_array):
    """
    input_array: shape (1, n_features_raw) BEFORE preprocessing
    The pipeline will handle preprocessing internally.
    We just pass a function to SHAP that accepts raw input.
    """
    model = joblib.load(BEST_MODEL_PATH)

    def f(X):
        # X is raw feature array, we need to ensure it goes through pipeline
        return model.predict_proba(X)[:, 1]

    background = np.load(BACKGROUND_DATA_PATH)
    explainer = shap.KernelExplainer(f, background)

    shap_values = explainer.shap_values(input_array)
    return shap_values

    # By Darsh
