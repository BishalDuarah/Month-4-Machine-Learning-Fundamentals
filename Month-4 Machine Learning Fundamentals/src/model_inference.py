import joblib
import numpy as np
import pandas as pd
import os


MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "random_forest_model.pkl")
PREPROCESSOR_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "preprocessor.pkl")


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor


def validate_input(input_data: dict) -> pd.DataFrame:
    """
    Validate and convert input dictionary to DataFrame.
    """
    if not isinstance(input_data, dict):
        raise ValueError("Input must be a dictionary.")

    df = pd.DataFrame([input_data])
    return df


def predict_churn(input_data: dict, threshold: float = 0.5):
    """
    Predict churn probability and label.
    """
    model, preprocessor = load_artifacts()

    df = validate_input(input_data)

    processed_data = preprocessor.transform(df)

    probability = model.predict_proba(processed_data)[0][1]
    prediction = int(probability >= threshold)

    return {
        "churn_probability": float(probability),
        "churn_prediction": prediction
    }