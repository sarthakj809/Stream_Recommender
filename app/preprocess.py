import numpy as np
import joblib

def preprocess_input(numeric_features, career_aspiration, scaler, career_ohe):
    """
    numeric_features: list or np.array of 7 numeric features (scaled)
    career_aspiration: string value of career aspiration
    scaler: loaded StandardScaler fitted on training numeric features
    career_ohe: loaded OneHotEncoder fitted on career_aspiration column

    Returns: combined numpy array of scaled numeric + one-hot career_aspiration
    """
    # Convert to numpy array
    numeric_array = np.array(numeric_features).reshape(1, -1)
    # Scale numeric
    numeric_scaled = scaler.transform(numeric_array)
    # One-hot encode career aspiration (expects 2D array)
    career_encoded = career_ohe.transform([[career_aspiration]])
    # Combine
    combined = np.hstack((numeric_scaled, career_encoded))
    return combined

def load_preprocessors(scaler_path, ohe_path, label_encoder_path):
    scaler = joblib.load(scaler_path)
    career_ohe = joblib.load(ohe_path)
    label_encoder = joblib.load(label_encoder_path)
    return scaler, career_ohe, label_encoder
