import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib
import os
from model import StreamANN  # your model definition file
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

# === Load preprocessing ===
try:
    scaler = joblib.load('app/scaler.pkl')
    career_ohe = joblib.load('app/career_ohe.pkl')
    le = joblib.load('app/label_encoder.pkl')
except FileNotFoundError as e:
    st.error(f"❌ Missing preprocessing file: {e.filename}")
    st.stop()

# === Prepare input dimensions ===
num_numeric_features = scaler.transform(np.zeros((1, 7))).shape[1]
num_career_features = len(career_ohe.get_feature_names_out(['career_aspiration']))
input_dim = num_numeric_features + num_career_features

# === Load model ===
model_path = 'app/model.pth'
if not os.path.exists(model_path):
    st.error(f"❌ Model file not found at: {model_path}")
    st.stop()

model = StreamANN(input_dim=input_dim, output_dim=4)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# === UI ===
st.title("🔮 Stream Recommendation System")
st.write("Enter the student's academic details and career aspiration to get a stream suggestion!")

with st.form("input_form"):
    math_score = st.number_input("Math Score", min_value=0, max_value=100)
    physics_score = st.number_input("Physics Score", min_value=0, max_value=100)
    biology_score = st.number_input("Biology Score", min_value=0, max_value=100)
    chemistry_score = st.number_input("Chemistry Score", min_value=0, max_value=100)
    english_score = st.number_input("English Score", min_value=0, max_value=100)
    history_score = st.number_input("History Score", min_value=0, max_value=100)
    geography_score = st.number_input("Geography Score", min_value=0, max_value=100)
    career_aspiration = st.selectbox("Career Aspiration", career_ohe.categories_[0].tolist())

    submitted = st.form_submit_button("Predict Stream")

if submitted:
    numeric_features = np.array([[math_score, physics_score, biology_score, chemistry_score,
                                  english_score, history_score, geography_score]])
    numeric_scaled = scaler.transform(numeric_features)

    try:
        career_encoded = career_ohe.transform([[career_aspiration]])
    except ValueError:
        st.error(f"❌ Career aspiration '{career_aspiration}' was not in training set.")
        st.stop()

    combined_input = np.hstack((numeric_scaled, career_encoded))
    input_tensor = torch.tensor(combined_input, dtype=torch.float32)

    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
        predicted_stream = le.inverse_transform([predicted_class])[0]

    st.success(f"🎓 **Recommended Stream:** {predicted_stream}")

    # === LIME Explainability ===
    try:
        background = np.load('app/background.npy')
    except Exception:
        st.warning("⚠️ Background data for LIME not found. Using current input as background.")
        background = combined_input

    feature_names = (
        ['math_score', 'physics_score', 'biology_score', 'chemistry_score',
         'english_score', 'history_score', 'geography_score']
        + list(career_ohe.get_feature_names_out(['career_aspiration']))
    )

    predict_fn = lambda x: model(torch.tensor(x, dtype=torch.float32)).detach().numpy()

    explainer = LimeTabularExplainer(
        training_data=background,
        feature_names=feature_names,
        class_names=le.classes_,
        mode="classification"
    )

    explanation = explainer.explain_instance(
        data_row=combined_input[0],
        predict_fn=predict_fn,
        num_features=10
    )

    fig = explanation.as_pyplot_figure()
    st.pyplot(fig)
