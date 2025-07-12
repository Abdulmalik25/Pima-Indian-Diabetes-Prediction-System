import streamlit as st
import pickle
import numpy as np

# Page config
st.set_page_config(page_title="🧬 Diabetes Predictor", layout="centered", page_icon="🩺")

# Minimal dark theme + visible sidebar + safe styles
st.markdown("""
    <style>
    body {
        background-color: #121212;
        color: #f1f1f1;
    }
    .stApp {
        background-color: #1e1e1e;
        color: white;
    }
    h2 {
        background: linear-gradient(to right, #00c6ff, #0072ff);
        -webkit-background-clip: text;
        color: transparent;
        font-weight: 700;
        text-align: center;
    }
    .stButton>button {
        background-color: #00c6ff;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Header with white-colored text
st.markdown("""
<h2 style='color: #FFFFFF; text-align: center;'>🩺 Diabetes Prediction System</h2>
<p style='text-align: center; color: #DDDDDD; font-size: 16px;'>
🔎 Enter patient information below to check diabetes risk using a trained machine learning model.
</p>
""", unsafe_allow_html=True)


# Load model and scaler
with open('diabetes_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Sidebar inputs (now all labels will be visible)
st.sidebar.header("📋 Patient Information")

pregnancies = st.sidebar.number_input("👶 Pregnancies", 0, 20, 2)
glucose = st.sidebar.slider("🍬 Glucose", 0, 200, 120)
blood_pressure = st.sidebar.slider("💓 Blood Pressure", 0, 122, 72)
skin_thickness = st.sidebar.slider("🩹 Skin Thickness", 0, 100, 20)
insulin = st.sidebar.slider("💉 Insulin", 0, 900, 79)
bmi = st.sidebar.slider("⚖️ BMI", 0.0, 70.0, 32.0)
dpf = st.sidebar.slider("📈 Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.sidebar.slider("📅 Age", 10, 100, 33)

# Predict button
if st.button("🔍 Predict Diabetes"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    st.markdown("## 🎯 Prediction Result")
    if pred == 1:
        st.error(f"⚠️ The model predicts the patient is **Diabetic** — Confidence: **{prob:.2%}**")
    else:
        st.success(f"✅ The model predicts the patient is **Not Diabetic** — Confidence: **{1 - prob:.2%}**")

# Footer with visible light color
st.markdown("---")
st.markdown("""
<p style='text-align: center; color: #AAAAAA; font-size: 14px;'>
Made with ❤️ using Streamlit
</p>
""", unsafe_allow_html=True)
