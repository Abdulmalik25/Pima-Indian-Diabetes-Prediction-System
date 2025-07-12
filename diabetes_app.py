import streamlit as st
import pickle
import numpy as np

# Load model and scaler
with open('diabetes_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("🩺 Diabetes Prediction App")

# Input UI
pregnancies = st.number_input("Pregnancies", 0, 20, 2)
glucose = st.slider("Glucose", 0, 200, 120)
blood_pressure = st.slider("Blood Pressure", 0, 122, 70)
skin_thickness = st.slider("Skin Thickness", 0, 100, 20)
insulin = st.slider("Insulin", 0, 846, 79)
bmi = st.slider("BMI", 0.0, 70.0, 32.0)
dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.slider("Age", 10, 100, 33)

# When user clicks predict
if st.button("Predict"):
    # Arrange features in the same order as training
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])

    # Apply the same scaler
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    # Output
    if prediction == 1:
        st.error(f"Prediction: Diabetic ⚠️ (Confidence: {prob:.2%})")
    else:
        st.success(f"Prediction: Not Diabetic ✅ (Confidence: {1 - prob:.2%})")
