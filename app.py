import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("model_gb.pkl")
st.title("Heart Disease Prediction App")

# Inputs
age = st.number_input("Age")
sex = st.selectbox("Sex", ["M", "F"])
chest = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
bp = st.number_input("Resting BP")
chol = st.number_input("Cholesterol")
fasting = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
maxhr = st.number_input("Max Heart Rate")
angina = st.selectbox("Exercise Angina", ["Y", "N"])
oldpeak = st.number_input("Oldpeak")
slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# Encoding SAME as training
sex = 1 if sex == "M" else 0
angina = 1 if angina == "Y" else 0

chest_dict = {"ATA":0, "NAP":1, "ASY":2, "TA":3}
ecg_dict = {"Normal":0, "ST":1, "LVH":2}
slope_dict = {"Up":0, "Flat":1, "Down":2}

chest = chest_dict[chest]
ecg = ecg_dict[ecg]
slope = slope_dict[slope]

# Prediction
if st.button("Predict"):
    input_data = np.array([[age, sex, chest, bp, chol, fasting, ecg, maxhr, angina, oldpeak, slope]])

    result = model.predict(input_data)

    if result[0] == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease")