import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle

# load model and scaler
model = load_model("heart_mlp_model.h5")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("❤️ Heart Failure Prediction App")

st.write("Enter patient info to predict if they are at risk of a **Death Event**.")

# input fields
age = st.number_input("Age", 18, 100, 50)
sex = st.selectbox("Sex", [0, 1])  # 1 = male, 0 = female
cp = st.slider("Chest Pain Type (0-3)", 0, 3, 0)
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG results (0-2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina (1 = yes, 0 = no)", [0, 1])
oldpeak = st.number_input("ST depression induced by exercise", 0.0, 6.0, 1.0, step=0.1)
slope = st.selectbox("Slope of peak exercise ST (0-2)", [0, 1, 2])
ca = st.selectbox("Number of major vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thal (1 = normal; 2 = fixed defect; 3 = reversible defect)", [1, 2, 3])

# collect input
input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                        exang, oldpeak, slope, ca, thal]])

# preprocess
input_scaled = scaler.transform(input_data)

# predict
if st.button("Predict"):
    prediction = model.predict(input_scaled)
    result = "🚨 High Risk (Death Event Likely)" if prediction[0][0] > 0.5 else "✅ Low Risk (Death Event Unlikely)"
    st.subheader("Result:")
    st.success(result)
