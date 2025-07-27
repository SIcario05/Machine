import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import pickle

# load model and scaler
model = load_model("heart_mlp_model.h5")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("🫀 Heart Failure Prediction App")

st.write("Enter clinical data to predict the risk of death event.")

# input fields based on dataset
age = st.number_input("Age", 20, 120, 60)
anaemia = st.selectbox("Anaemia (1 = Yes, 0 = No)", [0, 1])
creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase", 20, 8000, 500)
diabetes = st.selectbox("Diabetes (1 = Yes, 0 = No)", [0, 1])
ejection_fraction = st.slider("Ejection Fraction (%)", 10, 80, 40)
high_blood_pressure = st.selectbox("High Blood Pressure (1 = Yes, 0 = No)", [0, 1])
platelets = st.number_input("Platelets (kiloplatelets/mL)", 20000, 900000, 250000)
serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", 0.1, 10.0, 1.2)
serum_sodium = st.number_input("Serum Sodium (mEq/L)", 110, 150, 135)
sex = st.selectbox("Sex (1 = Male, 0 = Female)", [0, 1])
smoking = st.selectbox("Smoking (1 = Yes, 0 = No)", [0, 1])
time = st.slider("Follow-up period (days)", 0, 300, 100)

# build input array
input_data = np.array([[age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
                        high_blood_pressure, platelets, serum_creatinine, serum_sodium,
                        sex, smoking, time]])

# scale input
input_scaled = scaler.transform(input_data)

# predict
if st.button("Predict"):
    prediction = model.predict(input_scaled)
    result = "⚠️ High Risk (Death Event Likely)" if prediction[0][0] > 0.5 else "✅ Low Risk (Death Event Unlikely)"
    st.subheader("Prediction Result:")
    st.success(result)
