# 🫀 Heart Failure Prediction App

This is a simple Streamlit web app that predicts the risk of a **death event** based on clinical data.

### 💡 How it works
- The model is a **MultiLayer Perceptron (MLP)** trained on the Heart Failure Clinical Records Dataset.
- It uses 12 features (like age, serum sodium, diabetes, etc.) to predict the risk.
- Data is preprocessed with a `StandardScaler`.

### 📁 Files
- `streamlit_app.py` — the main app
- `heart_mlp_model.h5` — the trained model
- `scaler.pkl` — the scaler used in preprocessing
- `requirements.txt` — dependencies

### ▶️ How to run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
