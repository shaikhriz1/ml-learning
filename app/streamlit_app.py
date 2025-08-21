import os
import pandas as pd
import joblib
import streamlit as st

st.title("ML Demo App")

model_path = "models/model.joblib"
if not os.path.exists(model_path):
    st.warning("No model found. Train one with `python src/train.py` first.")
else:
    model = joblib.load(model_path)
    st.success("Model loaded.")

st.write("Upload a small CSV with the same columns used in training (except target).")
file = st.file_uploader("Upload CSV", type=["csv"])

if file is not None and os.path.exists(model_path):
    df = pd.read_csv(file)
    X = df.select_dtypes(include=["number"]).fillna(0)
    preds = model.predict(X)
    st.write("Predictions:")
    st.write(pd.DataFrame(preds, columns=["prediction"]))
