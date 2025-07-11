import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent  # Goes up from /app to project root

# Load saved model and columns
regressor = joblib.load(base_dir / "model" / "rf_regressor.pkl")
classifier = joblib.load(base_dir / "model" / "rf_classifier.pkl")
feature_columns = joblib.load(base_dir / "model" / "feature_columns.pkl")

# App setup
st.set_page_config(page_title="Will My Flight Be Delayed?", layout="centered")
st.title("✈️ Will My Flight Be Delayed?")
st.write("This app predicts the likelihood of a flight delay using airline and airport information.")

# === User Inputs ===
st.subheader("Flight Info")
month = st.slider("Month of Flight", 1, 12, 6)
carrier = st.text_input("Carrier Code (e.g. UA, DL)", "UA").upper()
airport = st.text_input("Airport Code (e.g. ORD, ATL)", "ORD").upper()

# === Build Input Row ===
input_data = {
    "month": [month],
}

# One-hot encode carrier and airport like training
for col in feature_columns:
    if col.startswith("carrier_"):
        input_data[col] = [1 if col == f"carrier_{carrier}" else 0]
    elif col.startswith("airport_"):
        input_data[col] = [1 if col == f"airport_{airport}" else 0]
    elif col not in input_data:
        input_data[col] = [0]  # For any leftover columns

# Format input
input_df = pd.DataFrame(input_data)[feature_columns]

# === Prediction ===
if st.button("Predict Delay Rate"):
    delay_rate = regressor.predict(input_df)[0]
    high_delay = classifier.predict(input_df)[0]

    st.subheader("🔍 Result")
    st.write(f"**Estimated Delay Rate:** `{delay_rate:.2%}`")

    if high_delay:
        st.error("⚠️ High Risk of Delay (20%+)")
    else:
        st.success("✅ Low Risk of Delay (< 20%)")

