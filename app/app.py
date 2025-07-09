import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load models
regressor = joblib.load("/Users/khalidmahmood/Coding Workspace/Will-My-Flight-Be-Delayed/model/rf_regressor.pkl")
classifier = joblib.load("/Users/khalidmahmood/Coding Workspace/Will-My-Flight-Be-Delayed/model/rf_classifier.pkl")
feature_columns = joblib.load("/Users/khalidmahmood/Coding Workspace/Will-My-Flight-Be-Delayed/model/feature_columns.pkl")

st.set_page_config(page_title="Will My Flight Be Delayed?", layout="centered")

st.title("✈️ Will My Flight Be Delayed?")
st.write("Predict delay rate and classify high/low risk based on flight info.")

# === User Inputs ===
st.subheader("Flight Details")

month = st.slider("Month", 1, 12, 6)
carrier = st.text_input("Carrier Code (e.g. UA, DL, AA)", "UA")
airport = st.text_input("Airport Code (e.g. ORD, ATL, LAX)", "ORD")

carrier_ct = st.number_input("Carrier-Caused Delays", 0.0, 1000.0, 10.0)
weather_ct = st.number_input("Weather-Caused Delays", 0.0, 1000.0, 5.0)
nas_ct = st.number_input("NAS-Caused Delays", 0.0, 1000.0, 8.0)
late_aircraft_ct = st.number_input("Late Aircraft Delays", 0.0, 1000.0, 7.0)
arr_flights = st.number_input("Number of Arrival Flights", 1.0, 5000.0, 100.0)
arr_del15 = st.number_input("Flights Delayed 15+ Minutes", 0.0, 5000.0, 20.0)

# === Build Input DataFrame ===
input_data = {
    "month": [month],
    "carrier_ct": [carrier_ct],
    "weather_ct": [weather_ct],
    "nas_ct": [nas_ct],
    "late_aircraft_ct": [late_aircraft_ct],
    "arr_flights": [arr_flights],
    "arr_del15": [arr_del15],
}

# One-hot encode carrier/airport
for col in feature_columns:
    if col.startswith("carrier_"):
        input_data[col] = [1 if col == f"carrier_{carrier}" else 0]
    elif col.startswith("airport_"):
        input_data[col] = [1 if col == f"airport_{airport}" else 0]
    elif col not in input_data:
        input_data[col] = [0]

input_df = pd.DataFrame(input_data)[feature_columns]

print("Model expects:", regressor.feature_names_in_)
print("You provided:", input_df.columns.tolist())



# === Prediction Button ===
if st.button("Predict Delay"):
    delay_rate = regressor.predict(input_df)[0]
    high_risk = classifier.predict(input_df)[0]

    st.subheader("🔍 Prediction Results")
    st.write(f"**Predicted Delay Rate:** `{delay_rate:.2f}`")

    if high_risk:
        st.error("⚠️ High Risk of Delay (≥ 20%)")
    else:
        st.success("✅ Low Risk of Delay (< 20%)")
