import streamlit as st
import pandas as pd
import numpy as np
import pickle
import lzma
import os
import subprocess
import gdown
from datetime import datetime

# Streamlit config
st.set_page_config(page_title="Flight Fare Predictor", page_icon="✈️", layout="centered")
st.title("✈️ Flight Fare Predictor")
st.write("Enter flight details to get the estimated fare.")

model_path = "model.lzma"
file_id = "YUeTITr2t61ldmCnEZxp1cWCngg3mYvG"
gdown_url = f"https://drive.google.com/uc?id={file_id}"

# Download if missing
if not os.path.exists(model_path):
    with st.spinner("📥 Downloading model..."):
        gdown.download(gdown_url, model_path, quiet=False)
        st.success("✅ Model downloaded.")


# Load model with pickle + lzma
with lzma.open(model_path, "rb") as f:
    model = pickle.load(f)

# Trained columns
trained_columns = ['Class', 'Total_stops', 'Duration_in_hours', 'Days_left', 'day_Friday', 'day_Monday',
 'day_Saturday', 'day_Sunday', 'day_Thursday', 'day_Tuesday', 'day_Wednesday', 'source_Ahmedabad',
 'source_Bangalore', 'source_Chennai', 'source_Delhi', 'source_Hyderabad', 'source_Kolkata', 'source_Mumbai', 
 'departure_12 PM - 6 PM', 'departure_6 AM - 12 PM', 'departure_After 6 PM', 'departure_Before 6 AM',
 'arrival_12 PM - 6 PM', 'arrival_6 AM - 12 PM', 'arrival_After 6 PM', 'arrival_Before 6 AM',
 'destination_Ahmedabad', 'destination_Bangalore', 'destination_Chennai', 'destination_Delhi', 
 'destination_Hyderabad', 'destination_Kolkata', 'destination_Mumbai']

# User inputs
flight_date = st.date_input("🗓️ Flight Date")
flight_class = st.selectbox("💺 Class", ['Economy', 'Business'])
source = st.selectbox("🌆 Source", ['Ahmedabad', 'Bangalore', 'Chennai', 'Delhi', 'Hyderabad', 'Kolkata', 'Mumbai'])
destination = st.selectbox("🌇 Destination", ['Ahmedabad', 'Bangalore', 'Chennai', 'Delhi', 'Hyderabad', 'Kolkata', 'Mumbai'])
departure_time = st.selectbox("🕕 Departure Time", ['Before 6 AM', '6 AM - 12 PM', '12 PM - 6 PM', 'After 6 PM'])
arrival_time = st.selectbox("🕗 Arrival Time", ['Before 6 AM', '6 AM - 12 PM', '12 PM - 6 PM', 'After 6 PM'])
stops = st.selectbox("🔁 Stops", ['non-stop', '1-stop', '2-stops', '2+-stops'])
duration = st.number_input("⏱️ Duration (in hours)", min_value=0.0, max_value=24.0, step=0.1, value=2.0)

# Preprocess
days_left = max((flight_date - datetime.today().date()).days, 0)
class_encoded = 0 if flight_class == "Economy" else 1
stops_mapping = {'non-stop': 0, '1-stop': 1, '2-stops': 2, '2+-stops': 3}
total_stops = stops_mapping[stops]

input_dict = {
    'Class': [class_encoded],
    'Total_stops': [total_stops],
    'Duration_in_hours': [duration],
    'Days_left': [days_left]
}
df = pd.DataFrame(input_dict)

# Encode weekday
day_name = flight_date.strftime("%A")
for d in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
    df[f'day_{d}'] = 1 if day_name == d else 0

# Encode time slots
time_slots = ['Before 6 AM', '6 AM - 12 PM', '12 PM - 6 PM', 'After 6 PM']
for slot in time_slots:
    df[f'arrival_{slot}'] = 1 if arrival_time == slot else 0
    df[f'departure_{slot}'] = 1 if departure_time == slot else 0

# One-hot encode cities
for city in ['Ahmedabad', 'Bangalore', 'Chennai', 'Delhi', 'Hyderabad', 'Kolkata', 'Mumbai']:
    df[f'source_{city}'] = 1 if source == city else 0
    df[f'destination_{city}'] = 1 if destination == city else 0

# Fill and reorder
for col in trained_columns:
    if col not in df.columns:
        df[col] = 0
df = df[trained_columns]

# Predict
if st.button("🔍 Predict Fare"):
    fare = model.predict(df)[0]
    st.success(f"💰 Estimated Fare: ₹{int(fare)}")
