import streamlit as st
import pandas as pd
import numpy as np
import pickle
import lzma
import os
import subprocess
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(page_title="Flight Fare Predictor", page_icon="✈️", layout="centered")
st.title("✈️ Flight Fare Predictor")
st.write("Enter flight details:") 
# Model load
# -----------------------------
model_path = "rf_flight_model.xz"

if not os.path.exists(model_path):
    with st.spinner("📥 Downloading model..."):
        gdown_url = "https://drive.google.com/uc?id=1YUeTITr2t61ldmCnEZxp1cWCngg3mYvG"
        subprocess.run(["gdown", gdown_url, "-O", model_path], check=True)
        st.success("✅ Model downloaded.")

with lzma.open(model_path, "rb") as f:
    model = pickle.load(f)

# -----------------------------
# Trained columns (must match training)
# -----------------------------
trained_columns = ['Class', 'Total_stops', 'Duration_in_hours', 'Days_left', 'day_Friday', 'day_Monday',
 'day_Saturday', 'day_Sunday', 'day_Thursday', 'day_Tuesday', 'day_Wednesday', 'source_Ahmedabad',
 'source_Bangalore', 'source_Chennai', 'source_Delhi', 'source_Hyderabad', 'source_Kolkata', 'source_Mumbai', 
 'departure_12 PM - 6 PM', 'departure_6 AM - 12 PM', 'departure_After 6 PM', 'departure_Before 6 AM',
 'arrival_12 PM - 6 PM', 'arrival_6 AM - 12 PM', 'arrival_After 6 PM', 'arrival_Before 6 AM',
 'destination_Ahmedabad', 'destination_Bangalore', 'destination_Chennai', 'destination_Delhi', 
 'destination_Hyderabad', 'destination_Kolkata', 'destination_Mumbai']

# -----------------------------
# User inputs
# -----------------------------
col1, col2 = st.columns(2)
with col1:
    flight_date = st.date_input("🗓️ Flight Date (the day you will fly)")
    flight_class = st.selectbox("💺 Class", ['Economy', 'Business'])
    source = st.selectbox("🌆 Source", ['Ahmedabad', 'Bangalore', 'Chennai', 'Delhi', 'Hyderabad', 'Kolkata', 'Mumbai'])
    destination = st.selectbox("🌇 Destination", ['Ahmedabad', 'Bangalore', 'Chennai', 'Delhi', 'Hyderabad', 'Kolkata', 'Mumbai'])

with col2:
    departure_time = st.selectbox("🕕 Departure Time", ['Before 6 AM', '6 AM - 12 PM', '12 PM - 6 PM', 'After 6 PM'])
    arrival_time = st.selectbox("🕗 Arrival Time", ['Before 6 AM', '6 AM - 12 PM', '12 PM - 6 PM', 'After 6 PM'])
    stops = st.selectbox("🔁 Stops", ['non-stop', '1-stop', '2-stops', '2+-stops'])
    duration = st.number_input("⏱️ Duration (in hours)", min_value=0.0, max_value=48.0, step=0.1, value=2.0)

st.markdown("---")

# -----------------------------
# Preprocessing function
# -----------------------------
def preprocess_template(flight_date, flight_class, source, destination, departure_time, arrival_time, stops, duration):
    """Return a DataFrame template (one row) with features derived from flight details.
       Note: Days_left will be overwritten for each candidate booking date before prediction.
    """
    days_left_now = max((flight_date - datetime.today().date()).days, 0)
    class_encoded = 0 if flight_class == "Economy" else 1
    stops_mapping = {'non-stop': 0, '1-stop': 1, '2-stops': 2, '2+-stops': 3}
    total_stops = stops_mapping[stops]

    df = pd.DataFrame({
        'Class': [class_encoded],
        'Total_stops': [total_stops],
        'Duration_in_hours': [duration],
        'Days_left': [days_left_now]  # placeholder; will be replaced for each booking date
    })

    # Weekday of the flight (these remain constant because flight day doesn't change)
    flight_weekday = flight_date.strftime("%A")
    for d in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
        df[f'day_{d}'] = 1 if flight_weekday == d else 0

    # Departure / arrival slots (constant)
    time_slots = ['Before 6 AM', '6 AM - 12 PM', '12 PM - 6 PM', 'After 6 PM']
    for slot in time_slots:
        df[f'arrival_{slot}'] = 1 if arrival_time == slot else 0
        df[f'departure_{slot}'] = 1 if departure_time == slot else 0

    # Cities one-hot (constant)
    for city in ['Ahmedabad', 'Bangalore', 'Chennai', 'Delhi', 'Hyderabad', 'Kolkata', 'Mumbai']:
        df[f'source_{city}'] = 1 if source == city else 0
        df[f'destination_{city}'] = 1 if destination == city else 0

    # Ensure all trained columns present
    for col in trained_columns:
        if col not in df.columns:
            df[col] = 0

    # Reorder to match training
    return df[trained_columns]

# -----------------------------
# Prediction for booking dates (today..horizon)
# -----------------------------
if st.button("🔍 Predict Lowest Fare"):
    today = datetime.today().date()
    days_until_flight = (flight_date - today).days

    if days_until_flight < 0:
        st.error("Selected flight date is in the past. Choose a future date.")
    else:
        # Build template (flight-specific features)
        template = preprocess_template(flight_date, flight_class, source, destination, departure_time, arrival_time, stops, duration)

        # Determine simulation horizon:
        # simulate booking dates from today up to min(6, days_until_flight) => that's up to 7 booking-days (0..6) or less if flight is sooner
        max_booking_offset = min(6, max(days_until_flight, 0))
        booking_offsets = list(range(0, max_booking_offset + 1))  # e.g., [0,1,2,...]

        booking_dates = [today + timedelta(days=o) for o in booking_offsets]
        predicted_fares = []

        for book_date in booking_dates:
            df_future = template.copy()
            # Days_left if you buy on book_date
            days_left_if_buy = max((flight_date - book_date).days, 0)
            df_future['Days_left'] = days_left_if_buy

            # NOTE: flight weekday/features remain that of the flight (we assume model expects flight-day weekday)
            # Predict
            pred = model.predict(df_future)[0]
            predicted_fares.append(float(pred))

        predicted_fares = np.array(predicted_fares, dtype=float)

        # Find best booking date (lowest fare)
        min_idx = int(np.argmin(predicted_fares))
        best_booking_date = booking_dates[min_idx]
        best_price = predicted_fares[min_idx]

        # Show today's predicted fare explicitly
        st.success(f"Predicted fare if you book **today ({today.strftime('%d %b %Y')})**: ₹{int(predicted_fares[0])}")

        # Decision message
        if min_idx == 0:
            st.success(f"✅ Book Now — booking today is the **lowest predicted price** (₹{int(best_price)})")
        else:
            days_wait = min_idx
            st.info(f"⏳ Wait — lowest predicted price is in **{days_wait} day(s)**: {best_booking_date.strftime('%d %b %Y')} (₹{int(best_price)})")

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(5, 2.5))
        ax.plot(booking_dates, predicted_fares, marker='o', linewidth=1, color='blue')
        ax.set_title("Predicted fare for different booking dates")
        ax.set_xlabel("Booking date", fontsize=8)
        ax.set_ylabel("Fare (₹)", fontsize=8)
        ax.tick_params(axis='x', rotation=30, labelsize=7)
        ax.tick_params(axis='y', labelsize=7)
        ax.scatter([booking_dates[np.argmin(predicted_fares)]], [min(predicted_fares)], color='red', s=60, zorder=5)
        ax.annotate(f"Lowest\n₹{int(min(predicted_fares))}", xy=(booking_dates[np.argmin(predicted_fares)], min(predicted_fares)),
                    xytext=(0, -28), textcoords="offset points", ha='center', fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))
        st.pyplot(fig)
