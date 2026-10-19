"""Streamlit frontend for the airline arrival-delay prediction API."""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("AIRLINE_API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(
    page_title="Airline Arrival Delay Predictor",
    page_icon="✈️",
    layout="centered",
)

st.title("✈️ Airline Arrival Delay Predictor")
st.write(
    "Estimate arrival delay after departure. Enter the flight and passenger "
    "details below; the frontend sends them to the prediction API."
)
st.info(
    "This model requires the departure delay to be known. It does not use "
    "post-flight service ratings or satisfaction survey answers."
)

try:
    health_response = requests.get(f"{API_URL}/health", timeout=3)
    health_response.raise_for_status()
    health = health_response.json()
except requests.RequestException as error:
    st.error(
        "The prediction API is unavailable. Start the backend first, then "
        "refresh this page. "
        f"Details: {error}"
    )
    st.stop()

test_mae = health.get("test_mae_minutes")
if test_mae is not None:
    st.caption(f"Model test-set MAE: {float(test_mae):.2f} minutes")

with st.form("flight-details"):
    st.subheader("Flight and passenger details")
    first_column, second_column = st.columns(2)

    with first_column:
        age = st.number_input("Passenger age", min_value=0, max_value=120, value=35)
        flight_distance = st.number_input(
            "Flight distance (miles)", min_value=0, max_value=20_000, value=800
        )
        departure_delay = st.number_input(
            "Departure delay (minutes)",
            min_value=0,
            max_value=10_000,
            value=20,
        )
        gender = st.selectbox("Gender", ["Female", "Male"])

    with second_column:
        customer_type = st.selectbox(
            "Customer type", ["Loyal Customer", "disloyal Customer"]
        )
        travel_type = st.selectbox(
            "Type of travel", ["Business travel", "Personal Travel"]
        )
        flight_class = st.selectbox("Flight class", ["Business", "Eco", "Eco Plus"])

    submitted = st.form_submit_button(
        "Predict arrival delay", type="primary", use_container_width=True
    )

if submitted:
    payload = {
        "age": age,
        "flight_distance": flight_distance,
        "departure_delay_minutes": departure_delay,
        "gender": gender,
        "customer_type": customer_type,
        "type_of_travel": travel_type,
        "flight_class": flight_class,
    }
    try:
        prediction_response = requests.post(
            f"{API_URL}/predict", json=payload, timeout=10
        )
        prediction_response.raise_for_status()
        result = prediction_response.json()
    except requests.HTTPError as error:
        detail = error.response.text if error.response is not None else str(error)
        st.error(f"The API rejected the prediction request: {detail}")
    except requests.RequestException as error:
        st.error(f"Could not reach the prediction API: {error}")
    else:
        delay = float(result["predicted_arrival_delay_minutes"])
        st.metric("Estimated arrival delay", f"{delay:.1f} min")
        st.caption(result["advisory"])

with st.expander("About this estimate"):
    st.write(
        "This is an educational model trained on passenger survey data. "
        "Departure delay is its most influential input. The dataset lacks "
        "route, weather, schedule, and aircraft-operation details; actual "
        "performance may differ, especially for severe delays."
    )
