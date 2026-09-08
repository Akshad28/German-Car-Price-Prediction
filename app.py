import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")

# Load the trained model/pipeline
@st.cache_resource
def load_model():
    # Replace 'car_price_model.pkl' with your actual exported model/pipeline file
    return joblib.load("car_price_model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Title and header
st.title("🚗 Car Price Predictor")
st.markdown("Enter the vehicle details below to estimate its market value.")

st.divider()

# Input layout
col1, col2 = st.columns(2)

with col1:
    company = st.selectbox("Brand", ["Audi", "BMW", "Volkswagen", "Mercedes"])
    
    # Example model mapping based on selected brand
    model_options = {
        "Audi": ["A1", "A3", "A4", "A5", "A6", "Q3", "Q5", "Q7", "TT"],
        "BMW": ["1 Series", "2 Series", "3 Series", "4 Series", "5 Series", "X1", "X3", "X5"],
        "Volkswagen": ["Golf", "Polo", "Passat", "Tiguan", "T-Roc", "Up"],
        "Mercedes": ["A Class", "B Class", "C Class", "E Class", "GLA Class", "GLC Class", "S Class"]
    }
    car_model = st.selectbox("Model", model_options.get(company, []))
    
    year = st.number_input("Year of Manufacture", min_value=1995, max_value=2024, value=2018, step=1)
    transmission = st.selectbox("Transmission", ["Manual", "Automatic", "Semi-Auto"])

with col2:
    mileage = st.number_input("Mileage (miles)", min_value=0, max_value=300000, value=35000, step=500)
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid", "Electric"])
    engine_size = st.number_input("Engine Size (L)", min_value=0.5, max_value=6.5, value=2.0, step=0.1)
    mpg = st.number_input("Fuel Economy (MPG)", min_value=10.0, max_value=100.0, value=50.0, step=0.5)

# Feature engineering matching training pipeline
CURRENT_YEAR = 2026
car_age = max(0, CURRENT_YEAR - year)
mileage_per_year = mileage / (car_age if car_age > 0 else 1)

st.divider()

# Prediction trigger
if st.button("Predict Price", use_container_width=True):
    # Construct input dataframe matching exact training columns
        # Build a one-row frame with all expected feature columns, matching training exactly
    expected_cols = model.feature_names_in_

    input_data = pd.DataFrame([{col: 0 for col in expected_cols}])

    for col, val in [("tax", 145), ("mpg", mpg), ("engineSize", engine_size),
                      ("Car_Age", car_age), ("mileage_per_year", mileage_per_year)]:
        if col in input_data.columns:
            input_data[col] = val

    for flag in [f"company_{company}", f"transmission_{transmission}", f"fuelType_{fuel_type}"]:
        if flag in input_data.columns:
            input_data[flag] = 1

    input_data = input_data[expected_cols]
    
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"### Estimated Price: **€{prediction:,.2f}**")
    except Exception as err:
        st.error(f"Prediction failed: {err}")
        