import os
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Pune Rental Predictor", page_icon="🏠")

st.title("🏠 Pune Rental Price Predictor")
st.write("Find out the fair market rent for flats across Pune's micro-markets.")

@st.cache_data
def load_locality_coords():
    try:
        raw = pd.read_csv("Indian_housing_Pune_data.csv")
        raw = raw[(raw["latitude"].between(18.3, 18.8)) & (raw["longitude"].between(73.6, 74.1))]
        return raw.groupby("location")[["latitude", "longitude"]].mean(), raw["latitude"].mean(), raw["longitude"].mean()
    except FileNotFoundError:
        # Fallback if CSV is not in root folder
        return None, 18.5204, 73.8567

locality_coords, mean_lat, mean_lon = load_locality_coords()

# User Selection Inputs
locality = st.selectbox("Select Locality", [
    "Aundh", "Balewadi", "Baner", "Dhanori", "Dhayari", "Hadapsar", "Hinjewadi", 
    "Kharadi", "Kondhwa", "Kothrud", "Lohegaon", "NIBM Annex Mohammadwadi", "Other", 
    "Pimple Saudagar", "Tathawade", "Undri", "Viman Nagar", "Vishrantwadi", 
    "Wadgaon Sheri", "Wagholi", "Wakad"
])

bhk = st.selectbox("Select BHK", [1, 2, 3, 4, 5, 6], index=1)
prop_type = st.selectbox("Select Property Type", ["Apartment", "Independent Floor", "Independent House", "Villa", "RK Studio Apartment"])

house_size = st.number_input("Area (sq ft)", min_value=100, max_value=10000, value=1000)
num_bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=6, value=2)
num_balconies = st.number_input("Number of Balconies", min_value=0, max_value=5, value=1)

# Dynamically pull mean coordinates for selected locality
if locality_coords is not None and locality in locality_coords.index:
    default_lat = float(locality_coords.loc[locality, "latitude"])
    default_lon = float(locality_coords.loc[locality, "longitude"])
else:
    default_lat, default_lon = mean_lat, mean_lon

latitude = st.number_input("Latitude", value=default_lat, format="%.4f")
longitude = st.number_input("Longitude", value=default_lon, format="%.4f")

# Make Prediction
if st.button("Predict Rent", type="primary"):
    payload = {
        "locality": locality,
        "bhk": bhk,
        "prop_type": prop_type,
        "house_size": house_size,
        "num_bathrooms": num_bathrooms,
        "num_balconies": num_balconies,
        "latitude": latitude,
        "longitude": longitude
    }

    try:
        BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/predict")
        response = requests.post(BACKEND_URL, json=payload, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            rent = result.get("predicted_rent", 0)
            st.success(f"💰 Estimated Monthly Rent: **₹ {rent:,.2f}**")
        else:
            st.error(f"Server error ({response.status_code}): {response.text}")           
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to FastAPI server. Make sure Uvicorn is running on port 8000!")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")