import pandas as pd
import streamlit as st
import joblib

model = joblib.load("rent_model.joblib")

#1 Define model columns

MODEL_COLUMNS = [
    'house_size', 'latitude', 'longitude', 'numBathrooms', 'numBalconies', 'bhk',
    'type_1 BHK Apartment', 'type_1 BHK Independent Floor', 'type_1 BHK Independent House', 'type_1 BHK Villa', 'type_1 RK Studio Apartment',
    'type_2 BHK Apartment', 'type_2 BHK Independent Floor', 'type_2 BHK Independent House', 'type_2 BHK Villa',
    'type_3 BHK Apartment', 'type_3 BHK Independent Floor', 'type_3 BHK Independent House', 'type_3 BHK Villa',
    'type_4 BHK Apartment', 'type_4 BHK Independent Floor', 'type_4 BHK Independent House', 'type_4 BHK Villa',
    'type_5 BHK Apartment', 'type_5 BHK Independent House', 'type_5 BHK Villa',
    'type_6 BHK Apartment', 'type_6 BHK Independent House',
    'loc_Aundh', 'loc_Balewadi', 'loc_Baner', 'loc_Dhanori', 'loc_Dhayari', 'loc_Hadapsar',
    'loc_Hinjewadi', 'loc_Kharadi', 'loc_Kondhwa', 'loc_Kothrud', 'loc_Lohegaon', 'loc_NIBM Annex Mohammadwadi',
    'loc_Other', 'loc_Pimple Saudagar', 'loc_Tathawade', 'loc_Undri', 'loc_Viman Nagar', 'loc_Vishrantwadi',
    'loc_Wadgaon Sheri', 'loc_Wagholi', 'loc_Wakad', 'size_per_bhk'
]

#2 Creat UI

st.title("🏠 Pune Rental Price Predictor")
st.write("Find out the fair market rent for flats across Pune's micro-markets.")

#User inputs

locality = st.selectbox("select Locality" , ["Aundh", "Balewadi", "Baner", "Dhanori", "Dhayari", "Hadapsar", "Hinjewadi", 
    "Kharadi", "Kondhwa", "Kothrud", "Lohegaon", "NIBM Annex Mohammadwadi", "Other", 
    "Pimple Saudagar", "Tathawade", "Undri", "Viman Nagar", "Vishrantwadi", 
    "Wadgaon Sheri", "Wagholi", "Wakad"
])

bhk = st.selectbox("select BHK" , [1,2,3,4,5,6] , index=1)

prop_type = st.selectbox("Select Property Type" , ["Apartment", "Independent Floor", "Independent House", "Villa", "RK Studio Apartment"])

house_size = st.number_input("Area (sq ft)" , min_value=100 , max_value=10000 , value=1000)
num_Bathrooms = st.number_input("Select Number of bathrooms" , min_value=0 , max_value=6 , value=2)
num_Balconies = st.number_input("Select Number of Balconies" , min_value=1 , max_value=5 , value=1)

latitude = st.number_input("Latitude (Pune default approx )" , value=18.5204)
longitude = st.number_input("Longitude (Pune default approx)" , value=73.8567)

#3 Creat button

if st.button("Predict Rent"):
    input_dict = {col: 0 for col in MODEL_COLUMNS} #Creating base directory

    # Fill in numerical inputs
    input_dict['house_size']  = house_size
    input_dict['numBalconies'] = num_Balconies
    input_dict['numBathrooms'] = num_Bathrooms
    input_dict['latitude'] = latitude
    input_dict['longitude'] = longitude
    input_dict['bhk'] = bhk
    input_dict['size_per_bhk'] = house_size / bhk


    #set the locality dummy variables to 1

    selected_loc_col = f"loc_{locality}"
    if selected_loc_col in input_dict:
        input_dict[selected_loc_col]=1

    #set the property type of dummy variable to 1

    selected_type_col = f"type_{bhk} BHK {prop_type}"

    #special handel for 1RK

    if bhk == 1 and prop_type == "RK Studio Apartment":
        selected_type_col = "type_1 RK Studio Apartment"

    if selected_type_col in input_dict:
        input_dict[selected_type_col]=1

    # Convert to DataFrame with the exact column order the model was trained on

    input_data = pd.DataFrame([input_dict] , columns=MODEL_COLUMNS)

    # Generating Prediction
    prediction = model.predict(input_data)[0]

    # Output

    st.success(f"💰 Estimated Monthly Rent: ₹{int(prediction):,} ")
