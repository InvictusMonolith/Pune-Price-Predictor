import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal

app = FastAPI()

model = joblib.load("rent_model.joblib")

MODEL_COLUMNS = [
    'house_size', 'latitude', 'longitude', 'numBathrooms', 'numBalconies', 'bhk',
    'type_1 BHK Apartment', 'type_1 BHK Independent Floor', 'type_1 BHK Independent House', 'type_1 BHK Villa', 'type_1 RK Studio Apartment',
    'type_2 BHK Apartment', 'type_2 BHK Independent Floor', 'type_2 BHK Independent House', 'type_2 BHK Villa',
    'type_3 BHK Apartment', 'type_3 BHK Independent Floor', 'type_3 BHK Independent House', 'type_3 BHK Villa',
    'type_4 BHK Apartment', 'type_4 BHK Independent Floor', 'type_4 BHK Independent House', 'type_4 BHK Villa',
    'type_5 BHK Apartment', 'type_5 BHK Independent House', 'type_5 BHK Villa',
    'type_6 BHK Apartment', 'type_6 BHK Independent House', 'loc_Aundh', 'loc_Balewadi', 'loc_Baner', 'loc_Dhanori', 'loc_Dhayari', 'loc_Hadapsar',
    'loc_Hinjewadi', 'loc_Kharadi', 'loc_Kondhwa', 'loc_Kothrud', 'loc_Lohegaon', 'loc_NIBM Annex Mohammadwadi',
    'loc_Other', 'loc_Pimple Saudagar', 'loc_Tathawade', 'loc_Undri', 'loc_Viman Nagar', 'loc_Vishrantwadi',
    'loc_Wadgaon Sheri', 'loc_Wagholi', 'loc_Wakad', 'size_per_bhk'
]

class RentalRequest(BaseModel):
    locality: Literal["Aundh", "Balewadi", "Baner", "Dhanori", "Dhayari", "Hadapsar", "Hinjewadi", 
                      "Kharadi", "Kondhwa", "Kothrud", "Lohegaon", "NIBM Annex Mohammadwadi", 
                      "Other", "Pimple Saudagar", "Tathawade", "Undri", "Viman Nagar", 
                      "Vishrantwadi", "Wadgaon Sheri", "Wagholi", "Wakad"]
    bhk: int = Field(..., gt=0, description="Number of BHKs (must be greater than 0)")
    prop_type: Literal["Apartment", "Independent Floor", "Independent House", "Villa", "RK Studio Apartment"]
    house_size: float
    num_bathrooms: int
    num_balconies: int
    latitude: float
    longitude: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the Pune Rental Price Predictor API!"}

@app.post("/predict")
def predict_rent(request: RentalRequest):
    try:
        # Initialize dictionary with zeros for all model columns
        input_dict = {col: 0.0 for col in MODEL_COLUMNS}

        # Populate continuous and discrete fields
        input_dict['house_size'] = request.house_size
        input_dict['size_per_bhk'] = request.house_size / request.bhk
        input_dict['numBalconies'] = float(request.num_balconies)
        input_dict['numBathrooms'] = float(request.num_bathrooms)
        input_dict['latitude'] = request.latitude
        input_dict['longitude'] = request.longitude
        input_dict['bhk'] = float(request.bhk)

        # Set locality dummy variable
        selected_loc_col = f"loc_{request.locality}"
        if selected_loc_col in input_dict:
            input_dict[selected_loc_col] = 1.0

        # Set property type dummy variable
        if request.bhk == 1 and request.prop_type == "RK Studio Apartment":
            selected_type_col = "type_1 RK Studio Apartment"
        else:
            selected_type_col = f"type_{request.bhk} BHK {request.prop_type}".strip()

        if selected_type_col in input_dict:
            input_dict[selected_type_col] = 1.0

        input_data = pd.DataFrame([input_dict], columns=MODEL_COLUMNS)

        # Generate prediction
        prediction = float(model.predict(input_data)[0])

        return {"predicted_rent": round(prediction, 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))