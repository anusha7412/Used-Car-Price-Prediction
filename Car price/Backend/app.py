# import streamlit as st
# import pandas as pd
# import joblib
# from sklearn.preprocessing import LabelEncoder

# # 1. Page Configuration
# st.set_page_config(page_title="Car Price Predictor", page_icon="🚗")

# # 2. Load the Model
# @st.cache_resource
# def load_model():
#     model = joblib.load('cardekho_dataset.csv')
#     return model

# model = load_model()

# # 3. Define the UI Layout
# st.title("🚗 Car Price Prediction")
# st.markdown("Enter the details of the car below to predict its selling price using Random Forest.")

# # Sidebar for inputs
# st.sidebar.header("Car Details")

# # Input Fields
# car_name = st.sidebar.text_input("Car Name")
# year = st.sidebar.number_input("Year of Manufacture", min_value=1990, max_value=2024, value=2020)
# present_price = st.sidebar.number_input("Present Price (Lakhs)", min_value=0.0, value=5.0)
# kms_driven = st.sidebar.number_input("Kms Driven", min_value=0, value=50000)

# # Categorical Inputs
# fuel_type = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
# seller_type = st.sidebar.selectbox("Seller Type", ["Dealer", "Individual"])
# transmission = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])
# owner = st.sidebar.selectbox("Owner", ["First Owner", "Second Owner", "Third Owner"])

# # 4. Prediction Logic
# if st.sidebar.button("Predict Price"):
#     try:
#         # Prepare data for prediction
#         # Note: We must encode the inputs exactly as we did in the training script
#         fuel_map = {"Petrol": 0, "Diesel": 1, "CNG": 2} # Adjust based on your training encoding
#         seller_map = {"Dealer": 0, "Individual": 1}
#         trans_map = {"Manual": 0, "Automatic": 1}
#         owner_map = {"First Owner": 0, "Second Owner": 1, "Third Owner": 2}

#         # Create a dataframe for the single input
#         input_data = pd.DataFrame({
#             'year': [year],
#             'present_price': [present_price],
#             'kms_driven': [kms_driven],
#             'fuel_type': [fuel_map[fuel_type]],
#             'seller_type': [seller_map[seller_type]],
#             'transmission': [trans_map[transmission]],
#             'owner': [owner_map[owner]]
#         })

#         # Make Prediction
#         predicted_price = model.predict(input_data)[0]

#         # Display Result
#         st.success(f"Predicted Selling Price: **₹ {predicted_price:.2f} Lakhs**")
        
#         # Optional: Show a breakdown
#         st.info(f"Based on the car name: {car_name}, manufactured in {year}.")

#     except Exception as e:
#         st.error(f"An error occurred: {e}")

# # 5. Footer
# st.markdown("---")
# st.caption("Built with Streamlit & Random Forest ML")
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel, Field
# import joblib
# import pandas as pd
# import numpy as np
# import os

# app = FastAPI(title="Car Price Predictor API", version="1.0.0")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Change to your frontend URL in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# MODEL_PATH = os.getenv("MODEL_PATH", "car_price_model.pkl")
# model = None

# @app.on_event("startup")
# def load_model():
#     global model
#     try:
#         model = joblib.load(MODEL_PATH)
#         print(f"Model loaded from {MODEL_PATH}")
#     except FileNotFoundError:
#         print(f"WARNING: Model file not found at {MODEL_PATH}. /predict will fail until it's available.")

# class CarInput(BaseModel):
#     vehicle_age: int = Field(..., ge=0, le=50, description="Age of vehicle in years")
#     km_driven: int = Field(..., ge=0, description="Total kilometers driven")
#     seller_type: int = Field(..., ge=0, le=2, description="0=Dealer, 1=Individual, 2=Trustmark Dealer")
#     fuel_type: int = Field(..., ge=0, le=4, description="0=CNG, 1=Diesel, 2=Electric, 3=LPG, 4=Petrol")
#     transmission_type: int = Field(..., ge=0, le=1, description="0=Automatic, 1=Manual")
#     mileage: float = Field(..., ge=0, description="Mileage in km/l")
#     engine: float = Field(..., ge=0, description="Engine displacement in cc")
#     max_power: float = Field(..., ge=0, description="Max power in bhp")
#     seats: int = Field(..., ge=2, le=14, description="Number of seats")

# class PredictionResponse(BaseModel):
#     predicted_price: float
#     predicted_price_formatted: str
#     inputs: dict

# @app.get("/")
# def root():
#     return {"status": "ok", "message": "Car Price Predictor API is running"}

# @app.get("/health")
# def health():
#     return {"status": "ok", "model_loaded": model is not None}

# @app.post("/predict", response_model=PredictionResponse)
# def predict(car: CarInput):
#     if model is None:
#         raise HTTPException(status_code=503, detail="Model not loaded. Check that car_price_model.pkl exists.")

#     input_df = pd.DataFrame([car.dict()])
#     try:
#         price = model.predict(input_df)[0]
#         price = float(np.clip(price, 100000, 5000000))
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

#     return PredictionResponse(
#         predicted_price=round(price, 2),
#         predicted_price_formatted=f"₹{price:,.0f}",
#         inputs=car.dict()
#     )

# @app.get("/encodings")
# def get_encodings():
#     """Return label encoding maps so frontend can show human-readable labels."""
#     return {
#         "seller_type": {"0": "CNG", "1": "Dealer", "2": "Individual", "3": "Trustmark Dealer"},
#         "fuel_type": {"0": "CNG", "1": "Diesel", "2": "Electric", "3": "LPG", "4": "Petrol"},
#         "transmission_type": {"0": "Automatic", "1": "Manual"}
#     }
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
try:
    model = joblib.load("car_price_model.pkl")
    print("Model loaded successfully!")
except:
    model = None
    print("Model not found!")

# Input schema
class CarInput(BaseModel):
    vehicle_age: int
    km_driven: int
    seller_type: int
    fuel_type: int
    transmission_type: int
    mileage: float
    engine: float
    max_power: float
    seats: int

# Home route
@app.get("/")
def home():
    return {"message": "Car Price Predictor API Running"}

# Prediction route
@app.post("/predict")
def predict(data: CarInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        df = pd.DataFrame([data.dict()])
        prediction = model.predict(df)[0]

        # Limit price range
        prediction = float(np.clip(prediction, 100000, 5000000))

        return {
            "predicted_price": round(prediction, 2),
            "predicted_price_formatted": f"₹{prediction:,.0f}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))