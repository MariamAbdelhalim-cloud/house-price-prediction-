"""
5_api_fastapi.py
-------------------
Stage 5 of the pipeline: Deployment (FastAPI REST API).

Run:
    uvicorn api_fastapi:app --reload

Then open http://127.0.0.1:8000/docs for interactive Swagger UI,
or POST to http://127.0.0.1:8000/predict with a JSON body like:

{
  "LotArea": 8000,
  "OverallQual": 7,
  "YearBuilt": 2005,
  "TotalBsmtSF": 900,
  "GrLivArea": 2000,
  "BedroomAbvGr": 3,
  "FullBath": 2,
  "GarageCars": 2,
  "Neighborhood": "Downtown"
}
"""

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="House Price Prediction API", version="1.0")

model = joblib.load("models/house_price_model.joblib")


class HouseFeatures(BaseModel):
    LotArea: int
    OverallQual: int
    YearBuilt: int
    TotalBsmtSF: int
    GrLivArea: int
    BedroomAbvGr: int
    FullBath: int
    GarageCars: int
    Neighborhood: str

    class Config:
        json_schema_extra = {
            "example": {
                "LotArea": 8000,
                "OverallQual": 7,
                "YearBuilt": 2005,
                "TotalBsmtSF": 900,
                "GrLivArea": 2000,
                "BedroomAbvGr": 3,
                "FullBath": 2,
                "GarageCars": 2,
                "Neighborhood": "Downtown",
            }
        }


@app.get("/")
def root():
    return {"message": "House Price Prediction API is running. See /docs for usage."}


@app.post("/predict")
def predict(features: HouseFeatures):
    input_df = pd.DataFrame([features.dict()])
    prediction = model.predict(input_df)[0]
    return {"predicted_price": round(float(prediction), 2)}
