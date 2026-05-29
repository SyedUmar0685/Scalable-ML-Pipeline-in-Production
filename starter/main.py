import os
import pickle

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from starter.ml.data import process_data
from starter.ml.model import inference

# Load model artifacts.
model_dir = os.path.join(os.path.dirname(__file__), "model")
with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(model_dir, "encoder.pkl"), "rb") as f:
    encoder = pickle.load(f)
with open(os.path.join(model_dir, "lb.pkl"), "rb") as f:
    lb = pickle.load(f)

app = FastAPI()

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


class CensusData(BaseModel):
    age: int = Field(...)
    workclass: str = Field(...)
    fnlgt: int = Field(...)
    education: str = Field(...)
    education_num: int = Field(..., alias="education-num")
    marital_status: str = Field(..., alias="marital-status")
    occupation: str = Field(...)
    relationship: str = Field(...)
    race: str = Field(...)
    sex: str = Field(...)
    capital_gain: int = Field(..., alias="capital-gain")
    capital_loss: int = Field(..., alias="capital-loss")
    hours_per_week: int = Field(..., alias="hours-per-week")
    native_country: str = Field(..., alias="native-country")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "examples": [
                {
                    "age": 39,
                    "workclass": "State-gov",
                    "fnlgt": 77516,
                    "education": "Bachelors",
                    "education-num": 13,
                    "marital-status": "Never-married",
                    "occupation": "Adm-clerical",
                    "relationship": "Not-in-family",
                    "race": "White",
                    "sex": "Male",
                    "capital-gain": 2174,
                    "capital-loss": 0,
                    "hours-per-week": 40,
                    "native-country": "United-States",
                }
            ]
        },
    }


@app.get("/")
async def root():
    return {"message": "Welcome to the Census Income Prediction API!"}


@app.post("/predict")
async def predict(data: CensusData):
    # Convert input to DataFrame with hyphenated column names.
    input_data = pd.DataFrame([{
        "age": data.age,
        "workclass": data.workclass,
        "fnlgt": data.fnlgt,
        "education": data.education,
        "education-num": data.education_num,
        "marital-status": data.marital_status,
        "occupation": data.occupation,
        "relationship": data.relationship,
        "race": data.race,
        "sex": data.sex,
        "capital-gain": data.capital_gain,
        "capital-loss": data.capital_loss,
        "hours-per-week": data.hours_per_week,
        "native-country": data.native_country,
    }])

    X, _, _, _ = process_data(
        input_data, categorical_features=cat_features, training=False,
        encoder=encoder, lb=lb
    )
    pred = inference(model, X)
    prediction = lb.inverse_transform(pred)[0]
    return {"prediction": prediction}
