from typing import Literal
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
from src.predict import predict_message


app = FastAPI()


class PredictionRequest(BaseModel):
    message: str = Field(min_length=1)

    @field_validator("message")  # @ - decorator
    @classmethod
    def validate_message(cls, value):
        if not value.strip():
            raise ValueError("Message cannot be empty or whitespace only.")
        return value.strip()


class PredictionResponse(BaseModel):
    message: str
    spam_probability: float = Field(ge=0.0, le=1.0) #greaterthan, lessthan
    prediction: Literal[0, 1]
    label: Literal["ham", "spam"]

@app.get("/")
def home():
    return {"message": "AI Spam Detector API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    return predict_message(request.message)
