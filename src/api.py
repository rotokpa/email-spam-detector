from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict_message


app = FastAPI()


class PredictionRequest(BaseModel):
    message: str


@app.get("/")
def home():
    return {"message": "AI Spam Detector API is running"}


@app.post("/predict")
def predict(request: PredictionRequest):
    return predict_message(request.message)