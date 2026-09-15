from fastapi import FastAPI
from src.predict import predict_message

app = FastAPI()


@app.get("/")
def home():
    return {"message": "AI Spam Detector API is running"}