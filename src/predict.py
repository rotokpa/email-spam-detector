import json
import joblib
from pathlib import Path


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

VECTORIZER_PATH = MODELS_DIR / "char_tfidf.joblib"
MODEL_PATH = MODELS_DIR / "logistic_regression.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"

vectorizer = joblib.load(VECTORIZER_PATH)
model = joblib.load(MODEL_PATH)

with open(METADATA_PATH, "r") as file:
    metadata = json.load(file)

threshold = metadata["threshold"]



def predict_message(message):
    message_vector = vectorizer.transform([message])

    spam_probability = model.predict_proba(
        message_vector
    )[0, 1]

    prediction = int(
        spam_probability >= threshold
    )

    if prediction ==  1:
        label = "spam"
    else:
        label = "ham"
        
    return {
        "message": message,
        "spam_probability": float(spam_probability),
        "prediction": prediction,
        "label": label
    }


if __name__ == "__main__":  # dunder 
    result = predict_message(
        "Congratulations! You have won a free prize. Call now to claim."
    )

    print(result)

