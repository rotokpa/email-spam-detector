from fastapi.testclient import TestClient
from src.api import app


client = TestClient(app)

def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "AI Spam Detector API is running"
    }

def test_predict_spam():
    response = client.post(
        "/predict",
        json={
            "message": "Congratulations! You have won a free prize. Call now to claim."
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] == 1
    assert data["label"] == "spam"
    assert 0.0 <= data["spam_probability"] <= 1.0


def test_predict_ham():
    response = client.post(
        "/predict",
        json={
            "message": "Hey, are we still meeting at 5pm today?"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] == 0
    assert data["label"] == "ham"
    assert 0.0 <= data["spam_probability"] <= 1.0


def test_predict_missing_message():
    response = client.post(
        "/predict",
        json={
            "text": "Congratulations! You won a prize!"
        }
    )

    assert response.status_code == 422