from src.predict import predict_message


def test_spam_message():                     # Spam test
    result = predict_message(
        "Congratulations! You have won a free prize. Call now to claim."
    )

    assert result["label"] == "spam"


def test_ham_message():                      # Ham test
    result = predict_message(
        "Hey, are we still meeting at 5pm today?"
    )

    assert result["label"] == "ham"


def test_spam_probability_range():           # Test the probability range
    result = predict_message(
        "Congratulations! You have won a free prize. Call now to claim."
    )

    assert 0.0 <= result["spam_probability"] <= 1.0


def test_prediction_matches_label():        #Test prediction
    result = predict_message(
        "Congratulations! You have won a free prize. Call now to claim."
    )

    assert result["prediction"] in (0, 1)   # incase of future bug

    if result["prediction"] == 1:
        assert result["label"] == "spam"
    else:
        assert result["label"] == "ham"