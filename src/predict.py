def predict_message(message, vectorizer, model, threshold):
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


