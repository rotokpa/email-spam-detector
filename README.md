# Model Comparison: Classical ML vs. DistilBERT

## 1. Overview

Two approaches were developed and evaluated for the SMS Spam Detection task:

1. **Classical Machine Learning:** Character-level TF-IDF combined with Logistic Regression.
2. **Transformer-based Deep Learning:** DistilBERT fine-tuned for binary text classification.

Both models were trained and evaluated using the cleaned SMS Spam Collection dataset, containing 5,169 unique messages after duplicate removal.

The dataset was divided into:
- Training: 3,618 messages
- Validation: 775 messages
- Testing: 776 messages

## 2. Model Architectures

### Classical ML: TF-IDF + Logistic Regression

The classical approach uses:
- Character-level TF-IDF vectorization.
- Character n-grams ranging from 3 to 5.
- Logistic Regression as the classification algorithm.
- Class weighting to account for class imbalance.
- A validation-selected classification threshold of approximately 0.571.

This approach converts text into numerical feature vectors before applying Logistic Regression.

### Transformer: DistilBERT

The transformer approach uses:
- Pretrained `distilbert-base-uncased`.
- Fine-tuning for binary classification.
- Maximum sequence length of 128 tokens.
- AdamW optimizer.
- Learning rate of 2e-5.
- Weight decay of 0.01.
- Three training epochs.
- Best checkpoint selected using validation F1 score.

Unlike traditional TF-IDF, DistilBERT uses contextual representations to learn relationships between words and their surrounding text.

## 3. Test Set Performance Comparison

| Metric | Classical ML | DistilBERT |
|---|---:|---:|
| Accuracy | 98.84% | **99.36%** |
| Precision | 97.85% | **100.00%** |
| Recall | 92.86% | **94.90%** |
| F1 Score | 95.29% | **97.38%** |
| False Positives | 2 | **0** |
| False Negatives | 7 | **5** |

## 4. Confusion Matrix Comparison

**Classical ML**

| | Predicted HAM | Predicted SPAM |
|---|---:|---:|
| Actual HAM | 676 | 2 |
| Actual SPAM | 7 | 91 |

**DistilBERT**

| | Predicted HAM | Predicted SPAM |
|---|---:|---:|
| Actual HAM | 678 | 0 |
| Actual SPAM | 5 | 93 |

## 5. Interpretation of Results

DistilBERT achieved better results across all reported test metrics.

Compared with the classical model, DistilBERT:

- Improved accuracy from 98.84% to 99.36%.
- Improved spam precision from 97.85% to 100%.
- Improved spam recall from 92.86% to 94.90%.
- Improved F1 score from 95.29% to 97.38%.
- Reduced false positives from 2 to 0.
- Reduced false negatives from 7 to 5.

These results indicate that DistilBERT performed better on this test set.

However, the classical ML model remains valuable because it is computationally lighter, easier to train, and less resource-intensive.

## 6. Error Analysis

Error analysis was conducted on both validation and test predictions.

Validation results:
- 9 misclassified messages.
- 7 false negatives.
- 2 false positives.

Test results:
- 5 misclassified messages.
- 5 false negatives.
- 0 false positives.

The investigation revealed that some spam messages resemble legitimate conversations, notifications, promotional announcements, and ordinary questions.

Some mistakes were also made with very high model-assigned probabilities, suggesting that the model can be overconfident in incorrect predictions.

No conclusions about probability calibration were established from this small error sample.

## 7. Final Conclusion

Both approaches successfully demonstrated that machine learning can effectively classify SMS messages as spam or ham.

DistilBERT achieved the strongest test performance and will serve as the primary model for the next stage of the project.

The classical TF-IDF and Logistic Regression model will remain as a baseline for comparison.

The project will now progress from model development and evaluation into application engineering, API development, and deployment.

**Next phase:**

1. Build a prediction API using FastAPI.
2. Load the saved DistilBERT model and tokenizer.
3. Implement an endpoint for classifying incoming messages.
4. Test the API using different SMS examples.
5. Build a simple frontend.
6. Containerize the application using Docker.
7. Deploy the application and publish a live demo.