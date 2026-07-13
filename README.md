# Health Status Predictor

An ML classification model that predicts health status/risk from lifestyle and biometric data.

## Overview

This project applies a full supervised learning workflow to a health dataset containing biometric measurements (Age, BMI, Blood Pressure, Cholesterol, Glucose Level, etc.) and lifestyle factors (Smoking, Alcohol, Diet, Exercise, Stress Level) to predict a binary health risk target.

## Dataset

- 9,549 records, 23 columns
- No missing values
- Features include: Age, BMI, Blood_Pressure, Cholesterol, Glucose_Level, Heart_Rate, Sleep_Hours, Exercise_Hours, Water_Intake, Stress_Level, Smoking, Alcohol, Diet, MentalHealth, PhysicalActivity, MedicalHistory, Allergies, plus one-hot encoded Diet_Type and Blood_Group columns
- Target: binary classification (health risk indicator)

## Preprocessing

- Null-value check (none found in this dataset)
- Boolean columns (Diet_Type, Blood_Group flags) converted to integer (0/1)
- Feature scaling applied before model training

## Models Compared

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.823 | 0.829 | 0.838 | 0.833 |
| Random Forest (tuned) | 0.929 | 0.923 | 0.943 | 0.933 |
| **Gradient Boosting (GridSearchCV tuned)** | **0.947** | **0.951** | **0.949** | **0.949** |

## Model Selection

The final model is a **Gradient Boosting Classifier**, tuned via `GridSearchCV` (5-fold cross-validation) over `n_estimators`, `learning_rate`, and `max_depth`. Best parameters: `learning_rate=0.1, max_depth=5, n_estimators=200`.

Recall was prioritized as a key metric alongside accuracy and F1, since in a health context, false negatives (missing an at-risk patient) carry a higher real-world cost than false positives.

Model reliability was verified through:
- Train/test accuracy comparison (checked for overfitting)
- 5-fold cross-validation (mean accuracy ~92.5%, std ~0.9%, confirming stable performance across data splits)
- Confusion matrix inspection for both classes

## Project Structure

├── predictor_model.ipynb   # Full notebook: EDA, preprocessing, model training, evaluation
├── novagen_dataset.csv     # Dataset
├── best_model.pkl          # Saved final trained model (Gradient Boosting)
├── requirements.txt        # Dependencies
└── .gitignore

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook predictor_model.ipynb
```

To load the saved model directly without retraining:

```python
import joblib
model = joblib.load('best_model.pkl')
predictions = model.predict(X_new)
```

## Tech Stack

Python, pandas, scikit-learn, matplotlib, joblib

## Author

[Samir B K](https://github.com/Samir-BK)