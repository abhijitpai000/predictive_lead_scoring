"""Predict on test_set using Trained Model using test_predict"""

import joblib
import pandas as pd

# Local Imports.
from .config import TARGET_COLS
from sklearn.metrics import classification_report, confusion_matrix


def test_model():
    """
    Tests Trained model in test_clean.csv
    Returns
    -------
        classification_report, confusion_matrix (normalized on True)
    """
    test_set = pd.read_csv('datasets/test_clean.csv')

    test_y = test_set[TARGET_COLS]
    test_X = test_set.drop(TARGET_COLS, axis=1)

    lead_scoring_model = joblib.load("models/lead_scoring_model.pkl")
    test_preds = lead_scoring_model.predict(test_X)
    report = classification_report(test_y, test_preds)
    conf_mx = confusion_matrix(test_y, test_preds, normalize="true")
    return report, conf_mx