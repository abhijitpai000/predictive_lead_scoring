"""Predict on test_set using Trained Model using test_predict"""

import joblib
import pandas as pd

# Local Imports.
from .config import TARGET_COLS
from sklearn.metrics import classification_report, confusion_matrix


def test_model(threshold=0.35):
    """
    Tests Trained model in test_clean.csv

    Parameters
    ----------
        threshold: float, default=0.35.
            Threshold for predicting probability.
    Yields
    ------
        lead_segments based on probability in "../datasets/lead_segments.csv"

    Returns
    -------
        classification_report, confusion_matrix (normalized on True)
    """
    test_set = pd.read_csv('datasets/test_clean.csv')

    test_y = test_set[TARGET_COLS]
    test_X = test_set.drop(TARGET_COLS, axis=1)

    lead_scoring_model = joblib.load("models/lead_scoring_model.pkl")

    test_preds = lead_scoring_model.predict_proba(test_X)[:, 1] > threshold
    _segment_leads(test_X, test_y, lead_scoring_model)

    report = classification_report(test_y, test_preds)
    conf_mx = confusion_matrix(test_y, test_preds, normalize="true")
    return report, conf_mx


def _segment_leads(features, target, model):
    """
    Segments Leads based on Predicted by the Model.

    Parameters
    ----------
        features: test_X
        target: test_y
        model: model.
    """
    scoring = features.copy()
    scoring["ground_truth"] = target.copy()
    scoring["class_1_prob"] = model.predict_proba(features)[:, 1]

    # Segmenting leads based on probability.
    scoring["predicted_probability"] = "low"
    scoring.loc[scoring["class_1_prob"] >= 0.35, "predicted_probability"] = "high"
    scoring.loc[(scoring["class_1_prob"] < 0.35) & (scoring["class_1_prob"] > 0.2), "predicted_probability"] = "medium"
    scoring.loc[scoring["class_1_prob"] <= 0.2, "predicted_probability"] = "low"

    scoring.to_csv("datasets/lead_scoring.csv", index=False)
    return
