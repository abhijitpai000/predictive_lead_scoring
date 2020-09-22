import pandas as pd
import numpy as np
import pytest


@pytest.fixture
def load_train_cv_results():
    cv_results = pd.read_csv("../models/cv_results.csv")
    roc = np.mean(cv_results['test_roc_auc'])
    recall = np.mean(cv_results['test_recall'])
    return roc, recall


def test_roc_score(load_train_cv_results):
    assert load_train_cv_results[0] > 0.75, "Attained ROC < 0.75"


def test_recall_score(load_train_cv_results):
    assert load_train_cv_results[1] > 0.60, "Attained Recall < 0.60"
