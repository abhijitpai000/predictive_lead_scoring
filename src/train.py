from sklearn.model_selection import cross_validate, StratifiedKFold
from lightgbm.sklearn import LGBMClassifier as lgb
import pandas as pd
from pathlib import Path
import joblib

from .config import TARGET_COLS


def train_model():
    """
    Trains Light Gradient Boosting Model. (LightGBM)

    Yields
    ------
        cv_results: 'models/cv_results.csv'
        Trained model: 'models/summer_sales_estimator.pkl'

    Returns
    -------
        dict, cross-validation roc_auc, precision & recall scores.
    """
    train_clean = pd.read_csv("datasets/train_clean.csv")

    features = train_clean.drop(TARGET_COLS, axis=1)
    target = train_clean[TARGET_COLS]

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    lgb_clf = lgb(class_weight='balanced', max_depth=9, num_leaves=9,
                  random_state=0)

    cv_results = cross_validate(lgb_clf, features, target,
                                scoring=['roc_auc', 'precision', 'recall'],
                                cv=skf)
    # Save CV Results.
    cv_path = Path.cwd() / "models/cv_results.csv"
    cv_results = pd.DataFrame(cv_results)
    cv_results.to_csv(cv_path, index=False)

    # Save Summer Sales Estimator.
    lgb_clf.fit(features, target)
    est_path = Path.cwd() / "models/lead_scoring_model.pkl"
    joblib.dump(lgb_clf, est_path)
    return cv_results
