# Local Package Imports.
from src.preprocess import make_dataset
from src.folds import generate_folds
from src.train import train_model
from src.predict import test_model

import numpy as np

if __name__ == '__main__':
    # Pre-process dataset.
    train_clean, test_clean = make_dataset(raw_file_name="raw.csv")

    # Generate Folds for train_clean
    train_folds = generate_folds(file_name="train_clean.csv",
                                 fold_type="skfold",
                                 n_splits=5,
                                 save_file_name="train_folds.csv")
    # Train Model
    cv_results = train_model()
    print(f"\nTRAIN SCORES:"
          f"\nROC_AUC: {np.mean(cv_results['test_roc_auc'])}"
          f"\nPrecision: {np.mean(cv_results['test_roc_auc'])}"
          f"\nRecall: {np.mean(cv_results['test_recall'])}")

    # Test Model.
    report, conf_mx = test_model()
    print(f"\nTEST RESULTS:"
          f"\n {report}"
          f"\nTrue Positive Rate: {round(conf_mx[1][1]*100, 3)}"
          f"\nFalse Positive Rate: {round(conf_mx[0][1]*100, 3)}"
          f"\nFalse Negative Rate: {round(conf_mx[1][0]*100, 3)}")
