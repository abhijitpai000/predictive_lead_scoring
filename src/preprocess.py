"""
Steps
-- Drop JobSatisfaction
- Split Data into Train and Test set
- Categorical Encoding
- Create Train Clean and Test Clean
"""

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

# Path and Serializing models.
from pathlib import Path
import joblib

# Local Imports
from .config import TARGET_COLS, CAT_COLS, NUM_COLS, DROP_COLS


# STEP 1
def _split_data(dataframe, shuffle=True):
    """
    Drops "Duration" columns and Splits data into train and test set (75:25 ratio).

    Parameters
    ----------
        dataframe: pandas dataframe.
        shuffle: bool, default=True.
            Shuffles data while splitting.
    Yields
    ------
        train_set: 'datasets/train_set'
        test_set: 'datasets/test_set'
    """
    # Drop Duration.
    dataframe.drop(DROP_COLS, axis=1, inplace=True)

    # Train_set, test_set split.
    train, test = train_test_split(dataframe,
                                   shuffle=shuffle,
                                   random_state=0)

    # Saving File.
    train_path = Path.cwd() / "datasets/train_set.csv"
    test_path = Path.cwd() / "datasets/test_set.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    return


def _train_clean():
    """
    Ordinal Encoding for Train Categorical Columns.
    NOTE:
    - Fits on entire dataset categorical features (to avoid 'unseen data' error).
    - Transforms train_set categorical only.
    Yields
    ------
        train_clean: 'datasets/trian_clean.csv'
        Ordinal Encoder: 'models/ord_encoder.pkl'
    Returns
    -------
        train_clean.
    """
    # Fit Categorical with Ordinal Encoder.
    full_data = pd.read_csv("datasets/raw.csv")

    full_cat_features = full_data[CAT_COLS]

    ord_encoder = OrdinalEncoder()
    ord_encoder.fit(full_cat_features)
    pkl_path = Path.cwd() / "models/ord_encoder.pkl"
    joblib.dump(ord_encoder, pkl_path)  # Saving ordinal encoder.

    # Transform Train set.
    train_set = pd.read_csv('datasets/train_set.csv')

    cat_data = train_set[CAT_COLS]
    num_data = train_set[NUM_COLS]

    # Fixing Target.
    target = train_set[TARGET_COLS]
    target = target.apply(
        lambda x: 1 if x == "yes" else 0
    )

    # Ordinal Encoding.
    cat_encoded_data = pd.DataFrame(ord_encoder.transform(cat_data),
                                    index=cat_data.index,
                                    columns=cat_data.columns)

    train_clean = pd.concat([cat_encoded_data, num_data, target], axis=1)
    clean_path = Path.cwd() / "datasets/train_clean.csv"
    train_clean.to_csv(clean_path, index=False)
    return train_clean


def _test_clean():
    """
    Ordinal Encoding for Test Categorical Columns.
    Yields
    ------
        test_clean: 'datasets/test_clean.csv'
    Returns
    -------
        test_clean.
    """
    test_set = pd.read_csv('datasets/test_set.csv')

    cat_data = test_set[CAT_COLS]
    num_data = test_set[NUM_COLS]

    # Fixing Target.
    target = test_set[TARGET_COLS]
    target = target.apply(
        lambda x: 1 if x == "yes" else 0
    )

    ord_encoder = joblib.load("models/ord_encoder.pkl")

    # Ordinal Encoding.
    cat_encoded_data = pd.DataFrame(ord_encoder.transform(cat_data),
                                    index=cat_data.index,
                                    columns=cat_data.columns)

    test_clean = pd.concat([cat_encoded_data, num_data, target], axis=1)
    clean_path = Path.cwd() / "datasets/test_clean.csv"
    test_clean.to_csv(clean_path, index=False)
    return test_clean


def make_dataset(raw_file_name):
    """
    Pre-Processes data.
    Parameters
    ----------
        raw_file_name: str, with .csv.
    Yields
    ------
        dropped_data: 'datasets/dropped_data.csv'
        selected_data: 'datasets/selected_data.csv'
        train_set: 'datasets/train_set.csv'
        train_clean: 'datasets/train_clean.csv'
        test_set: 'datasets/test_set.csv'
    Returns
    -------
        train_clean, test_clean.
    """

    # Creating Splits.
    raw_df = pd.read_csv(f"datasets/{raw_file_name}.csv", delimiter=";")

    _split_data(raw_df, shuffle=True)

    # Pre-Processing Train set and test set.
    train_clean = _train_clean()
    test_clean = _test_clean()

    return train_clean, test_clean
