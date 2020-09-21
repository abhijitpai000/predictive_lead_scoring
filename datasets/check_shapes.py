"""Sanity Checks Shape of datasets."""

import pandas as pd

if __name__ == '__main__':
    raw = pd.read_csv("raw.csv")

    train_set = pd.read_csv("train_clean.csv")
    train_clean = pd.read_csv("train_clean.csv")

    test_set = pd.read_csv("test_set.csv")
    test_clean = pd.read_csv("test_clean.csv")

    print(f"RAW DATA:"
          f"\nraw: {raw.shape}"
          f"\nTRAIN SET:"
          f"\ntrain_set: {train_set.shape}"
          f"\ntrain_clean: {train_clean.shape}"
          f"\nTRAIN SET:"
          f"\ntest_set: {test_set.shape}"
          f"\ntest_clean: {test_clean.shape}")