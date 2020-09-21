"""Sanity Check: Files Generated by make_dataset() module"""

import pandas as pd
import pytest


# Raw Data.
def test_raw_data():
    """raw data test"""
    raw = pd.read_csv("../datasets/raw.csv")
    assert raw.shape == (41188, 21), "Raw data failed"


@pytest.fixture
def train():
    train_set = pd.read_csv("../datasets/train_clean.csv")
    train_clean = pd.read_csv("../datasets/train_clean.csv")
    return train_set, train_clean


def test_train_set(train):
    """test set shape"""
    assert train[0].shape == (30891, 21), "Train set Failed"


def test_train_clean(train):
    """test clean shape"""
    assert train[1].shape == (30891, 21), "Train Clean Failed"


@pytest.fixture
def test():
    test_set = pd.read_csv("../datasets/test_set.csv")
    test_clean = pd.read_csv("../datasets/test_clean.csv")
    return test_set, test_clean


def test_test_set(test):
    """test set shape"""
    assert test[0].shape == (10297, 21), "Test set Failed"


def test_test_clean(test):
    """test clean shape"""
    assert test[0].shape == (10297, 21), "Test Clean Failed"