import os
import sys

import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

from data_preprocessing import load_data, split_data


def test_load_data_returns_matching_features_and_target():
    X, y = load_data()

    assert len(X) == len(y)
    assert X.shape[1] == 30
    assert y.name == "target"


def test_split_data_has_expected_test_size():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    assert len(X_test) == len(y_test)
    assert len(X_train) == len(y_train)
    assert len(X_test) == 114
