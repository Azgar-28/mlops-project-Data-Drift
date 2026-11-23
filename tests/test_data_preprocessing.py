import os
import sys

import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

from data_preprocessing import load_data, split_data
import drift_detection
from drift_detection import match_feature_columns


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


def test_drift_matching_accepts_different_numeric_feature_headers():
    reference = pd.DataFrame({
        "known_a": [1.0, 2.0],
        "known_b": [3.0, 4.0],
    })
    incoming = pd.DataFrame({
        "sensor_x": [1.5, 2.5],
        "sensor_y": [3.5, 4.5],
    })

    assert match_feature_columns(reference, incoming) == [
        ("known_a", "sensor_x"),
        ("known_b", "sensor_y"),
    ]


def test_drift_matching_prefers_names_and_ignores_target():
    reference = pd.DataFrame({
        "known_a": [1, 2],
        "known_b": ["low", "high"],
    })
    incoming = pd.DataFrame({
        "known_b": ["high", "low"],
        "target": [0, 1],
        "sensor_x": [2, 3],
    })

    assert match_feature_columns(reference, incoming) == [
        ("known_b", "known_b"),
        ("known_a", "sensor_x"),
    ]


def test_drift_detection_accepts_renamed_subset_features(tmp_path, monkeypatch):
    features, target = load_data()
    reference = features.copy()
    reference["target"] = target
    incoming = features.iloc[:200, :3].copy()
    incoming.iloc[:, 0] = incoming.iloc[:, 0] * 1.5
    incoming.columns = ["sensor_1", "sensor_2", "sensor_3"]

    reference_path = tmp_path / "training.csv"
    incoming_path = tmp_path / "incoming.csv"
    report_path = tmp_path / "report.html"
    reference.to_csv(reference_path, index=False)
    incoming.to_csv(incoming_path, index=False)

    monkeypatch.setattr(drift_detection, "TRAINING_DATA", reference_path)
    monkeypatch.setattr(drift_detection, "NEW_DATA", incoming_path)
    monkeypatch.setattr(drift_detection, "DRIFT_REPORT", report_path)

    result = drift_detection.detect_drift()

    assert isinstance(result, bool)
    assert report_path.is_file()
