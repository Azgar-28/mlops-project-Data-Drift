import os
import sys
import pandas as pd
from pathlib import Path

from evidently import Report
from evidently.presets import DataDriftPreset

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from config import DRIFT_REPORT, NEW_DATA, TRAINING_DATA


def _feature_kind(series):
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(series)
    ):
        return "categorical"
    return "unsupported"


def match_feature_columns(reference_features, incoming_features):
    """Match exact feature names first, then compatible columns by order."""
    incoming_features = incoming_features.drop(columns=["target"], errors="ignore")
    reference_columns = list(reference_features.columns)
    incoming_columns = list(incoming_features.columns)
    pairs = []

    for reference_column in reference_columns:
        if reference_column not in incoming_columns:
            continue
        reference_kind = _feature_kind(reference_features[reference_column])
        incoming_kind = _feature_kind(incoming_features[reference_column])
        if reference_kind == incoming_kind and reference_kind != "unsupported":
            pairs.append((reference_column, reference_column))

    used_reference = {reference for reference, _ in pairs}
    used_incoming = {incoming for _, incoming in pairs}

    unmatched_reference = [
        column for column in reference_columns
        if column not in used_reference
        and _feature_kind(reference_features[column]) != "unsupported"
    ]
    unmatched_incoming = [
        column for column in incoming_columns
        if column not in used_incoming
        and _feature_kind(incoming_features[column]) != "unsupported"
    ]

    for reference_column in unmatched_reference:
        reference_kind = _feature_kind(reference_features[reference_column])
        match_index = next(
            (
                index for index, incoming_column in enumerate(unmatched_incoming)
                if _feature_kind(incoming_features[incoming_column]) == reference_kind
            ),
            None,
        )
        if match_index is None:
            continue
        incoming_column = unmatched_incoming.pop(match_index)
        pairs.append((reference_column, incoming_column))

    return pairs


def detect_drift():
    training_path = TRAINING_DATA
    new_data_path = NEW_DATA
    report_path = DRIFT_REPORT

    # Check whether input files exist
    if not os.path.exists(training_path):
        raise FileNotFoundError(
            f"Reference dataset not found: {training_path}"
        )

    if not os.path.exists(new_data_path):
        raise FileNotFoundError(
            f"Incoming dataset not found: {new_data_path}"
        )

    # Load datasets
    reference_data = pd.read_csv(training_path)
    new_data = pd.read_csv(new_data_path)

    print("Reference data shape:", reference_data.shape)
    print("New data shape:", new_data.shape)

    # Validate datasets
    if reference_data.empty:
        raise ValueError("Reference dataset is empty.")

    if new_data.empty:
        raise ValueError("Incoming dataset is empty.")

    if reference_data.columns.duplicated().any():
        raise ValueError(
            "Reference dataset contains duplicate column names."
        )

    if new_data.columns.duplicated().any():
        raise ValueError(
            "Incoming dataset contains duplicate column names."
        )

    # Remove target column from reference features
    reference_features = reference_data.drop(
        columns=["target"],
        errors="ignore"
    )

    feature_pairs = match_feature_columns(reference_features, new_data)
    if not feature_pairs:
        raise ValueError(
            "No compatible feature columns found. Upload at least one "
            "numeric or categorical feature with usable values."
        )

    reference_columns = [reference for reference, _ in feature_pairs]
    incoming_columns = [incoming for _, incoming in feature_pairs]
    feature_names = [f"feature_{index + 1}" for index in range(len(feature_pairs))]

    reference_features = reference_features[reference_columns].copy()
    incoming_features = new_data[incoming_columns].copy()
    reference_features.columns = feature_names
    incoming_features.columns = feature_names

    print("\n================================")
    print("       DATASET INFORMATION")
    print("================================")

    print("Reference rows:", len(reference_features))
    print("Incoming rows:", len(incoming_features))
    print("Features compared:", len(feature_pairs))

    ignored_columns = [
        col for col in new_data.columns
        if col not in incoming_columns and col != "target"
    ]

    print("Ignored incoming columns:", ignored_columns)

    print("\nFeature mapping used (reference -> uploaded):")
    for reference_column, incoming_column in feature_pairs:
        print(f"- {reference_column} -> {incoming_column}")
    if any(reference != incoming for reference, incoming in feature_pairs):
        print(
            "NOTE: Unmatched headers were paired by feature order and "
            "compatible data type."
        )

    # Check columns with no usable values
    unusable_columns = [
        col for col in feature_names
        if (
            reference_features[col].notna().sum() == 0
            or incoming_features[col].notna().sum() == 0
        )
    ]

    if unusable_columns:
        raise ValueError(
            "These shared columns contain no usable values: "
            + ", ".join(unusable_columns)
        )

    print("\nFeatures used for drift detection:")

    for col in feature_names:
        print("-", col)

    # Generate Evidently drift report
    report = Report([
        DataDriftPreset(drift_share=0.01)
    ])

    # Current data first, reference data second
    result = report.run(
        incoming_features,
        reference_features
    )

    # Create report directory if necessary
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # Save HTML report
    result.save_html(str(report_path))

    # Extract drift information
    result_dict = result.dict()
    drifted_columns = 0
    drift_share = 0.0

    for metric in result_dict.get("metrics", []):
        metric_value = metric.get("value")
        if isinstance(metric_value, dict):
            if "count" in metric_value:
                drifted_columns = int(metric_value["count"])
                drift_share = float(metric_value.get("share", 0.0))
                break

    if drifted_columns == 0:
        drift_details = result_dict.get("metrics", [])
        drifted_columns = sum(
            1
            for metric in drift_details
            if isinstance(metric, dict)
            and metric.get("value") is True
        )

    # Drift is detected if at least one feature is drifted
    drift_detected = drifted_columns > 0

    print("\n================================")
    print("       DRIFT DETECTION")
    print("================================")

    print("Drifted columns:", drifted_columns)
    print("Drift share:", drift_share)
    print(
        "Drift detected:",
        "YES" if drift_detected else "NO"
    )

    print("\nDrift report created successfully!")
    print("Report saved at:", report_path)

    return drift_detected


if __name__ == "__main__":
    detect_drift()