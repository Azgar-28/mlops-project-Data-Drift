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

    # Find matching columns in reference-column order
    common_columns = [
        col for col in reference_features.columns
        if col in new_data.columns
    ]

    # Stop if no matching columns exist
    if not common_columns:
        print("\nWARNING: No matching feature columns found.")
        print("Drift detection cannot be performed.")
        print("Please upload a CSV with compatible feature names.")

        return None

    # Select matching columns
    reference_features = reference_features[common_columns].copy()
    incoming_features = new_data[common_columns].copy()

    print("\n================================")
    print("       DATASET INFORMATION")
    print("================================")

    print("Reference rows:", len(reference_features))
    print("Incoming rows:", len(incoming_features))
    print("Features compared:", len(common_columns))

    ignored_columns = [
        col for col in new_data.columns
        if col not in common_columns
    ]

    print("Ignored incoming columns:", ignored_columns)

    # Check incompatible data types
    incompatible_columns = []

    for col in common_columns:

        reference_type = pd.api.types.infer_dtype(
            reference_features[col].dropna(),
            skipna=True
        )

        incoming_type = pd.api.types.infer_dtype(
            incoming_features[col].dropna(),
            skipna=True
        )

        if (
            reference_type != "empty"
            and incoming_type != "empty"
            and reference_type != incoming_type
        ):
            incompatible_columns.append(
                (col, reference_type, incoming_type)
            )

    if incompatible_columns:

        details = "; ".join(
            f"{col}: reference={ref_type}, incoming={new_type}"
            for col, ref_type, new_type in incompatible_columns
        )

        raise ValueError(
            "Matching columns have incompatible data types. "
            "Preprocess these columns before drift detection: "
            + details
        )

    # Check columns with no usable values
    unusable_columns = [
        col for col in common_columns
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

    for col in common_columns:
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
    result.save_html(report_path)

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