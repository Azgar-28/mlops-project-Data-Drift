import os
import sys
import json
import shutil
from datetime import datetime, timezone

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from data_preprocessing import load_data, split_data
from drift_detection import detect_drift

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from config import (
    MLFLOW_DB,
    MLFLOW_TRACKING_URI,
    MODEL_DIR,
    MODEL_PATH,
    MODEL_CANDIDATE,
    MODEL_BACKUP,
    MODEL_MANIFEST,
    NEW_DATA,
)

MLFLOW_DB.parent.mkdir(parents=True, exist_ok=True)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

mlflow.set_experiment("Automated_Model_Retraining")


def retrain_model():

    print("\n================================")
    print("        MODEL RETRAINING")
    print("================================")

    # Load original dataset
    X, y = load_data()

    training_source = "original_labelled_dataset"

    # Use incoming data for retraining only when it provides a valid target column.
    if NEW_DATA.exists():
        incoming = pd.read_csv(NEW_DATA)

        if "target" in incoming.columns:
            missing_features = [column for column in X.columns if column not in incoming.columns]
            extra_features = [column for column in incoming.columns if column != "target" and column not in X.columns]

            if missing_features:
                print("Incoming training data is missing features:", missing_features)
            elif extra_features:
                print("Incoming training data contains ignored extra features:", extra_features)
            elif len(incoming) >= 20 and incoming["target"].notna().sum() >= 20:
                incoming_features = incoming[X.columns].apply(pd.to_numeric, errors="coerce")
                incoming_target = incoming["target"]

                labelled_rows = incoming_features.notna().all(axis=1) & incoming_target.notna()
                incoming_features = incoming_features.loc[labelled_rows]
                incoming_target = incoming_target.loc[labelled_rows]

                if len(incoming_features) >= 20 and incoming_target.nunique() >= 2:
                    X = pd.concat([X, incoming_features], ignore_index=True)
                    y = pd.concat([y, incoming_target], ignore_index=True)
                    training_source = "original_plus_incoming_labelled_data"
                    print("Incoming labelled rows added:", len(incoming_features))
                else:
                    print("Incoming labels were insufficient for retraining; using fallback data.")
            else:
                print("Incoming dataset has too few labelled rows; using fallback data.")
        else:
            print("Incoming dataset has no target column; using fallback data.")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state if "random_state" in locals() else 42,
        stratify=y
    )

    # Model parameters
    n_estimators = 100
    random_state = 42

    # Start MLflow run
    with mlflow.start_run(run_name="Automatic_Retraining"):

        # Create model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )

        # Train model
        model.fit(X_train, y_train)

        # Predictions
        predictions = model.predict(X_test)

        # Accuracy
        accuracy = accuracy_score(
            y_test,
            predictions
        )

        print("New Model Accuracy:", accuracy)

        # Log parameters
        mlflow.log_param(
            "model_type",
            "RandomForestClassifier"
        )

        mlflow.log_param(
            "n_estimators",
            n_estimators
        )

        mlflow.log_param(
            "random_state",
            random_state
        )

        mlflow.log_param(
            "training_reason",
            "Data Drift Detected"
        )

        mlflow.log_param(
            "training_source",
            training_source
        )

        # Log metric
        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        # Save the retrained model as a candidate pending approval.
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_CANDIDATE)

        manifest = {
            "status": "pending_approval",
            "accuracy": float(accuracy),
            "training_source": training_source,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "candidate_path": str(MODEL_CANDIDATE),
            "active_path": str(MODEL_PATH),
            "backup_path": str(MODEL_BACKUP),
        }

        temporary_manifest = MODEL_MANIFEST.with_suffix(".tmp")
        with open(temporary_manifest, "w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2)
        temporary_manifest.replace(MODEL_MANIFEST)

        # Log model to MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="retrained_model"
        )

        print("Candidate model saved successfully!")
        print("Approval is required before activation.")
        print("MLflow retraining run logged!")
        print("Candidate path:", MODEL_CANDIDATE)


def main():

    print("================================")
    print(" AUTOMATED MODEL RETRAINING")
    print("================================")

    # Check for drift
    drift_detected = detect_drift()

    if drift_detected:

        print("\nWARNING: Data drift detected!")
        print("Starting automatic model retraining...")

        retrain_model()

    else:

        print("\nNo data drift detected.")
        print("Existing model will be used.")


if __name__ == "__main__":
    main()