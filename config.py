import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
STORAGE_DIR = Path(
    os.environ.get("MLOPS_STORAGE_DIR", PROJECT_DIR)
).resolve()
DATA_DIR = STORAGE_DIR / "data"
MODEL_DIR = STORAGE_DIR / "models"
INCOMING_DIR = STORAGE_DIR / "incoming_data"
TRAINING_DATA = DATA_DIR / "training_data.csv"
NEW_DATA = DATA_DIR / "new_data.csv"
DRIFT_REPORT = DATA_DIR / "drift_report.html"
MODEL_PATH = MODEL_DIR / "model.pkl"
MODEL_CANDIDATE = MODEL_DIR / "model_candidate.pkl"
MODEL_BACKUP = MODEL_DIR / "model_backup.pkl"
MODEL_MANIFEST = MODEL_DIR / "model_manifest.json"
MLFLOW_DB = STORAGE_DIR / "mlflow.db"
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{MLFLOW_DB.as_posix()}",
)
WATCHER_STATUS = STORAGE_DIR / "watcher_status.txt"
PROCESSED_FILES = STORAGE_DIR / "processed_files.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
INCOMING_DIR.mkdir(parents=True, exist_ok=True)
