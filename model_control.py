import argparse
import json
import os
import shutil
from datetime import datetime, timezone

from config import MODEL_BACKUP, MODEL_CANDIDATE, MODEL_MANIFEST, MODEL_PATH


def load_manifest():
    if not MODEL_MANIFEST.exists():
        raise RuntimeError("No model manifest found.")
    with open(MODEL_MANIFEST, "r", encoding="utf-8") as file:
        return json.load(file)


def save_manifest(manifest):
    temporary = MODEL_MANIFEST.with_suffix(".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    temporary.replace(MODEL_MANIFEST)


def approve():
    manifest = load_manifest()
    if not MODEL_CANDIDATE.exists():
        raise RuntimeError("No candidate model is available.")

    if MODEL_PATH.exists():
        shutil.copy2(MODEL_PATH, MODEL_BACKUP)

    temporary_active = MODEL_PATH.with_suffix(".tmp.pkl")
    shutil.copy2(MODEL_CANDIDATE, temporary_active)
    os.replace(temporary_active, MODEL_PATH)

    manifest["status"] = "approved"
    manifest["approved_at"] = datetime.now(timezone.utc).isoformat()
    save_manifest(manifest)
    print("Candidate model approved and activated.")


def rollback():
    if not MODEL_BACKUP.exists():
        raise RuntimeError("No backup model is available for rollback.")

    temporary_active = MODEL_PATH.with_suffix(".tmp.pkl")
    shutil.copy2(MODEL_BACKUP, temporary_active)
    os.replace(temporary_active, MODEL_PATH)

    manifest = load_manifest() if MODEL_MANIFEST.exists() else {}
    manifest["status"] = "rolled_back"
    manifest["rolled_back_at"] = datetime.now(timezone.utc).isoformat()
    save_manifest(manifest)
    print("Active model restored from backup.")


def main():
    parser = argparse.ArgumentParser(description="Model candidate approval controls")
    parser.add_argument("action", choices=["approve", "rollback"])
    args = parser.parse_args()

    if args.action == "approve":
        approve()
    else:
        rollback()


if __name__ == "__main__":
    main()
