import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent


def run_script(relative_path: str) -> int:
    return subprocess.run(
        [sys.executable, str(PROJECT_DIR / relative_path)],
        cwd=PROJECT_DIR,
        check=False,
    ).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Automated Model Retraining System")
    parser.add_argument(
        "command",
        choices=["setup", "train", "detect", "retrain", "watch", "dashboard"],
    )
    args = parser.parse_args()

    commands = {
        "setup": "src/data_preprocessing.py",
        "train": "src/train_model.py",
        "detect": "src/drift_detection.py",
        "retrain": "src/retrain.py",
        "watch": "watcher.py",
    }

    if args.command == "dashboard":
        return subprocess.run(
            ["streamlit", "run", "dashboard.py"], cwd=PROJECT_DIR, check=False
        ).returncode

    return run_script(commands[args.command])


if __name__ == "__main__":
    raise SystemExit(main())
