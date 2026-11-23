import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path

from config import (
    DATA_DIR,
    NEW_DATA,
    STORAGE_DIR,
    TRAINING_DATA,
)

PROJECT_DIR = Path(__file__).resolve().parent
processes = []


def stop_processes(*_):
    for process in reversed(processes):
        if process.poll() is None:
            process.terminate()

    for process in reversed(processes):
        if process.poll() is None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()

    raise SystemExit(0)


def start_process(command):
    process = subprocess.Popen(command, cwd=PROJECT_DIR)
    processes.append(process)
    return process


def main():
    signal.signal(signal.SIGTERM, stop_processes)
    signal.signal(signal.SIGINT, stop_processes)

    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    baked_data = PROJECT_DIR / "data"
    for source, destination in (
        (baked_data / "training_data.csv", TRAINING_DATA),
        (baked_data / "new_data.csv", NEW_DATA),
    ):
        if not destination.exists() and source.exists():
            shutil.copy2(source, destination)

    os.environ["MLFLOW_TRACKING_URI"] = (
        f"sqlite:///{(STORAGE_DIR / 'mlflow.db').as_posix()}"
    )

    start_process([
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(PROJECT_DIR / "dashboard.py"),
        "--server.address=0.0.0.0",
        f"--server.port={os.environ.get('PORT', '10000')}",
        "--server.headless=true",
        "--server.maxUploadSize=50",
        "--browser.gatherUsageStats=false",
    ])
    start_process([sys.executable, str(PROJECT_DIR / "watcher.py")])

    while True:
        for process in processes:
            exit_code = process.poll()
            if exit_code is not None:
                raise RuntimeError(
                    f"Service process {process.args} exited with code {exit_code}."
                )
        time.sleep(2)


if __name__ == "__main__":
    main()
