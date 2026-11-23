import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from config import (
    DATA_DIR,
    MLFLOW_DB,
    MODEL_PATH,
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


def wait_for_mlflow(process):
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("MLflow exited before becoming healthy.")
        try:
            with urllib.request.urlopen(
                "http://127.0.0.1:5000/health", timeout=2
            ) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(1)

    raise TimeoutError("MLflow did not become healthy within 90 seconds.")


def start_process(command):
    process = subprocess.Popen(command, cwd=PROJECT_DIR)
    processes.append(process)
    return process


def main():
    signal.signal(signal.SIGTERM, stop_processes)
    signal.signal(signal.SIGINT, stop_processes)

    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not TRAINING_DATA.exists():
        subprocess.run(
            [sys.executable, str(PROJECT_DIR / "src" / "data_preprocessing.py")],
            cwd=PROJECT_DIR,
            check=True,
        )

    mlflow_uri = f"sqlite:///{MLFLOW_DB.as_posix()}"
    mlflow_process = start_process([
        sys.executable,
        "-m",
        "mlflow",
        "server",
        "--host",
        "127.0.0.1",
        "--port",
        "5000",
        "--backend-store-uri",
        mlflow_uri,
        "--default-artifact-root",
        str(STORAGE_DIR / "mlruns"),
    ])
    wait_for_mlflow(mlflow_process)
    os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000"

    if not MODEL_PATH.exists():
        subprocess.run(
            [sys.executable, str(PROJECT_DIR / "src" / "train_model.py")],
            cwd=PROJECT_DIR,
            check=True,
        )

    start_process([sys.executable, str(PROJECT_DIR / "watcher.py")])
    start_process([
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(PROJECT_DIR / "dashboard.py"),
        "--server.address=0.0.0.0",
        f"--server.port={os.environ.get('PORT', '10000')}",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
    ])

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
