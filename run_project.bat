
@echo off
cd /d "%~dp0"

echo Starting Automated Model Retraining System...

if not exist ".venv\Scripts\python.exe" (
    py -3.12 -m venv .venv
    if errorlevel 1 exit /b 1
)

call .venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python src\data_preprocessing.py

start "MLflow" cmd /k "cd /d "%~dp0" && call .venv\Scripts\activate.bat && mlflow ui --backend-store-uri sqlite:///%~dp0mlflow.db --port 5000"
start "File Watcher" cmd /k "cd /d "%~dp0" && call .venv\Scripts\activate.bat && python watcher.py"

streamlit run dashboard.py
