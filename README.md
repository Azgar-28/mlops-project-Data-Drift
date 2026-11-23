# Automated Model Retraining System Using Data Drift Detection

## 1. Project Overview

This project implements an automated machine learning monitoring and retraining workflow. It detects data drift in incoming data and triggers model retraining when drift is detected.

The system uses Python, Evidently, Scikit-learn, MLflow, Streamlit, and Watchdog to monitor data, manage model training, track experiments, and display system status.

## 2. Objectives

- Detect data drift between reference and incoming datasets.
- Generate an HTML drift detection report.
- Automatically trigger model retraining when drift is detected.
- Track training metrics and parameters using MLflow.
- Monitor incoming CSV files automatically.
- Display model and monitoring information through a Streamlit dashboard.

## 3. Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Evidently
- MLflow
- Streamlit
- Watchdog
- DVC
- Git and GitHub

## 4. Project Structure

```text
PythonProject1/
├── data/
│   ├── training_data.csv
│   ├── new_data.csv
│   └── drift_report.html
├── incoming_data/
├── models/
│   └── model.pkl
├── src/
│   ├── data_preprocessing.py
│   ├── drift_detection.py
│   ├── retrain.py
│   └── train_model.py
├── dashboard.py
├── watcher.py
├── mlflow.db
├── requirements.txt
└── README.md
```

Note: The structure above describes the main project files. Some files, such as `requirements.txt`, should be created if they do not already exist.

## 5. System Workflow

1. The watcher monitors the `incoming_data` directory.
2. When a new CSV file arrives, the watcher waits for the file to finish copying.
3. The incoming CSV is copied to `data/new_data.csv`.
4. The drift detection module compares incoming data with the reference data.
5. If drift is detected, the retraining script starts.
6. A Random Forest classifier is trained and evaluated.
7. The trained model is saved as `models/model.pkl`.
8. MLflow records model parameters, accuracy, and run information.
9. The HTML drift report and monitoring status are made available to the dashboard.

## 6. Installation and Setup

### Step 1: Open the project directory

```powershell
cd C:\path\to\mlops-project--2-main
```

### Step 2: Create and activate the virtual environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Docker setup

Start Docker Desktop, then run:

```powershell
docker compose up --build
```

Services:

- Dashboard: http://localhost:8501
- MLflow: http://localhost:5000
- Watcher: runs automatically inside the `watcher` container

### Deploy to Render

1. Push this project to a GitHub repository.
2. In Render, choose **New → Blueprint** and connect that repository.
3. Review the `mlops-drift-monitor` web service and create it.
4. After the first deploy, open the service URL. Render's persistent disk stores datasets, models, reports, and MLflow data across restarts.

The Render service starts the dashboard and CSV watcher together. MLflow tracking uses a SQLite database on the persistent disk without starting an additional MLflow web server, reducing memory use. Only the Streamlit dashboard is exposed publicly. The Blueprint uses a paid Starter service because Render persistent disks are not available on free web services.

For incoming CSVs on Render, use the dashboard's **Import CSV** page to upload a file. The hosted service cannot monitor a folder on your personal computer.

The Render deployment is configured by `render.yaml` and `render_start.py`.

### Model approval controls

```powershell
python model_control.py approve
python model_control.py rollback
```

Approval activates a candidate model; rollback restores the previous backup.

### DVC setup

The dataset files are tracked by DVC. Configure a DVC remote for your own machine or cloud storage before pushing dataset versions:

```powershell
python main.py setup
dvc remote add -d storage <your-dvc-remote-url>
dvc add data/training_data.csv data/new_data.csv
dvc push
```

The local `dvc_storage` directory used during development is not published to GitHub. Render generates the sample datasets during its container build and first startup.

### Step 3: Install dependencies

If `requirements.txt` is available:

```powershell
pip install -r requirements.txt
```

Otherwise, install the required packages:

```powershell
pip install pandas numpy scikit-learn joblib evidently mlflow streamlit watchdog dvc
```

## 7. Running the Project

For the quickest setup, run:

```powershell
.\run_project.bat
```

Or run one CLI command at a time:

```powershell
python main.py setup
python main.py train
python main.py detect
python main.py retrain
python main.py watch
python main.py dashboard
```

To run components manually, open separate terminals.

### Terminal 1: Start MLflow

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open: http://localhost:5000

### Terminal 2: Start the Streamlit dashboard

```powershell
streamlit run dashboard.py
```

Open: http://localhost:8501

### Terminal 3: Start automatic file monitoring

```powershell
python watcher.py
```

Copy a CSV file into the `incoming_data` directory to trigger the workflow.

## 8. Experimental Results

The following results were observed during testing:

| Metric | Result |
|---|---|
| Reference dataset | 569 rows, 31 columns |
| Incoming dataset | 200 rows, 30 columns |
| Drifted columns | 2 |
| Drift share | 6.67% |
| Model | RandomForestClassifier |
| Number of estimators | 100 |
| Random state | 42 |
| Model accuracy | 96.49% |
| Model artifact | `models/model.pkl` |

The reported accuracy was calculated on a held-out split of the original labelled dataset. It does not represent evaluation on the incoming data.

## 9. MLflow Experiment Tracking

MLflow records:

- Model parameters
- Accuracy metric
- Training run information
- Logged model information and artifacts

Experiment name: `Automated_Model_Retraining`

Run name: `Automatic_Retraining`

## 10. Limitations

- The current retraining script trains on the original labelled dataset. Incoming CSV files are used for drift detection, not directly for training.
- Incoming data must be validated before it is used for future retraining.
- The watcher runs locally and must remain active to detect incoming files.
- The processed-file tracking currently lasts only for the active watcher session.
- Production deployment, external data ingestion, and persistent monitoring are future improvements.

## 11. Future Enhancements

- Incorporate validated, labelled incoming data into retraining.
- Add persistent duplicate-event tracking.
- Implement model performance monitoring and approval workflows.
- Add automatic dashboard refresh.
- Configure a DVC remote for dataset storage.
- Deploy the application to a cloud environment.
- Add automated testing and a CI/CD pipeline.

## 12. Conclusion

The project demonstrates a local automated machine learning workflow that detects data drift, triggers model retraining, logs experiments in MLflow, and provides a monitoring dashboard. It establishes a foundation for more robust and production-ready MLOps systems.
