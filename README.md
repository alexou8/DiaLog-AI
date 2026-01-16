# DiaLog — Diabetes Tracking + ML Pattern Insights (Informational Only)

An end-to-end ML pipeline that:
- stores diabetes-related logs (meals, meds, glucose) locally in SQLite
- preprocesses & engineers time-based features
- trains an ML classifier to estimate the probability of a "spike event"
- tunes hyperparameters
- generates weekly insights reports

## ⚠️ Disclaimer
This project is for informational pattern insights only and does NOT provide medical advice,
diagnosis, or medication dosing recommendations.

## Quickstart
### 1) Create a virtual env and install deps
pip install -r requirements.txt

### 2) Initialize the database
python scripts/init_db.py

### 3) Ingest sample data
python scripts/ingest_csv.py data/sample_logs.csv

### 4) Preprocess + create ML dataset
python scripts/preprocess_data.py

### 5) Train model
python scripts/train_model.py

### 6) Tune hyperparams (optional)
python scripts/tune_hyperparams.py

### 7) Generate weekly report
python scripts/weekly_report.py

## Data format (CSV)
See data/sample_logs.csv
