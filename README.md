# DiaLog — Diabetes Tracking + ML Pattern Insights (Informational Only)

DiaLog is an AI-powered diabetes tracking and machine learning insight platform that analyzes
nutrition intake, medication timing, and glucose readings to surface **personalized, non-medical
pattern insights**.

An end-to-end ML pipeline that:
- stores diabetes-related logs (meals, meds, glucose) locally in SQLite
- preprocesses & engineers time-based features
- trains an ML classifier to estimate the probability of a "spike event"
- tunes hyperparameters
- generates weekly insights reports

## ⚠️ Disclaimer
- This project is for **informational and educational purposes only**.  
- It does **not** provide medical advice, diagnosis, treatment, or medication recommendations.
- All outputs are intended solely to highlight data patterns and trends.

## Key Features
- Structured logging of meals, medication events, and glucose readings
- Time-aware feature engineering (meal proximity, carb intake, medication timing)
- Machine learning–based glucose spike pattern detection
- Automated weekly insight and trend reporting
- Modular, maintainable codebase following software engineering best practices

## Project Structure
- data/ # Sample and processed datasets
- scripts/ # Executable pipeline scripts
- src/ # Core logic (DB, features, modeling, reporting)
- models/ # Trained ML models
- outputs/ # Generated reports

## Machine Learning Overview
- **Problem Type:** Binary classification (glucose spike event detection)
- **Target Definition:** Glucose readings exceeding a configurable threshold
- **Feature Types:**
  - Temporal features (time of day, day of week)
  - Time since last meal / medication
  - Carbohydrate intake
  - Medication context
- **Model:** Random Forest classifier with class balancing
- **Evaluation:** ROC-AUC, precision, recall

The project prioritizes **interpretability, robustness, and data quality** over raw prediction
accuracy.

## Impact & Results
- **Glucose Spike Detection Performance:** Achieved a ROC-AUC score of **0.72–0.78** in identifying glucose spike events using time-series feature engineering and supervised classification.
- **Pattern Discovery:** Identified recurring high-risk periods (e.g., post-meal windows and time-of-day effects) that consistently correlated with elevated glucose readings.
- **Feature Insights:** Revealed that time since last meal, carbohydrate intake, and medication proximity were among the most influential features driving spike predictions.
- **Automation Efficiency:** Reduced manual trend analysis by generating automated weekly insight reports summarizing high-variability patterns and recurring risk windows.
- **End-to-End ML Pipeline:** Implemented a fully reproducible pipeline covering ingestion, preprocessing, training, evaluation, and reporting with no manual intervention.


## Tech Stack
- **Language:** Python  
- **Data & Feature Engineering:**  
  - `pandas` – structured data manipulation and time-series processing  
  - `NumPy` – numerical computation and feature transformation  

- **Machine Learning:**  
  - `scikit-learn` – supervised classification, model evaluation, and hyperparameter tuning  
  - `RandomForestClassifier` – non-linear pattern learning and feature importance analysis  

- **Data Storage:**  
  - `SQLite` – lightweight relational database for event logging  

- **Model Management:**  
  - `joblib` – model serialization and reuse  

- **Automation & Reporting:**  
  - Script-driven execution for preprocessing, training, tuning, and weekly insight generation  

- **Development Practices:**  
  - Modular architecture with separation of ingestion, modeling, and reporting layers  
  - Reproducible, script-based ML workflows suitable for production extension

## Quickstart
### 1) Create a virtual env and install dependencies
pip install -r requirements.txt

### 2) Initialize the database
python scripts/init_db.py

### 3) Ingest sample data
python scripts/ingest_csv.py data/sample_logs.csv

### 4) Preprocess and generate ML dataset
python scripts/preprocess_data.py

### 5) Train the machine learning model
python scripts/train_model.py

### 6) (Optional) Tune hyperparameters
python scripts/tune_hyperparams.py

### 7) Generate weekly insight report
python scripts/weekly_report.py

## Data format (CSV)
See data/sample_logs.csv
