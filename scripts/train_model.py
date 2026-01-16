import joblib
import pandas as pd
from src.config import PATHS
from src.modeling import train_classifier

if __name__ == "__main__":
    df = pd.read_csv(PATHS.PROCESSED)
    result = train_classifier(df)

    PATHS.MODELS.mkdir(parents=True, exist_ok=True)
    model_path = PATHS.MODELS / "spike_classifier.joblib"
    joblib.dump(result.model, model_path)

    print("Model trained.")
    print(f"ROC-AUC: {result.roc_auc:.3f}")
    print(result.metrics_text)
    print(f"Saved model to: {model_path}")
