import joblib
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from src.config import PATHS
from src.modeling import train_classifier

# Minimal tuning wrapper: reuse the pipeline but search RF params
if __name__ == "__main__":
    df = pd.read_csv(PATHS.PROCESSED)
    base = train_classifier(df).model

    X = df[["hour","dayofweek","mins_since_meal","last_meal_carbs","mins_since_med","last_med_units","last_med_name"]]
    y = df["label_spike"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    param_dist = {
        "clf__n_estimators": [200, 400, 800],
        "clf__max_depth": [None, 6, 10, 16],
        "clf__min_samples_split": [2, 4, 8],
        "clf__min_samples_leaf": [1, 2, 4],
    }

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=12,
        cv=5,
        scoring="roc_auc",
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train, y_train)
    best = search.best_estimator_

    PATHS.MODELS.mkdir(parents=True, exist_ok=True)
    out = PATHS.MODELS / "spike_classifier_tuned.joblib"
    joblib.dump(best, out)

    print("Best params:", search.best_params_)
    print("Best CV ROC-AUC:", search.best_score_)
    print("Saved tuned model to:", out)
