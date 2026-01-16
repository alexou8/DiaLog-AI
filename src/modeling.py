from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

@dataclass
class TrainResult:
    model: Pipeline
    metrics_text: str
    roc_auc: float

def train_classifier(df: pd.DataFrame, random_state: int = 42) -> TrainResult:
    features = ["hour", "dayofweek", "mins_since_meal", "last_meal_carbs", "mins_since_med", "last_med_units", "last_med_name"]
    target = "label_spike"

    X = df[features].copy()
    y = df[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y if y.nunique() > 1 else None
    )

    numeric = ["hour", "dayofweek", "mins_since_meal", "last_meal_carbs", "mins_since_med", "last_med_units"]
    categorical = ["last_med_name"]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), numeric),
            ("cat", Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                                   ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=random_state,
        class_weight="balanced",
    )

    pipe = Pipeline(steps=[("preprocess", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None

    report = classification_report(y_test, preds, zero_division=0)
    auc = roc_auc_score(y_test, proba) if proba is not None and y_test.nunique() > 1 else 0.0

    return TrainResult(model=pipe, metrics_text=report, roc_auc=auc)
