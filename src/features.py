import pandas as pd
import numpy as np

def to_dataframe(db_rows: list[tuple]) -> pd.DataFrame:
    df = pd.DataFrame(
        db_rows,
        columns=["timestamp", "event_type", "carbs_g", "med_name", "med_units", "glucose_mgdl", "notes"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def build_training_examples(
    df: pd.DataFrame,
    spike_threshold_mgdl: float = 200.0,
    lookback_hours: float = 6.0,
) -> pd.DataFrame:
    """
    Create examples anchored at each glucose reading.
    Features: time since last meal, carbs in last meal, time since last med, med units, hour-of-day, etc.
    Label: spike_event = glucose >= threshold
    """
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek

    meals = df[df["event_type"] == "meal"][["timestamp", "carbs_g"]].copy()
    meds = df[df["event_type"] == "med"][["timestamp", "med_name", "med_units"]].copy()
    glc = df[df["event_type"] == "glucose"][["timestamp", "glucose_mgdl", "hour", "dayofweek"]].copy()

    glc["label_spike"] = (glc["glucose_mgdl"] >= spike_threshold_mgdl).astype(int)

    # For each glucose reading, find most recent meal and med within lookback window
    lookback = pd.Timedelta(hours=lookback_hours)

    meal_times = meals["timestamp"].to_numpy()
    meal_carbs = meals["carbs_g"].fillna(0).to_numpy()

    med_times = meds["timestamp"].to_numpy()
    med_units = meds["med_units"].fillna(0).to_numpy()
    med_names = meds["med_name"].fillna("unknown").to_numpy()

    def last_event_before(ts, times_array):
        idx = np.searchsorted(times_array, ts) - 1
        return idx if idx >= 0 else None

    # Ensure arrays are sorted
    meals = meals.sort_values("timestamp")
    meds = meds.sort_values("timestamp")
    meal_times = meals["timestamp"].to_numpy()
    meal_carbs = meals["carbs_g"].fillna(0).to_numpy()
    med_times = meds["timestamp"].to_numpy()
    med_units = meds["med_units"].fillna(0).to_numpy()
    med_names = meds["med_name"].fillna("unknown").to_numpy()

    rows = []
    for _, r in glc.iterrows():
        ts = r["timestamp"]

        mi = last_event_before(ts, meal_times)
        if mi is not None and (ts - meal_times[mi]) <= lookback:
            mins_since_meal = (ts - meal_times[mi]).total_seconds() / 60.0
            last_meal_carbs = float(meal_carbs[mi]) if not pd.isna(meal_carbs[mi]) else 0.0
        else:
            mins_since_meal = np.nan
            last_meal_carbs = 0.0

        mdi = last_event_before(ts, med_times)
        if mdi is not None and (ts - med_times[mdi]) <= lookback:
            mins_since_med = (ts - med_times[mdi]).total_seconds() / 60.0
            last_med_units = float(med_units[mdi]) if not pd.isna(med_units[mdi]) else 0.0
            last_med_name = str(med_names[mdi]) if med_names[mdi] else "unknown"
        else:
            mins_since_med = np.nan
            last_med_units = 0.0
            last_med_name = "none"

        rows.append(
            {
                "timestamp": ts,
                "hour": int(r["hour"]),
                "dayofweek": int(r["dayofweek"]),
                "mins_since_meal": mins_since_meal,
                "last_meal_carbs": last_meal_carbs,
                "mins_since_med": mins_since_med,
                "last_med_units": last_med_units,
                "last_med_name": last_med_name,
                "glucose_mgdl": float(r["glucose_mgdl"]),
                "label_spike": int(r["label_spike"]),
            }
        )

    out = pd.DataFrame(rows)

    # Basic cleaning
    out["mins_since_meal"] = out["mins_since_meal"].fillna(out["mins_since_meal"].median())
    out["mins_since_med"] = out["mins_since_med"].fillna(out["mins_since_med"].median())
    return out
