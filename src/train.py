"""
Minimal training script: loads a CSV, trains a model, saves it.
Edit paths as needed.
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib

DATA_PATH = os.getenv("DATA_PATH", "data/raw/dataset.csv")
TARGET = os.getenv("TARGET", "")  # set your target column name here
TASK = os.getenv("TASK", "classification")  # "classification" or "regression"
MODEL_OUT = os.getenv("MODEL_OUT", "models/model.joblib")

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Sample CSV missing at {DATA_PATH}. Put a small dataset there or change DATA_PATH.")
    df = pd.read_csv(DATA_PATH)
    if TARGET == "" or TARGET not in df.columns:
        raise ValueError(f"Please set TARGET env var to a valid target column from CSV. Found columns: {list(df.columns)}")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Simple numeric-only fallback (drop non-numeric)
    X = X.select_dtypes(include=["number"]).fillna(0)
    if X.empty:
        raise ValueError("No numeric features found after selection; add preprocessing or encode categoricals.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if TASK == "regression":
        model = RandomForestRegressor(random_state=42)
    else:
        model = RandomForestClassifier(random_state=42)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    if TASK == "regression":
        metric = mean_squared_error(y_test, preds, squared=False)
        print(f"RMSE: {metric:.4f}")
    else:
        metric = accuracy_score(y_test, preds)
        print(f"Accuracy: {metric:.4f}")

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    print(f"Saved model to {MODEL_OUT}")

if __name__ == "__main__":
    main()
