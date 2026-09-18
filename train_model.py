"""
3_train_model.py
------------------
Stage 3 of the pipeline: Model Training.
Stage 4 of the pipeline: Model Saving (joblib + pickle) is done at the
end of this same script, since saving happens right after training.

Run:
    python src/train_model.py
"""

import pickle

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_collection import clean_data, load_from_csv

TARGET = "SalePrice"


def build_pipeline(numeric_features, categorical_features, model):
    """Bundle preprocessing + model into a single sklearn Pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n--- {name} ---")
    print(f"RMSE: {rmse:,.2f}")
    print(f"MAE:  {mae:,.2f}")
    print(f"R2:   {r2:.4f}")
    return {"model": name, "rmse": rmse, "mae": mae, "r2": r2}


def main():
    # 1. Load + clean data (reuses the data_collection stage)
    df = load_from_csv("data/house_data.csv")
    df = clean_data(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "str"]).columns.tolist()
    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Train + compare a couple of candidate models
    candidates = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
    }

    results = []
    fitted_pipelines = {}
    for name, model in candidates.items():
        pipe = build_pipeline(numeric_features, categorical_features, model)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        results.append(evaluate(name, y_test, preds))
        fitted_pipelines[name] = pipe

    # 4. Pick the best model by R2
    best = max(results, key=lambda r: r["r2"])
    best_name = best["model"]
    best_pipeline = fitted_pipelines[best_name]
    print(f"\n>>> Best model: {best_name} (R2 = {best['r2']:.4f})")

    # 5. Save the model (Stage 4 — Model Saving)
    joblib.dump(best_pipeline, "models/house_price_model.joblib")
    with open("models/house_price_model.pkl", "wb") as f:
        pickle.dump(best_pipeline, f)

    # Save the feature list too — the apps need to know what columns to expect
    joblib.dump(
        {"numeric_features": numeric_features, "categorical_features": categorical_features},
        "models/feature_schema.joblib",
    )

    print("\nModel saved to:")
    print("  models/house_price_model.joblib")
    print("  models/house_price_model.pkl")
    print("  models/feature_schema.joblib")


if __name__ == "__main__":
    main()
