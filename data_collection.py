"""
2_data_collection.py
---------------------
Stage 2 of the pipeline: Data Collection.

Two sources are demonstrated, matching the course requirement:
  A) CSV  -> load_from_csv()   (primary source used by this project)
  B) API  -> load_from_api()   (example pattern for pulling live listings)

Run directly to preview the collected/cleaned data:
    python src/data_collection.py
"""

import pandas as pd
import requests


# ---------------------------------------------------------------------
# A) LOAD FROM CSV  (main data source for this project)
# ---------------------------------------------------------------------
def load_from_csv(path: str = "data/house_data.csv") -> pd.DataFrame:
    """Load the housing dataset from a local CSV file."""
    df = pd.read_csv(path)
    print(f"[CSV] Loaded {len(df)} rows, {df.shape[1]} columns from {path}")
    return df


# ---------------------------------------------------------------------
# B) LOAD FROM AN API  (example — swap in a real real-estate API)
# ---------------------------------------------------------------------
def load_from_api(api_url: str, params: dict | None = None) -> pd.DataFrame:
    """
    Example pattern for pulling housing/listing data from a REST API
    (e.g. a real-estate data provider on RapidAPI, a city open-data
    portal, etc.). Replace `api_url` and `params` with your real
    endpoint and API key.

    Example usage:
        df = load_from_api(
            "https://api.example.com/v1/listings",
            params={"city": "Cairo", "limit": 100, "api_key": "YOUR_KEY"}
        )
    """
    response = requests.get(api_url, params=params, timeout=15)
    response.raise_for_status()
    payload = response.json()

    # Adjust this line to match the real API's response structure,
    # e.g. payload["results"] or payload["data"]["listings"]
    records = payload.get("results", payload)

    df = pd.DataFrame(records)
    print(f"[API] Loaded {len(df)} rows from {api_url}")
    return df


# ---------------------------------------------------------------------
# CLEANING (shared by whichever source you use)
# ---------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: handle missing values, drop duplicates."""
    df = df.drop_duplicates()

    # Fill numeric missing values with the median, categorical with mode
    for col in df.columns:
        if df[col].dtype in ["int64", "float64"]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    print(f"[CLEAN] {len(df)} rows remain after cleaning.")
    return df


if __name__ == "__main__":
    # --- Using CSV (default path for this project) ---
    df = load_from_csv("data/house_data.csv")
    df = clean_data(df)
    print(df.describe(include="all").T)

    # --- Example of the API path (commented out; needs a real endpoint/key) ---
    # api_df = load_from_api("https://api.example.com/v1/listings",
    #                         params={"city": "Cairo", "api_key": "YOUR_KEY"})
    # api_df = clean_data(api_df)
