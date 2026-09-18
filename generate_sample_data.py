"""
generate_sample_data.py
------------------------
Creates a synthetic but realistic house-price dataset (house_data.csv).

Use this ONLY if you don't yet have a real dataset (e.g. from Kaggle's
"House Prices - Advanced Regression Techniques" competition). Once you
download the real dataset, just replace data/house_data.csv with it and
skip this script — the rest of the pipeline (train_model.py, apps) will
work the same way as long as the target column is named 'SalePrice'.

Run:
    python data/generate_sample_data.py
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 1500  # number of houses

neighborhoods = ["Downtown", "Suburb_A", "Suburb_B", "Rural", "Uptown"]
neighborhood_premium = {"Downtown": 1.35, "Uptown": 1.25, "Suburb_A": 1.05,
                         "Suburb_B": 1.0, "Rural": 0.8}

data = pd.DataFrame({
    "LotArea": np.random.randint(2000, 20000, N),
    "OverallQual": np.random.randint(1, 11, N),          # 1 (poor) - 10 (excellent)
    "YearBuilt": np.random.randint(1950, 2024, N),
    "TotalBsmtSF": np.random.randint(0, 2500, N),
    "GrLivArea": np.random.randint(600, 4000, N),         # above-ground living area
    "BedroomAbvGr": np.random.randint(1, 6, N),
    "FullBath": np.random.randint(1, 4, N),
    "GarageCars": np.random.randint(0, 4, N),
    "Neighborhood": np.random.choice(neighborhoods, N),
})

# Build a realistic price using a formula + noise
base_price = (
    data["GrLivArea"] * 90
    + data["TotalBsmtSF"] * 40
    + data["LotArea"] * 2
    + data["OverallQual"] * 9000
    + (2024 - data["YearBuilt"]) * -150
    + data["BedroomAbvGr"] * 3000
    + data["FullBath"] * 5000
    + data["GarageCars"] * 4000
    + 20000
)

premium = data["Neighborhood"].map(neighborhood_premium)
noise = np.random.normal(0, 15000, N)

data["SalePrice"] = (base_price * premium + noise).clip(lower=30000).round(0)

out_path = "data/house_data.csv"
data.to_csv(out_path, index=False)
print(f"Sample dataset saved to {out_path} ({len(data)} rows).")
print(data.head())
