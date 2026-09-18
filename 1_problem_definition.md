# 1. Problem Definition — House Price Prediction

## Background
Real estate pricing is influenced by many interacting factors — location, size,
condition, and amenities — which makes manual valuation slow, inconsistent, and
often subjective. A data-driven model can estimate a property's fair market
value quickly and objectively, helping buyers, sellers, agents, and lenders make
better decisions.

## Problem Statement
Given structural, locational, and quality-related features of a house, predict
its **sale price** (a continuous numeric value).

## Objective
Build and deploy an end-to-end machine learning system that:
1. Collects and cleans housing data.
2. Trains a regression model to predict house prices.
3. Saves the trained model for reuse.
4. Serves predictions through a web app (Streamlit) and an API (FastAPI).

## Type of Problem
- **Supervised Learning → Regression** (target variable `SalePrice` is continuous,
  not a category — this is the key difference from a classification problem
  like churn, where the target is Yes/No).

## Inputs and Outputs
- **Input (features):** square footage, number of bedrooms/bathrooms, lot size,
  year built, overall quality, neighborhood/location, garage size, etc.
- **Output (target):** predicted sale price in currency units (e.g. USD).

## Data Sources
- Historical housing sales data (CSV) — e.g. Kaggle's "House Prices - Advanced
  Regression Techniques" (Ames Housing dataset), or a local real estate dataset.
- Optionally, live listing data pulled from a real-estate API (e.g. Zillow,
  RapidAPI real-estate endpoints) to supplement or update the dataset.

## Scope and Constraints
- Single-family residential properties only (not commercial or land-only).
- Prices reflect the market conditions/time period of the training data.
- Model accuracy depends on data quality and completeness (missing values,
  outliers, regional coverage).

## Success Criteria / Evaluation Metrics
- **RMSE** (Root Mean Squared Error) — penalizes large prediction errors.
- **MAE** (Mean Absolute Error) — average absolute error in price units.
- **R² Score** — proportion of price variance explained by the model.
- Target: R² > 0.80 on the held-out test set (adjust based on dataset size/quality).

## Stakeholders
- Home buyers and sellers (fair pricing).
- Real estate agents (competitive, data-backed listings).
- Banks/lenders (mortgage and collateral valuation).
- Property investors (identifying under/over-valued properties).

## Project Pipeline (matches course requirement)
1. **Problem Definition** — this document.
2. **Data Collection** — `src/data_collection.py` (CSV + optional API).
3. **Model Training** — `src/train_model.py`.
4. **Model Saving** — model persisted with `joblib`/`pickle` in `models/`.
5. **Deployment** — `app_streamlit.py` (UI) and `api_fastapi.py` (REST API).
