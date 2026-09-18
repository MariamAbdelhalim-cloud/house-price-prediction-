# 🏠 House Price Prediction — End-to-End ML Project

This project follows the standard 5-stage pipeline:

1. **Problem Definition** → `1_problem_definition.md`
2. **Data Collection** (CSV + API) → `data/`, `src/data_collection.py`
3. **Model Training** → `src/train_model.py`
4. **Model Saving** (Joblib + Pickle) → done inside `train_model.py`, saved to `models/`
5. **Deployment** (Streamlit + FastAPI) → `app_streamlit.py`, `api_fastapi.py`

## Project Structure
```
house-price-prediction/
├── 1_problem_definition.md     # Stage 1
├── data/
│   ├── generate_sample_data.py # creates a sample dataset (skip if you have a real one)
│   └── house_data.csv          # the dataset used for training (generated or your own)
├── src/
│   ├── data_collection.py      # Stage 2: load from CSV / API, clean data
│   └── train_model.py          # Stage 3 + 4: train, evaluate, save the model
├── models/                     # Stage 4 output: saved model files
│   ├── house_price_model.joblib
│   ├── house_price_model.pkl
│   └── feature_schema.joblib
├── app_streamlit.py            # Stage 5: Streamlit UI
├── api_fastapi.py              # Stage 5: FastAPI REST API
├── requirements.txt
└── README.md
```

## How to Run — Step by Step

### 0. Setup
```bash
pip install -r requirements.txt
```

### 1. Problem Definition
Read `1_problem_definition.md`. No code to run — this is the written stage
(objective, inputs/outputs, evaluation metrics, etc.).

### 2. Data Collection
If you don't have a real dataset yet (e.g. Kaggle's "House Prices - Advanced
Regression Techniques"), generate a sample one:
```bash
python data/generate_sample_data.py
```
If you DO have a real dataset, just save it as `data/house_data.csv` with a
`SalePrice` column as the target — everything downstream works unchanged.

Preview / test the collection + cleaning step:
```bash
python src/data_collection.py
```
This file also contains `load_from_api()`, showing the pattern for pulling
data from a real-estate API instead of (or in addition to) the CSV.

### 3 & 4. Model Training + Saving
```bash
python src/train_model.py
```
This trains a Linear Regression and a Random Forest model, evaluates both
with RMSE / MAE / R², picks the best one, and **saves it** to:
- `models/house_price_model.joblib` (recommended for loading back)
- `models/house_price_model.pkl` (pickle format, same purpose — shown for
  completeness since both are commonly requested)

### 5. Deployment

**Option A — Streamlit (interactive web UI):**
```bash
streamlit run app_streamlit.py
```
Opens a browser form where you enter house details and get a predicted price.

**Option B — FastAPI (REST API):**
```bash
uvicorn api_fastapi:app --reload
```
Then visit `http://127.0.0.1:8000/docs` for interactive Swagger docs, or send
a POST request to `/predict` with a JSON body (example in `api_fastapi.py`).

## Notes for Your Report
- **Why regression, not classification?** The target (`SalePrice`) is a
  continuous number, not a category — this is the key conceptual difference
  from a churn-prediction (classification) project.
- **Model comparison:** the training script compares Linear Regression vs.
  Random Forest and automatically keeps the better one — mention this in your
  report as "model selection based on R²."
- **Swap in real data anytime:** replace `data/house_data.csv` with the real
  Kaggle Ames Housing CSV (rename its target column to `SalePrice` if needed)
  — no other code changes are required.
