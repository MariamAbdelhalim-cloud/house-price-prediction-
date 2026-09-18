"""
5_app_streamlit.py
--------------------
Stage 5 of the pipeline: Deployment (Streamlit UI).

Run:
    streamlit run app_streamlit.py
"""

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

st.title("🏠 House Price Prediction")
st.write("Enter the details of a house below to estimate its sale price.")


@st.cache_resource
def load_model():
    return joblib.load("models/house_price_model.joblib")


model = load_model()

col1, col2 = st.columns(2)

with col1:
    lot_area = st.number_input("Lot Area (sq ft)", min_value=500, max_value=50000, value=8000)
    overall_qual = st.slider("Overall Quality (1=Poor, 10=Excellent)", 1, 10, 5)
    year_built = st.number_input("Year Built", min_value=1900, max_value=2026, value=2000)
    total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", min_value=0, max_value=5000, value=800)

with col2:
    gr_liv_area = st.number_input("Above-Ground Living Area (sq ft)", min_value=300, max_value=8000, value=1800)
    bedrooms = st.slider("Bedrooms", 1, 6, 3)
    full_bath = st.slider("Full Bathrooms", 1, 4, 2)
    garage_cars = st.slider("Garage Capacity (cars)", 0, 4, 2)

neighborhood = st.selectbox(
    "Neighborhood", ["Downtown", "Suburb_A", "Suburb_B", "Rural", "Uptown"]
)

if st.button("Predict Price", type="primary"):
    input_df = pd.DataFrame([{
        "LotArea": lot_area,
        "OverallQual": overall_qual,
        "YearBuilt": year_built,
        "TotalBsmtSF": total_bsmt_sf,
        "GrLivArea": gr_liv_area,
        "BedroomAbvGr": bedrooms,
        "FullBath": full_bath,
        "GarageCars": garage_cars,
        "Neighborhood": neighborhood,
    }])

    prediction = model.predict(input_df)[0]
    st.success(f"### Estimated Sale Price: ${prediction:,.0f}")

st.caption("Model: trained with scikit-learn, served via a Streamlit UI. "
           "See api_fastapi.py for the REST API version.")
