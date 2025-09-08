import streamlit as st
import numpy as np
import joblib

# --------------------------
# Load trained model and scaler
# --------------------------
lasso_model = joblib.load("model/lasso_model.pkl")
scaler = joblib.load("model/scaler_lasso.pkl")

# --------------------------
# Streamlit UI
# --------------------------
st.title("🏠 House Price Prediction - Lasso Regression")

st.sidebar.header("Input House Features")
size_sqft = st.sidebar.number_input("Size (sqft)", min_value=500, max_value=5000, value=2000)
num_bedrooms = st.sidebar.selectbox("Number of Bedrooms", [1, 2, 3, 4, 5], index=2)
num_bathrooms = st.sidebar.selectbox("Number of Bathrooms", [1, 2, 3, 4], index=1)
age_years = st.sidebar.number_input("Age of House (years)", min_value=0, max_value=100, value=10)
location_score = st.sidebar.slider("Location Score (1-10)", 1.0, 10.0, 5.0)

# --------------------------
# Prepare input for prediction
# --------------------------
import pandas as pd

X_input = pd.DataFrame([[size_sqft, num_bedrooms, num_bathrooms, age_years, location_score]],
                       columns=["size_sqft","num_bedrooms","num_bathrooms","age_years","location_score"])
X_scaled = scaler.transform(X_input)


# --------------------------
# Make prediction
# --------------------------
if st.button("Predict House Price"):
    price_pred = lasso_model.predict(X_scaled)[0]
    st.success(f"Predicted House Price: ${price_pred:,.2f}")

# --------------------------
# Optional: Show model info
# --------------------------
if st.checkbox("Show Lasso Coefficients"):
    coef_dict = dict(zip(["size_sqft","num_bedrooms","num_bathrooms","age_years","location_score"], lasso_model.coef_))
    st.write(coef_dict)
    st.write(f"Intercept: {lasso_model.intercept_:.2f}")
