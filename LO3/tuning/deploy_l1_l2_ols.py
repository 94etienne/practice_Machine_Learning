import streamlit as st
import numpy as np
import joblib

# ------------------------------
# Load scaler and model parameters
# ------------------------------
scaler = joblib.load("models_l1_l2_ols/scaler.pkl")

# OLS
ols_params = np.load("models_l1_l2_ols/ols_model_params.npz")
ols_coef = ols_params["coef"]
ols_intercept = ols_params["intercept"]

# Ridge
ridge_params = np.load("models_l1_l2_ols/ridge_model_params.npz")
ridge_coef = ridge_params["coef"]
ridge_intercept = ridge_params["intercept"]

# Lasso
lasso_params = np.load("models_l1_l2_ols/lasso_model_params.npz")
lasso_coef = lasso_params["coef"]
lasso_intercept = lasso_params["intercept"]

# ------------------------------
# Prediction function
# ------------------------------
def predict(model_name, X_scaled):
    if model_name == "OLS":
        return np.dot(X_scaled, ols_coef) + ols_intercept
    elif model_name == "Ridge":
        return np.dot(X_scaled, ridge_coef) + ridge_intercept
    elif model_name == "Lasso":
        return np.dot(X_scaled, lasso_coef) + lasso_intercept
    else:
        return None

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🏠 House Price Prediction")

st.sidebar.header("Select Model")
model_name = st.sidebar.selectbox("Choose regression model", ["OLS", "Ridge", "Lasso"])

st.sidebar.header("Input House Features")
size_sqft = st.sidebar.number_input("Size (sqft)", min_value=500, max_value=5000, value=2000)
num_bedrooms = st.sidebar.selectbox("Number of Bedrooms", [1, 2, 3, 4, 5], index=2)
num_bathrooms = st.sidebar.selectbox("Number of Bathrooms", [1, 2, 3, 4], index=1)
age_years = st.sidebar.number_input("Age of House (years)", min_value=0, max_value=100, value=10)
location_score = st.sidebar.slider("Location Score (1-10)", 1.0, 10.0, 5.0)

# Prepare input for model
X_input = np.array([[size_sqft, num_bedrooms, num_bathrooms, age_years, location_score]])
X_scaled = scaler.transform(X_input)

# Prediction button
if st.button("Predict"):
    prediction = predict(model_name, X_scaled)
    st.success(f"Predicted House Price using {model_name}: ${prediction[0]:,.2f}")
