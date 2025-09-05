import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# Load the trained model and scaler
model = load("models/linear_regression_model2_scaled.joblib")
scaler = load("models/scaler_model2.joblib")

# Streamlit app title
st.title("House Price Prediction App")
st.write("Predict house prices based on various features using a trained regression model.")

# Input fields for house features
st.sidebar.header("Input Features")
house_size = st.sidebar.number_input("House Size (sq ft)", min_value=800, max_value=5000, value=2000, step=100)
bedrooms = st.sidebar.number_input("Number of Bedrooms", min_value=2, max_value=6, value=3, step=1)
bathrooms = st.sidebar.number_input("Number of Bathrooms", min_value=1, max_value=4, value=2, step=1)
house_age = st.sidebar.number_input("House Age (years)", min_value=0, max_value=50, value=10, step=1)
garage_size = st.sidebar.number_input("Garage Size (number of cars)", min_value=0, max_value=3, value=2, step=1)
lot_size = st.sidebar.number_input("Lot Size (sq ft)", min_value=3000, max_value=20000, value=8000, step=500)

# Predict button
if st.sidebar.button("Predict"):
    # Create a DataFrame for the input
    input_data = pd.DataFrame({
        'house_size': [house_size],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'house_age': [house_age],
        'garage_size': [garage_size],
        'lot_size': [lot_size]
    })

    # Scale the input data
    input_scaled = scaler.transform(input_data)

    # Make prediction
    predicted_price = model.predict(input_scaled)[0]

    # Display the prediction
    st.write("### Predicted House Price")
    st.write(f"${predicted_price:,.2f}")

    # Save the input and prediction to a CSV file
    result_data = input_data.copy()
    result_data["Predicted Price"] = [predicted_price]
    try:
        existing_data = pd.read_csv("predictions.csv")
        updated_data = pd.concat([existing_data, result_data], ignore_index=True)
    except FileNotFoundError:
        updated_data = result_data
    updated_data.to_csv("predictions.csv", index=False)
    st.success("Prediction saved to 'predictions.csv'.")

# Example dataset for batch predictions

example_data = pd.DataFrame({
    'house_size': [1800, 3200, 1200, 4000],
    'bedrooms': [3, 5, 2, 6],
    'bathrooms': [2, 4, 1, 4],
    'house_age': [5, 2, 25, 8],
    'garage_size': [2, 3, 1, 3],
    'lot_size': [7000, 12000, 5000, 15000]
})

# Ensure column names match the training data
example_data = example_data[['house_size', 'bedrooms', 'bathrooms', 'house_age', 'garage_size', 'lot_size']]

# Scale the example data
example_scaled = scaler.transform(example_data)

# Predict prices for the example data
example_data["Predicted Price"] = model.predict(example_scaled).round(2)

# Save example predictions to CSV
example_data.to_csv("example_predictions.csv", index=False)

# Display the example predictions
# st.write("### Example Batch Prediction Results")
# st.dataframe(example_data)

# Display saved predictions with pagination
st.write("### Prediction Results")
try:
    saved_data = pd.read_csv("predictions.csv")
    page = st.number_input("Page Number", min_value=1, max_value=(len(saved_data) // 5) + 1, step=1)
    start_idx = (page - 1) * 5
    end_idx = start_idx + 5
    st.dataframe(saved_data.iloc[start_idx:end_idx])
except FileNotFoundError:
    st.write("No saved predictions found.")