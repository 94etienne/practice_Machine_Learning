import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler
model_filename = "model/ridge_model_l2.pkl"
scaler_filename = "model/scaler.pkl"
loaded_model = joblib.load(model_filename)
scaler = joblib.load(scaler_filename)

# Streamlit app title
st.title("House Price Prediction App")

# Input fields for user data
st.header("Enter House Details:")
size_sqft = st.number_input("Size (in sqft):", min_value=800, max_value=4000, value=1500, step=100)
num_bedrooms = st.selectbox("Number of Bedrooms:", [2, 3, 4, 5])
num_bathrooms = st.selectbox("Number of Bathrooms:", [1, 2, 3, 4])
age_years = st.number_input("Age of the House (in years):", min_value=0, max_value=50, value=10)
location_score = st.number_input("Location Score (1-10):", min_value=1, max_value=10, value=5)

# Predict button
if st.button("Predict Price"):
    # Create a DataFrame for the input
    input_data = pd.DataFrame({
        'size_sqft': [size_sqft],
        'num_bedrooms': [num_bedrooms],
        'num_bathrooms': [num_bathrooms],
        'age_years': [age_years],
        'location_score': [location_score]
    })

    # Scale the input data using the loaded scaler
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = loaded_model.predict(input_scaled)[0]

    # Display the prediction in a styled card
    with st.container():
        st.markdown(
            f"""
            <div style="
                background-color: #e6f2ff;  
                border-radius: 15px;
                padding: 20px;
                text-align: center;
                box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
                max-width: 700px;
                margin: auto;
            ">
                <h3 style="color: #333; margin-bottom: 10px;">Predicted House Price</h3>
                <h2 style="color: #0066cc; margin-top: 0;">${prediction:,.2f}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
