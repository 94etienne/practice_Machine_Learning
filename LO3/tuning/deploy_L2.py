import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
ridge_model = joblib.load('model/ridge_model.pkl')
scaler = joblib.load('model/scaler.pkl')

st.title("House Price Prediction App")

st.header("Enter House Details:")
size_sqft = st.number_input("Size (in sqft):", min_value=800, max_value=4000, value=1500, step=50)
num_bedrooms = st.selectbox("Number of Bedrooms:", [1, 2, 3, 4, 5])
num_bathrooms = st.selectbox("Number of Bathrooms:", [1, 2, 3, 4])
age_years = st.number_input("Age of the House (years):", min_value=0, max_value=50, value=10)
location_score = st.number_input("Location Score (1-10):", min_value=1, max_value=10, value=5)

if st.button("Predict Price"):
    input_data = pd.DataFrame({
        'size_sqft': [size_sqft],
        'num_bedrooms': [num_bedrooms],
        'num_bathrooms': [num_bathrooms],
        'age_years': [age_years],
        'location_score': [location_score]
    })

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = ridge_model.predict(input_scaled)[0]

    # Display in a card
    st.markdown(
        f"""
        <div style="
            background-color: #e6f2ff;
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
            max-width: 600px;
            margin: auto;
        ">
            <h3>Predicted House Price</h3>
            <h2 style="color: #0066cc;">${prediction:,.2f}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
