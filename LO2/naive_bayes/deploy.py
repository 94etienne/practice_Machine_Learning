import streamlit as st
import pandas as pd
import joblib

# Load the saved model, scaler, and encoders
model = joblib.load("model/tennis_model.joblib")
scaler = joblib.load("model/scaler.joblib")
label_encoders = joblib.load("model/label_encoders.joblib")

# Streamlit app
st.title("Play Tennis Prediction App")
st.write("Enter the weather conditions to predict if you can play tennis.")

# Input fields
outlook = st.selectbox("Outlook", label_encoders['outlook'].classes_)
temperature = st.text_input("Temperature (°F)")
humidity = st.text_input("Humidity (%)")
windy = st.selectbox("Windy", [True, False])

# Predict button
if st.button("Predict"):
    try:
        # Convert temperature and humidity to numeric values
        temperature = float(temperature)
        humidity = float(humidity)

        # Create a DataFrame for the input
        input_data = pd.DataFrame({
            'outlook': [outlook],
            'temperature': [temperature],
            'humidity': [humidity],
            'windy': [windy]
        })

        # Encode categorical features
        for col in ['outlook', 'windy']:
            input_data[col] = label_encoders[col].transform(input_data[col])

        # Scale numerical features
        input_data[['temperature', 'humidity']] = scaler.transform(input_data[['temperature', 'humidity']])

        # Make prediction
        prediction = model.predict(input_data)
        prediction_label = label_encoders['play_tennis'].inverse_transform(prediction)[0]

        # Display result
        st.write(f"Prediction: **{prediction_label}**")
    except ValueError:
        st.error("Please enter valid numeric values for Temperature and Humidity.")