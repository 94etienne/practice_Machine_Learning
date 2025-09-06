import streamlit as st
import numpy as np
import joblib

# Load the models
classifier = joblib.load('model/decision_tree_classifier.pkl')
regressor = joblib.load('model/decision_tree_regressor.pkl')

# Streamlit app
st.title("Student Performance Prediction")
st.write("This app predicts whether a student will pass or fail (classification) and their exact score (regression).")

# Input fields
st.header("Enter Student Details")
hours_studied = st.number_input("Hours Studied per Week", 0, 10, 5)
attendance = st.number_input("Attendance Percentage", 40, 100, 70)
sleep_hours = st.number_input("Sleep Hours per Night", 4, 10, 7)

# Predict button
if st.button("Predict"):
    # Prepare input data
    student_data = np.array([[hours_studied, attendance, sleep_hours]])

    # Make predictions
    grade_pred = classifier.predict(student_data)[0]
    score_pred = regressor.predict(student_data)[0]

# Display results in a responsive and attractive card
st.markdown(
    f"""
    <div style="
        background: linear-gradient(135deg, #4CAF50, #81C784);
        color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
        text-align: center;
        font-family: Arial, sans-serif;
    ">
        <h2 style="margin-bottom: 20px;">🎯 Prediction Results</h2>
        <p style="font-size: 18px; margin: 10px 0;">
            <strong>Predicted Grade:</strong> 
            <span style="font-size: 20px; font-weight: bold;">{'PASS' if grade_pred == 1 else 'FAIL'}</span>
        </p>
        <p style="font-size: 18px; margin: 10px 0;">
            <strong>Predicted Score:</strong> 
            <span style="font-size: 20px; font-weight: bold;">{score_pred:.2f}</span>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)