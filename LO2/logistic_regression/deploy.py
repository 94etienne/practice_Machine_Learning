import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the saved model and scaler
model = joblib.load("model/logistic_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# Streamlit app
st.title("Logistic Regression Prediction App")
st.write("Enter the input features to predict the target outcome.")

# Input fields
fare = st.text_input("Fare", "7.25")
age = st.text_input("Age", "22")
sex = st.selectbox("Sex", ["male", "female"])
family_size = st.text_input("Family Size", "1")
sibsp = st.text_input("Siblings/Spouses Aboard", "0")
parch = st.text_input("Parents/Children Aboard", "0")
embarked_c = st.selectbox("Embarked_C (1 for Yes, 0 for No)", [1, 0])
embarked_s = st.selectbox("Embarked_S (1 for Yes, 0 for No)", [1, 0])

# Predict button
if st.button("Predict"):
    try:
        # Validate inputs
        if not fare.replace('.', '', 1).isdigit() or not age.replace('.', '', 1).isdigit():
            st.error("Fare and Age must be numeric values.")
        elif not family_size.isdigit() or not sibsp.isdigit() or not parch.isdigit():
            st.error("Family Size, Siblings/Spouses, and Parents/Children must be integers.")
        else:
            # Convert numeric inputs to float/int
            fare = float(fare)
            age = float(age)
            family_size = int(family_size)
            sibsp = int(sibsp)
            parch = int(parch)

            # Create a DataFrame for the input
            input_data = pd.DataFrame({
                'Fare': [fare],
                'Age': [age],
                'Sex': [1 if sex == "male" else 0],  # Encode 'Sex' manually
                'FamilySize': [family_size],
                'SibSp': [sibsp],
                'Parch': [parch],
                'Embarked_C': [embarked_c],
                'Embarked_S': [embarked_s],
                # Add missing one-hot encoded columns with default values
                'Title_Mr': [1 if sex == "male" else 0],
                'Title_Miss': [1 if sex == "female" else 0],
                'Title_Mrs': [0],
                'Sex_Pclass_male_3': [0],
                'Sex_Pclass_female_2': [0],
                'Sex_Pclass_female_3': [0],
                'Pclass_3': [0],
                'AgeGroup_Middle_Age': [0],
                'Sex_Pclass_male_2': [0]
            })

            # Scale numerical features
            input_scaled = scaler.transform(input_data)

            # Make prediction
            prediction = model.predict(input_scaled)
            prediction_prob = model.predict_proba(input_scaled)

            # Display result
            st.write(f"Prediction: **{'Survived' if prediction[0] == 1 else 'Not Survived'}**")
            st.write(f"Probability of Not Survived: {prediction_prob[0][0]:.2f}")
            st.write(f"Probability of Survived: {prediction_prob[0][1]:.2f}")
    except ValueError:
        st.error("An error occurred. Please check your inputs.")