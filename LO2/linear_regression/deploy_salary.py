import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Streamlit app title
st.title("Salary Prediction App")
st.write("Predict salaries based on years of experience using a Linear Regression model.")

# Load or generate dataset
@st.cache
def load_data():
    try:
        # Try loading dataset from CSV
        df = pd.read_csv('salary_data.csv')
    except FileNotFoundError:
        # Generate synthetic dataset
        np.random.seed(42)
        n_samples = 300
        years_experience = np.random.randint(1, 21, n_samples)
        salaries = 40000 + years_experience * 2000 + np.random.normal(0, 5000, n_samples)
        df = pd.DataFrame({
            'years_of_experience': years_experience,
            'salary': salaries.round(2)
        })
    return df

df = load_data()

# Display dataset
if st.checkbox("Show Dataset"):
    st.write(df.head())

# Split dataset
X = df[['years_of_experience']]
y = df['salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Model coefficients
st.write("### Model Coefficients")
st.write(f"Intercept (β₀): {model.intercept_:.2f}")
st.write(f"Slope (β₁): {model.coef_[0]:.2f}")

# Evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write("### Model Performance")
st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"Root Mean Squared Error: {rmse:.2f}")
st.write(f"Mean Absolute Error: {mae:.2f}")
st.write(f"R-squared Score: {r2:.4f}")

# Visualization
st.write("### Data Visualization")
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Subplot 1: Training data with regression line
ax[0].scatter(X_train, y_train, color='blue', alpha=0.6, label='Training Data')
ax[0].plot(X_train, model.predict(X_train), color='red', linewidth=2, label='Regression Line')
ax[0].set_xlabel('Years of Experience')
ax[0].set_ylabel('Salary ($)')
ax[0].set_title('Linear Regression - Training Data')
ax[0].legend()
ax[0].grid(True, alpha=0.3)

# Subplot 2: Predictions vs Actual
ax[1].scatter(y_test, y_pred, color='green', alpha=0.6)
ax[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax[1].set_xlabel('Actual Salary ($)')
ax[1].set_ylabel('Predicted Salary ($)')
ax[1].set_title('Predictions vs Actual Values')
ax[1].grid(True, alpha=0.3)

st.pyplot(fig)

# Predict salaries for new data
st.write("### Predict Salaries for New Data")
new_experience = st.text_input("Enter years of experience (comma-separated, e.g., 6, 12, 15):", "6, 12, 15")
if st.button("Predict"):
    try:
        new_experience = np.array([[float(x)] for x in new_experience.split(",")])
        new_predictions = model.predict(new_experience)
        new_data_df = pd.DataFrame({
            'Years of Experience': new_experience.flatten(),
            'Predicted Salary ($)': new_predictions
        })
        new_data_df['Predicted Salary ($)'] = new_data_df['Predicted Salary ($)'].apply(lambda x: f"${x:,.2f}")
        st.write(new_data_df)
    except ValueError:
        st.error("Please enter valid numeric values for years of experience.")