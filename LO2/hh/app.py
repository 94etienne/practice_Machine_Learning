import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import OneHotEncoder

# Load the trained model
model = joblib.load("best_model.pkl")

# Define categorical options (from the model)
LOCATIONS = ["Huye", "Ngoma", "Rukira", "Tumba"]
AMENITIES = ["Hospital, University", "Market, University", "School, Hospital", "School, Market"]
ZONING_TYPES = [
    "Agricultural - Agricultural", "Agricultural - Commercial", "Agricultural - Residential",
    "Commercial - Agricultural", "Commercial - Commercial", "Commercial - Residential",
    "Residential - Agricultural", "Residential - Commercial", "Residential - Residential"
]

# Mock data for visualization (replace with your actual data if available)
def generate_mock_data():
    np.random.seed(42)
    sizes = np.random.randint(50, 300, 100)
    distances = np.random.uniform(1, 20, 100)
    locations = np.random.choice(LOCATIONS, 100)
    amenities = np.random.choice(AMENITIES, 100)
    zoning = np.random.choice(ZONING_TYPES, 100)
    prices = 100000 + sizes * 500 - distances * 2000  # Simulated pricing formula
    return pd.DataFrame({
        "Size (sqm)": sizes,
        "Distance to City Center (km)": distances,
        "Location": locations,
        "Nearby Amenities": amenities,
        "Zoning_LandType": zoning,
        "Predicted Price": prices
    })

df = generate_mock_data()

# Dashboard UI
st.title("🏠 Real Estate Price Predictor")
st.markdown("Predict property prices based on size, location, and amenities.")

# Sidebar for user input
st.sidebar.header("Property Details")
size = st.sidebar.number_input("Size (sqm)", min_value=30, max_value=500, value=100)
distance = st.sidebar.number_input("Distance to City Center (km)", min_value=0.1, max_value=50.0, value=5.0)
location = st.sidebar.selectbox("Location", LOCATIONS)
amenities = st.sidebar.selectbox("Nearby Amenities", AMENITIES)
zoning = st.sidebar.selectbox("Zoning/Land Type", ZONING_TYPES)

# Predict button
if st.sidebar.button("Predict Price"):
    input_data = pd.DataFrame({
        "Size (sqm)": [size],
        "Distance to City Center (km)": [distance],
        "Location": [location],
        "Nearby Amenities": [amenities],
        "Zoning_LandType": [zoning]
    })
    
    # Preprocess and predict
    prediction = model.predict(input_data)[0]
    st.success(f"### Predicted Price: **${prediction:,.2f}**")

# Visualizations
st.header("📊 Data Insights")

# 1. Distribution of Numerical Features
st.subheader("1. Distribution of Size and Distance")
col1, col2 = st.columns(2)
with col1:
    fig_size = px.histogram(df, x="Size (sqm)", nbins=20, title="Size Distribution")
    st.plotly_chart(fig_size, use_container_width=True)
with col2:
    fig_dist = px.histogram(df, x="Distance to City Center (km)", nbins=20, title="Distance Distribution")
    st.plotly_chart(fig_dist, use_container_width=True)

# 2. Impact of Categorical Features on Price
st.subheader("2. Impact of Location and Amenities on Price")
fig_location = px.box(df, x="Location", y="Predicted Price", title="Price by Location")
st.plotly_chart(fig_location, use_container_width=True)

fig_amenities = px.box(df, x="Nearby Amenities", y="Predicted Price", title="Price by Amenities")
st.plotly_chart(fig_amenities, use_container_width=True)

# 3. Feature Importance (if XGBoost)
st.subheader("3. Feature Importance")
try:
    feature_importance = model.named_steps["regressor"].feature_importances_
    features = ["Size", "Distance", *model.named_steps["preprocessor"].transformers_[1][1].get_feature_names_out()]
    fig_importance = px.bar(
        x=features,
        y=feature_importance,
        labels={"x": "Feature", "y": "Importance"},
        title="XGBoost Feature Importance"
    )
    st.plotly_chart(fig_importance, use_container_width=True)
except:
    st.warning("Feature importance not available for this model.")