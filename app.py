import streamlit as st
import pandas as pd
import numpy as np
import requests

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# --------------------------------
# PAGE CONFIG
# --------------------------------
st.set_page_config(
    page_title="Tea Garden Climate Impact Analyzer",
    page_icon="🍃",
    layout="wide"
)

st.title("🍃 Climate Impact Analyzer for Tea Gardens")
st.write("Early-warning and decision support system for tea plantation management")

# --------------------------------
# REGION SELECTION
# --------------------------------
st.sidebar.header("🌍 Select Tea Growing Region")

region = st.sidebar.selectbox(
    "Region",
    ["Assam", "Darjeeling", "Nilgiris"]
)

region_coords = {
    "Assam": (26.2006, 92.9376),
    "Darjeeling": (27.0410, 88.2663),
    "Nilgiris": (11.4064, 76.6932)
}

lat, lon = region_coords[region]

# --------------------------------
# LOAD DATA
# --------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("tea_garden_climate_data_500.csv")

df = load_data()

# --------------------------------
# ENCODE TARGETS
# --------------------------------
le_stress = LabelEncoder()
le_risk = LabelEncoder()

df["stress_enc"] = le_stress.fit_transform(df["stress_level"])
df["risk_enc"] = le_risk.fit_transform(df["risk_level"])

features = [
    "temperature_c",
    "rainfall_mm",
    "humidity_pct",
    "soil_moisture_pct",
    "heat_index"
]

X = df[features]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Logistic Regression models
stress_model = LogisticRegression(max_iter=500, random_state=42)
risk_model = LogisticRegression(max_iter=500, random_state=42)

stress_model.fit(X_scaled, df["stress_enc"])
risk_model.fit(X_scaled, df["risk_enc"])

# --------------------------------
# LIVE WEATHER (Real API)
# --------------------------------
st.sidebar.header("🌦 Live Weather")

API_KEY = "8d0bdfc2d7fe4af92beb0397c9c0797a"  # Replace with your API key

try:
    if API_KEY != "":
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        response = requests.get(url)
        if response.status_code == 200:
            weather = response.json()
            temperature_live = weather['main']['temp']
            humidity_live = weather['main']['humidity']
            description_live = weather['weather'][0]['description'].title()

            st.sidebar.write(f"🌡 Temperature: {temperature_live} °C")
            st.sidebar.write(f"💧 Humidity: {humidity_live} %")
            st.sidebar.write(f"☁ Weather: {description_live}")
        else:
            st.sidebar.warning("⚠ Unable to fetch live weather. Using default values.")
            temperature_live = 28.5
            humidity_live = 72
            description_live = "Partly Cloudy"
except:
    st.sidebar.warning("⚠ Error connecting to OpenWeather API. Using default values.")
    temperature_live = 28.5
    humidity_live = 72
    description_live = "Partly Cloudy"

# --------------------------------
# INPUT SECTION (Dynamic slider ranges)
# --------------------------------
st.header("📥 Climate Parameters")

col1, col2 = st.columns(2)

with col1:
    temperature = st.slider(
        "Temperature (°C)",
        float(df['temperature_c'].min()), float(df['temperature_c'].max()),
        float(temperature_live)
    )
    rainfall = st.slider(
        "Rainfall (mm)",
        float(df['rainfall_mm'].min()), float(df['rainfall_mm'].max()),
        10.0
    )

with col2:
    humidity = st.slider(
        "Humidity (%)",
        float(df['humidity_pct'].min()), float(df['humidity_pct'].max()),
        float(humidity_live)
    )
    soil_moisture = st.slider(
        "Soil Moisture (%)",
        float(df['soil_moisture_pct'].min()), float(df['soil_moisture_pct'].max()),
        45.0
    )

# Heat index calculation
heat_index = (temperature * humidity) / 100

# --------------------------------
# PREDICTION
# --------------------------------
input_df = pd.DataFrame({
    "temperature_c": [temperature],
    "rainfall_mm": [rainfall],
    "humidity_pct": [humidity],
    "soil_moisture_pct": [soil_moisture],
    "heat_index": [heat_index]
})

# Scale input
input_scaled = scaler.transform(input_df)

# Predict using Logistic Regression
stress_pred = stress_model.predict(input_scaled)
risk_pred = risk_model.predict(input_scaled)

stress_label = le_stress.inverse_transform(stress_pred)[0]
risk_label = le_risk.inverse_transform(risk_pred)[0]

# --------------------------------
# RESULTS
# --------------------------------
st.subheader("📊 Prediction Results")

st.metric("🌱 Stress Level", stress_label)
st.metric("⚠ Climate Risk Level", risk_label)

# --------------------------------
# RECOMMENDATIONS
# --------------------------------
st.subheader("🛠 Recommendations")

if stress_label == "High":
    st.error("🚨 Increase irrigation, apply shading nets, and monitor plants closely.")
elif stress_label == "Medium":
    st.warning("⚠ Moderate stress. Ensure adequate watering and frequent monitoring.")
else:
    st.success("✅ Low stress. Normal operations recommended.")

# --------------------------------
# FOOTER
# --------------------------------
st.markdown("---")
st.caption("Climate Impact Analyzer | Decision Support System for Tea Estates")
