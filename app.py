import streamlit as st
import numpy as np
import pandas as pd
import joblib
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Load Ensemble Model
model = None
try:
    model = joblib.load("Ensemble_model.joblib")  # Load trained Ensemble model
except Exception as e:
    st.warning(f"⚠️ Could not load Ensemble model: {e}")

# Open-Meteo API for weather data
API_URL = "https://api.open-meteo.com/v1/forecast"

# Cities with coordinates
cities = {
    "Lagos": {"lat": 6.5244, "lon": 3.3792},
    "Port Harcourt": {"lat": 4.8156, "lon": 7.0498},
    "Kano": {"lat": 12.0022, "lon": 8.5927},
    "Abuja": {"lat": 9.0579, "lon": 7.4951},
    "Ibadan": {"lat": 7.3776, "lon": 3.9470},
    "Ota": {"lat": 6.6804, "lon": 3.2356},
}

# Map numbers to weather conditions
weather_conditions = {0: "Sunny", 1: "Cloudy", 2: "Rainy"}

# Streamlit UI
st.title("🌦️ Weather Prediction System")
st.sidebar.header("User Input")

# User selects city and date
city = st.sidebar.selectbox("Select a City", list(cities.keys()))
date = st.sidebar.date_input("Select a Date", datetime.today() + timedelta(days=1))

# Get latitude & longitude
lat, lon = cities[city]["lat"], cities[city]["lon"]

# Fetch weather data for selected date and 7-day forecast
start_date = datetime.today().strftime("%Y-%m-%d")
end_date = (datetime.today() + timedelta(days=6)).strftime("%Y-%m-%d")

params = {
    "latitude": lat,
    "longitude": lon,
    "hourly": ["temperature_2m", "relative_humidity_2m", "windspeed_10m", "precipitation"],
    "daily": ["temperature_2m_max", "temperature_2m_min", "relative_humidity_2m_mean", "windspeed_10m_max", "precipitation_sum"],
    "timezone": "auto",
    "start_date": date.strftime("%Y-%m-%d"),
    "end_date": end_date,
}
response = requests.get(API_URL, params=params)

if response.status_code == 200:
    data = response.json()
    daily_data = data.get("daily", {})
    hourly_data = data.get("hourly", {})

    # === DISPLAY HOURLY WEATHER FORECAST FOR SELECTED DATE === #
    if hourly_data:
        df_hourly = pd.DataFrame({
            "Time": pd.to_datetime(hourly_data["time"]).strftime("%Y-%m-%d %H:%M"),
            "Temperature (°C)": hourly_data["temperature_2m"],
            "Humidity (%)": hourly_data["relative_humidity_2m"],
            "Wind Speed (km/h)": hourly_data["windspeed_10m"],
            "Precipitation (mm)": hourly_data["precipitation"]
        })

        st.subheader(f"📍 {city} Hourly Weather Data for {date.strftime('%Y-%m-%d')}")
        st.dataframe(df_hourly)

        # Machine Learning Prediction (Daily Average)
        temp_c = np.mean(hourly_data["temperature_2m"])
        humidity = np.mean(hourly_data["relative_humidity_2m"])
        wind_kmph = np.mean(hourly_data["windspeed_10m"])
        precip_mm = np.mean(hourly_data["precipitation"])

        # Generate missing features
        selected_datetime = datetime.combine(date, datetime.min.time())  # Use midnight time
        day = selected_datetime.day
        month = selected_datetime.month
        hour = selected_datetime.hour  # Always 0 since we use midnight

        # Apply cyclical encoding
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)

        # Prepare input data with 11 features (to match training data)
        input_features = np.array([temp_c, humidity, wind_kmph, precip_mm, 
                                   day, hour, hour_sin, hour_cos, 
                                   month, month_sin, month_cos]).reshape(1, -1)

        # Make Prediction
        if model:
            try:
                prediction = model.predict(input_features)[0]
                weather_label = weather_conditions.get(prediction, "Unknown Condition")
                st.subheader("📊 Machine Learning Prediction")
                st.success(f"🔮 **Predicted Weather Condition for {date.strftime('%Y-%m-%d')}: {weather_label}**")
            except Exception as e:
                st.error(f"⚠️ Prediction Error: {e}")
        else:
            st.warning("⚠️ No model available for prediction.")

    else:
        st.error("⚠️ No hourly weather data found for the selected date.")

    # === DISPLAY DAILY WEATHER FORECAST (7 DAYS) === #
    if daily_data:
        df_daily = pd.DataFrame({
            "Date": pd.to_datetime(daily_data["time"]).strftime("%Y-%m-%d"),
            "Max Temperature (°C)": daily_data["temperature_2m_max"],
            "Min Temperature (°C)": daily_data["temperature_2m_min"],
            "Humidity (%)": daily_data["relative_humidity_2m_mean"],
            "Wind Speed (km/h)": daily_data["windspeed_10m_max"],
            "Precipitation (mm)": daily_data["precipitation_sum"]
        })

        st.subheader(f"📍 {city} 7-Day Weather Forecast")
        st.dataframe(df_daily)

        # === VISUALIZE DAILY WEATHER PARAMETERS === #
        st.subheader("📉 Weather Trends for the Next 7 Days")

        # Set seaborn style
        sns.set_style("whitegrid")

        # Plot Max & Min Temperature Graph
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=df_daily["Date"], y=df_daily["Max Temperature (°C)"], color="#ffcc00", label="Max Temp", ax=ax)
        sns.barplot(x=df_daily["Date"], y=df_daily["Min Temperature (°C)"], color="#ff6666", label="Min Temp", ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_title("Max & Min Temperature Over 7 Days")
        ax.set_xlabel("Date")
        ax.set_ylabel("Temperature (°C)")
        ax.legend()
        st.pyplot(fig)

        # Plot Humidity Graph
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=df_daily["Date"], y=df_daily["Humidity (%)"], color="#3399ff", ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_title("Humidity Over 7 Days")
        ax.set_xlabel("Date")
        ax.set_ylabel("Humidity (%)")
        st.pyplot(fig)

        # Plot Wind Speed Graph
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=df_daily["Date"], y=df_daily["Wind Speed (km/h)"], color="#33cc33", ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_title("Wind Speed Over 7 Days")
        ax.set_xlabel("Date")
        ax.set_ylabel("Wind Speed (km/h)")
        st.pyplot(fig)

        # Plot Precipitation Graph
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=df_daily["Date"], y=df_daily["Precipitation (mm)"], color="#0066cc", ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_title("Precipitation Over 7 Days")
        ax.set_xlabel("Date")
        ax.set_ylabel("Precipitation (mm)")
        st.pyplot(fig)

else:
    st.error("⚠️ Failed to fetch weather data.")
