# 🌦️ Weather Prediction System using Machine Learning

A web-based application that predicts daily weather conditions (Sunny, Cloudy, or Rainy) for a selected city and date. It utilizes real-time weather data from the Open-Meteo API and an ensemble machine learning model for accurate forecasting. The app also displays hourly weather parameters and interactive visualizations.

---

## 📌 Features

- 📍 City and date selection for weather prediction
- ⏱️ Hourly weather forecast including:
  - Temperature (°C)
  - Humidity (%)
  - Wind Speed (km/h)
  - Precipitation (mm)
- 🔮 Weather condition prediction using a trained machine learning model
- 📊 Separate bar graphs for each weather parameter
- ⚙️ Built with Streamlit for an interactive user interface

---

## 🧠 Technologies & Libraries Used

| Purpose              | Technology / Library                |
|----------------------|-------------------------------------|
| Programming Language | Python                              |
| Web Framework        | Streamlit                           |
| Machine Learning     | Scikit-learn (Ensemble Model), Joblib |
| Data Handling        | Pandas, NumPy                       |
| Visualization        | Matplotlib, Seaborn                 |
| API Integration      | Open-Meteo API                      |
| Feature Engineering  | Cyclical Encoding (sin/cos)         |

---

## 🔗 API Reference

- **Open-Meteo API**  
  URL: [https://open-meteo.com/](https://open-meteo.com/)  
  Used to retrieve hourly weather data for temperature, humidity, wind speed, and precipitation.

---

## ⚙️ Project Structure & Execution Steps

### ✅ 1. Data Extraction

- **Script:** `data_extraction.py`
- **Purpose:** Fetch weather data using Open-Meteo API
- **Output:** `weather_data.csv`

🔧 **2. Data Preprocessing**

- **Script:** data_processing.py`  
- **Purpose:** Clean, format, and generate training features  
- **Output:** `Training_data.csv`

### 🧠 3. Model Training

**Script:** `3_model_building.py`  
**Purpose:** Train multiple models and save the best ensemble model  
**Output Models:**
- `SVC_model.joblib`
- `KNN_model.joblib`
- `AdaBoost_model.joblib`
- `XGB_model.joblib`
- `Ensemble_model.joblib`

### 🔄 4. Select ML Model for Deployment

**Edit** the `app.py` file and **load your desired model**, for example:
```python
model = joblib.load("models/Ensemble_model.joblib")

### 🌐 5. Build Web App
**Script:** `app.py`  
**Purpose:** Create a Streamlit-based web application where users can:
- 📍 Select a city and date
- ⏱️ View hourly weather data from Open-Meteo API
- 📊 Visualize parameters (temperature, humidity, wind speed, precipitation)
- 🔮 Get predicted weather condition (Sunny, Cloudy, Rainy) using the selected machine learning model

### 🚀 6. Run the Application

**Steps:**
1. Open a terminal or command prompt  
2. Navigate to your project directory  
3. Run the Streamlit application using the following command:
```bash
streamlit run app.py

📊 Sample Output
Predicted weather condition: Sunny, Cloudy, or Rainy
Hourly weather forecast displayed in a table
Interactive bar graphs for each parameter











