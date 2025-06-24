import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
data = pd.read_csv("weather_data.csv")

# Display basic info and summary
data.info()
print(data.describe())

# Checking available columns
print("Available columns:", data.columns)

# Checking number of cities in dataset
print("Cities in dataset:", data["city"].nunique())
print(data["city"].value_counts())

# Standardizing column names (removing spaces if any)
data.columns = data.columns.str.strip()

# Checking available weather conditions
weather_col = None
for col in data.columns:
    if "weather" in col.lower() and "condition" in col.lower():
        weather_col = col
        break

if weather_col:
    print("Weather Conditions in dataset:")
    print(data[weather_col].value_counts())
    # Handling missing values
data.fillna(method='ffill', inplace=True)  # Forward fill missing values

# Label Encoding categorical features
label_encoder = LabelEncoder()
data['city_encoded'] = label_encoder.fit_transform(data['city'])

if weather_col:
    data['condition_encoded'] = label_encoder.fit_transform(data[weather_col])

# Convert date column to datetime format
data['date'] = pd.to_datetime(data['date'])

# Extract useful time-based features
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data['hour'] = data['date'].dt.hour

# Perform cyclical encoding for month and hour
data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)

# Columns to scale (check for existence before scaling)
features_to_scale = ['pressure_mb', 'cloud_cover_%', 'dew_point_c', 'uv_index',
                     'visibility_km', 'wind_gust_kph', 'humidity', 'precip_mm',
                     'temp_c', 'wind_kmph']

# Remove any missing columns from scaling list
features_to_scale = [col for col in features_to_scale if col in data.columns]

if features_to_scale:
    scaler = StandardScaler()
    data[features_to_scale] = scaler.fit_transform(data[features_to_scale])
else:
    print("Warning: No valid features found for scaling.")

# Drop unnecessary columns before training
drop_columns = ['city', 'date']
if weather_col:
    drop_columns.append(weather_col)
training_data = data.drop(drop_columns, axis=1)

# Save preprocessed dataset
training_data.to_csv('Training_data.csv', index=False)
print("Preprocessed data saved as 'Training_data.csv'")
