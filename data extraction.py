import requests
import pandas as pd

# Open-Meteo API endpoint for historical weather data
url = "https://archive-api.open-meteo.com/v1/archive"

# List of cities with their latitude and longitude
cities = {
    "Lagos": {"lat": 6.5244, "lon": 3.3792},
    "Port Harcourt": {"lat": 4.8156, "lon": 7.0498},
    "Kano": {"lat": 12.0022, "lon": 8.5927},
    "Abuja": {"lat": 9.0579, "lon": 7.4951},
    "Ibadan": {"lat": 7.3776, "lon": 3.9470},
    "Ota": {"lat": 6.6804, "lon": 3.2356},
}

# Date range for which to obtain historical weather data
start_date = "2022-06-01"
end_date = "2023-05-12"

# List to store weather data for each city
all_city_data = []

# Loop through cities and obtain weather data for each city
for city, coords in cities.items():
    lat, lon = coords["lat"], coords["lon"]
    
    # Parameters for API request
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["temperature_2m", "relative_humidity_2m", "windspeed_10m", "precipitation"],
        "timezone": "auto",
    }
    
    # Make API request
    response = requests.get(url, params=params)
    response.raise_for_status()  # Raise an error for bad status codes
    data = response.json()
    
    # Extract hourly data
    hourly_data = data.get("hourly", {})
    timestamps = hourly_data.get("time", [])
    temps = hourly_data.get("temperature_2m", [])
    humidities = hourly_data.get("relative_humidity_2m", [])
    windspeeds = hourly_data.get("windspeed_10m", [])
    precipitations = hourly_data.get("precipitation", [])
    
    # Combine data into a list of dictionaries
    city_data = [
        {
            "city": city,
            "date": timestamps[i],
            "temp_c": temps[i],
            "humidity": humidities[i],
            "wind_kmph": windspeeds[i],
            "precip_mm": precipitations[i],
        }
        for i in range(len(timestamps))
    ]
    
    # Append city data to overall list
    all_city_data.extend(city_data)

# Convert all city data to a Pandas DataFrame
weather_df = pd.DataFrame(all_city_data)
print(weather_df.head(10))

# Save DataFrame to CSV file
weather_df.to_csv("weather_data.csv", index=False)
