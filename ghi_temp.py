# ==============================================================================
#  Hybrid Solar Grid Output Forecaster
#  Copyright (c) 2025 Yashas Vishwakarma
#
#  DUAL LICENSING NOTICE:
#  This source code is protected by copyright law and is available under two
#  distinct licensing models. You may choose to use it under:
#
#  1. OPEN SOURCE (GPLv3):
#     Free for academic, personal, and open-source projects.
#     Condition: If you distribute software using this code, your ENTIRE
#     project must also be open-source under GPLv3.
#
#  2. COMMERCIAL LICENSE:
#     Required for proprietary (closed-source) commercial products.
#     Allows you to keep your source code private and provides legal support.
#
#  For commercial licensing inquiries, contact: [yashasakvish@gmail.com]
#  Full terms available in the LICENSE file.
# ==============================================================================


import pandas as pd
import numpy as np
import requests
import tkinter as tk
from tkinter import filedialog
import os

print("======================================================")
print("  AUTOMATED WEATHER BACKCASTER & DATA MERGER  ")
print("======================================================")

# 1. Select the incomplete CSV
root = tk.Tk()
root.withdraw()
print("\nPlease select your CSV file (Needs at least Timestamp and Power)...")
file_path = filedialog.askopenfilename(title="Select Power CSV", filetypes=[("CSV files", "*.csv")])

if not file_path:
    print("No file selected. Exiting.")
    exit()

print(f"Reading: {os.path.basename(file_path)}")

# 2. Get Location Data from User
print("\nWe need the location of the solar farm to fetch the historical weather.")
lat = input("Enter Latitude (e.g., 12.97): ").strip()
lon = input("Enter Longitude (e.g., 79.16): ").strip()

# 3. Load and Clean User Data
try:
    df_user = pd.read_csv(file_path)
except:
    df_user = pd.read_csv(file_path, encoding='ISO-8859-1')

df_user.columns = df_user.columns.str.strip()

# Smart Column Finder
time_col = next((col for col in df_user.columns if any(x in col.lower() for x in ["timestamp", "date", "time"])), None)
pow_col = next((col for col in df_user.columns if "power" in col.lower() or "output" in col.lower()), None)

if not time_col or not pow_col:
    print("\nERROR: Could not automatically find the Timestamp or Power columns.")
    print(f"Columns found: {df_user.columns.tolist()}")
    exit()

print(f"\nFound Time Column: '{time_col}'")
print(f"Found Power Column: '{pow_col}'")

# Format Time and Power
df_user['Timestamp'] = pd.to_datetime(df_user[time_col], dayfirst=True, errors='coerce')
df_user.dropna(subset=['Timestamp'], inplace=True)
df_user['Actual_Output'] = pd.to_numeric(df_user[pow_col], errors='coerce')

# Compress user data to strict Hourly averages to match the API
print("Compressing power data to hourly averages...")
df_user.set_index('Timestamp', inplace=True)
df_user = df_user.resample('h').mean()
df_user.dropna(subset=['Actual_Output'], inplace=True)
df_user.reset_index(inplace=True)

# 4. Determine Timeframe for API
start_date = df_user['Timestamp'].min().strftime('%Y-%m-%d')
end_date = df_user['Timestamp'].max().strftime('%Y-%m-%d')
print(f"\nDetected Data Range: {start_date} to {end_date}")

# 5. Fetch Open-Meteo Archive Data
print(f"Fetching historical GHI and Temperature from Open-Meteo...")
url = "https://archive-api.open-meteo.com/v1/archive"
p = {
    "latitude": lat, 
    "longitude": lon, 
    "start_date": start_date, 
    "end_date": end_date, 
    "hourly": "temperature_2m,shortwave_radiation", 
    "timezone": "auto"
}

try:
    r = requests.get(url, params=p)
    r.raise_for_status()
    d = r.json()['hourly']
    
    # Create Weather DataFrame
    df_weather = pd.DataFrame({
        'Timestamp': pd.to_datetime(d['time']), 
        'Temperature': d['temperature_2m'], 
        'GHI': np.array(d['shortwave_radiation']) / 1000.0  # Convert W to kW
    })
    
    # Remove timezone information to ensure smooth merging
    df_user['Timestamp'] = df_user['Timestamp'].dt.tz_localize(None)
    df_weather['Timestamp'] = df_weather['Timestamp'].dt.tz_localize(None)

    # 6. Merge Weather and Power Data
    print("Merging weather data with power data...")
    final_df = pd.merge(df_weather, df_user[['Timestamp', 'Actual_Output']], on='Timestamp', how='inner')
    
    # 7. Save Final Output
    output_filename = "Training_Data_Complete.csv"
    final_df.to_csv(output_filename, index=False)
    
    print("\n======================================================")
    print("  SUCCESS! PIPELINE COMPLETE  ")
    print("======================================================")
    print(f"File Saved: {output_filename}")
    print(f"Total Rows: {len(final_df):,}")
    print("Columns:    [Timestamp, Temperature, GHI, Actual_Output]")
    print("This file is now perfectly formatted to train your XGBoost AI.")

except requests.exceptions.HTTPError as err:
    print(f"\nAPI ERROR: {err}")
    print("The API rejected the request. Check your coordinates or date ranges.")
except Exception as e:
    print(f"\nUNEXPECTED ERROR: {e}")