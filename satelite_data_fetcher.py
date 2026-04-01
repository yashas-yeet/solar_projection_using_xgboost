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
lat = input("Enter Latitude (e.g., -23.76 for Alice Springs): ").strip()
lon = input("Enter Longitude (e.g., 133.87): ").strip()

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

# --- FIX #1: THE DATE CORRUPTION BUG ---
try:
    df_user['Timestamp'] = pd.to_datetime(df_user[time_col], format='mixed', errors='coerce')
except ValueError:
    df_user['Timestamp'] = pd.to_datetime(df_user[time_col], infer_datetime_format=True, errors='coerce')

df_user.dropna(subset=['Timestamp'], inplace=True)
df_user['Actual_Output'] = pd.to_numeric(df_user[pow_col], errors='coerce')

# Compress user data to strict Hourly averages to match the API
print("Compressing power data to hourly averages...")
df_user.set_index('Timestamp', inplace=True)

# THE FIX: Only calculate the mean for the Actual_Output column, drop text columns
df_user = df_user[['Actual_Output']].resample('1h').mean() 

df_user.dropna(subset=['Actual_Output'], inplace=True)
df_user.reset_index(inplace=True)

# 4. Determine Timeframe for API
start_date = df_user['Timestamp'].min().strftime('%Y-%m-%d')
end_date = df_user['Timestamp'].max().strftime('%Y-%m-%d')
print(f"\nDetected Data Range: {start_date} to {end_date}")

# 5. Fetch Open-Meteo Archive Data (UPGRADED PAYLOAD)
print(f"Fetching advanced historical irradiance from Open-Meteo...")
url = "https://archive-api.open-meteo.com/v1/archive"
p = {
    "latitude": lat, 
    "longitude": lon, 
    "start_date": start_date, 
    "end_date": end_date, 
    "hourly": "temperature_2m,shortwave_radiation,direct_radiation,diffuse_radiation,cloudcover", 
    "timezone": "auto"
}

try:
    r = requests.get(url, params=p)
    r.raise_for_status()
    d = r.json()['hourly']
    
    # Create Weather DataFrame with Advanced Features
    df_weather = pd.DataFrame({
        'Timestamp': pd.to_datetime(d['time']), 
        'Temperature': d['temperature_2m'], 
        'GHI': np.array(d['shortwave_radiation']) / 1000.0,  
        'DNI': np.array(d['direct_radiation']) / 1000.0,     # Direct Beam (Clear Sky)
        'DHI': np.array(d['diffuse_radiation']) / 1000.0,    # Scattered Light (Clouds/Dust)
        'CloudCover': d['cloudcover']                        # % of sky covered
    })
    
    # Remove timezone information to ensure smooth merging
    df_user['Timestamp'] = df_user['Timestamp'].dt.tz_localize(None)
    df_weather['Timestamp'] = df_weather['Timestamp'].dt.tz_localize(None)

    # 6. Merge Weather and Power Data
    print("Merging weather data with power data...")
    final_df = pd.merge(df_weather, df_user[['Timestamp', 'Actual_Output']], on='Timestamp', how='inner')
    
    # --- FIX #2: THE AUTO-ALIGNER ---
    print("\nRunning Timezone Auto-Aligner (Cross-Correlation)...")
    best_shift = 0
    best_corr = -1
    
    # Test shifting the data forwards and backwards by up to 14 hours
    for shift in range(-14, 15):
        corr = final_df['GHI'].corr(final_df['Actual_Output'].shift(shift))
        if corr > best_corr:
            best_corr = corr
            best_shift = shift
            
    # Apply the optimal shift to perfectly sync the satellite with the ground sensors
    if best_shift != 0:
        print(f"-> Time-Shift Detected! Re-aligning panel data by {best_shift} hours to match Satellite GHI.")
        final_df['Actual_Output'] = final_df['Actual_Output'].shift(best_shift)
    else:
        print("-> Data is already perfectly aligned with Satellite GHI (0 hour shift).")
        
    final_df.dropna(inplace=True) # Drop the empty edge rows created by the shift

    # --- FIX #3: THE OUTAGE / NIGHTTIME CLEANER ---
    print("\nRunning Aggressive Data Cleaner...")
    initial_rows = len(final_df)
    
    # 1. Drop Nighttime "ghost" data (AI doesn't need to learn how to predict night)
    final_df = final_df[final_df['GHI'] > 0.05].copy()
    
    # 2. Drop "Inverter Dead" Outages (High Sun, 0 Power)
    peak_power = final_df['Actual_Output'].quantile(0.95)
    outage_mask = (final_df['GHI'] > 0.4) & (final_df['Actual_Output'] < (peak_power * 0.1))
    final_df = final_df[~outage_mask].copy()
    
    print(f"-> Cleaned out {initial_rows - len(final_df):,} dead-panel / nighttime rows.")

    # 7. Save Final Output
    output_filename = "Training_Data_Complete.csv"
    final_df.to_csv(output_filename, index=False)
    
    print("\n======================================================")
    print("  SUCCESS! PIPELINE COMPLETE  ")
    print("======================================================")
    print(f"File Saved: {output_filename}")
    print(f"Total Rows: {len(final_df):,}")
    print("Columns:    [Timestamp, Temperature, GHI, DNI, DHI, CloudCover, Actual_Output]")
    print(f"Final AI Correlation (R-value): {best_corr:.4f}")
    print("This file is now perfectly formatted to train your XGBoost AI.")

except requests.exceptions.HTTPError as err:
    print(f"\nAPI ERROR: {err}")
    print("The API rejected the request. Check your coordinates or date ranges.")
except Exception as e:
    print(f"\nUNEXPECTED ERROR: {e}")