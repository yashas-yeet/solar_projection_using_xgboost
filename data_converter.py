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
import tkinter as tk
from tkinter import filedialog
import os

print("--- SOLAR DATA PIPELINE (Raw -> Hourly) ---")

# 1. Open File Dialog
root = tk.Tk()
root.withdraw()
print("Please select your downloaded Raw DKASC CSV file...")
file_path = filedialog.askopenfilename(title="Select Raw CSV", filetypes=[("CSV files", "*.csv")])

if not file_path:
    print("No file selected. Exiting.")
    exit()

print(f"\nReading file: {os.path.basename(file_path)}...")

# 2. READ CSV
try:
    df = pd.read_csv(file_path)
except:
    df = pd.read_csv(file_path, encoding='ISO-8859-1')

df.columns = df.columns.str.strip()
clean_df = pd.DataFrame()

# --- 3. SMART COLUMN FINDER ---
# Find Timestamp
time_col = next((col for col in df.columns if any(x in col.lower() for x in ["timestamp", "date", "time"])), None)
if time_col:
    clean_df['Timestamp'] = pd.to_datetime(df[time_col], dayfirst=True, errors='coerce')
else:
    print("ERROR: Could not find a Timestamp column!"); exit()

# Find GHI
ghi_col = next((col for col in df.columns if "global" in col.lower() and "radiation" in col.lower()), None)
if not ghi_col: ghi_col = next((col for col in df.columns if "irradiance" in col.lower()), None)
clean_df['GHI'] = (df[ghi_col] / 1000.0) if ghi_col else 0 

# Find Power
pow_col = next((col for col in df.columns if "active_power" in col.lower() or ("power" in col.lower() and "active" in col.lower())), None)
if not pow_col: pow_col = next((col for col in df.columns if "power" in col.lower()), None)
if pow_col: clean_df['Actual_Output'] = df[pow_col]

# Find Temperature
temp_col = next((col for col in df.columns if "temperature" in col.lower() and "ambient" in col.lower()), None)
if not temp_col: temp_col = next((col for col in df.columns if "temperature" in col.lower()), None)
clean_df['Temperature'] = df[temp_col] if temp_col else 25 

# --- 4. CLEAN & COMPRESS TO HOURLY ---
print("\nCleaning data and compressing to Hourly Averages...")
clean_df.dropna(subset=['Timestamp', 'Actual_Output'], inplace=True)
clean_df.sort_values('Timestamp', inplace=True)

clean_df.set_index('Timestamp', inplace=True)
df_hourly = clean_df.resample('h').mean()
df_hourly.dropna(inplace=True)
df_hourly.reset_index(inplace=True)

# --- 5. SAVE ---
hourly_filename = "Training_Data_Hourly.csv"
df_hourly.to_csv(hourly_filename, index=False)

print("\n" + "="*50)
print("PIPELINE COMPLETE!")
print(f" -> SAVED: {hourly_filename} ({len(df_hourly):,} rows)")
print("="*50)