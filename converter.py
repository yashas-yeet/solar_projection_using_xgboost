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
#  For commercial licensing inquiries, contact: [Your Email Here]
#  Full terms available in the LICENSE file.
# ==============================================================================

import pandas as pd
import tkinter as tk
from tkinter import filedialog
import os

print("--- ROBUST SOLAR DATA CONVERTER ---")

# 1. Open File Dialog
root = tk.Tk(); root.withdraw()
print("Please select your downloaded CSV file...")
file_path = filedialog.askopenfilename(title="Select the Raw DKASC CSV", filetypes=[("CSV files", "*.csv")])

if file_path:
    print(f"Reading file: {os.path.basename(file_path)}...")
    
    # READ CSV - Skip the first few rows if they are metadata
    # We try reading normally first.
    try:
        df = pd.read_csv(file_path)
    except:
        # Fallback: sometimes these files have weird encoding
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
    
    print(f"Loaded {len(df):,} rows.")
    
    # 2. SMART COLUMN FINDER
    # We strip spaces from column names to fix " Timestamp" vs "Timestamp" issues
    df.columns = df.columns.str.strip()
    print("Columns found:", df.columns.tolist())
    
    clean_df = pd.DataFrame()

    # --- FIND THE RIGHT COLUMNS AUTOMATICALLY ---
    
    # A. Find Timestamp
    time_col = None
    for col in df.columns:
        if "timestamp" in col.lower() or "date" in col.lower() or "time" in col.lower():
            time_col = col
            break
            
    if time_col:
        print(f"Found Timestamp column: '{time_col}'")
        clean_df['Timestamp'] = pd.to_datetime(df[time_col], dayfirst=True, errors='coerce')
    else:
        print("ERROR: Could not find a Timestamp column!")
        exit()

    # B. Find GHI (Sunlight)
    ghi_col = None
    for col in df.columns:
        if "global" in col.lower() and "radiation" in col.lower():
            ghi_col = col
            break
            
    if ghi_col:
        print(f"Found GHI column: '{ghi_col}'")
        clean_df['GHI'] = df[ghi_col] / 1000.0 # Convert W to kW
    else:
        print("WARNING: Could not find GHI column. Looking for 'Irradiance'...")
        # Fallback search
        for col in df.columns:
            if "irradiance" in col.lower():
                ghi_col = col
                clean_df['GHI'] = df[ghi_col] / 1000.0
                break
        if not ghi_col: clean_df['GHI'] = 0 # Default to 0 if missing

    # C. Find Active Power (Output)
    pow_col = None
    for col in df.columns:
        if "active_power" in col.lower() or ("power" in col.lower() and "active" in col.lower()):
            pow_col = col
            break
            
    if pow_col:
        print(f"Found Power column: '{pow_col}'")
        clean_df['Actual_Output'] = df[pow_col]
    else:
        print("WARNING: Could not find 'Active_Power'. Using first column with 'Power'...")
        for col in df.columns:
            if "power" in col.lower():
                clean_df['Actual_Output'] = df[col]
                break

    # D. Find Temperature
    temp_col = None
    for col in df.columns:
        if "temperature" in col.lower() and "ambient" in col.lower(): # Prefer Ambient
            temp_col = col
            break
    if not temp_col:
        for col in df.columns:
             if "temperature" in col.lower():
                temp_col = col
                break
                
    if temp_col:
        print(f"Found Temp column: '{temp_col}'")
        clean_df['Temperature'] = df[temp_col]
    else:
        clean_df['Temperature'] = 25 # Default temp if missing

    # 3. CLEAN & SAVE
    clean_df.dropna(subset=['Timestamp', 'Actual_Output'], inplace=True)
    
    # Sort by time just in case
    clean_df.sort_values('Timestamp', inplace=True)
    
    output_filename = "Training_Data_Fixed.csv"
    clean_df.to_csv(output_filename, index=False)
    
    print("\n" + "="*40)
    print(f"SUCCESS! Saved as: {output_filename}")
    print(f"Rows: {len(clean_df):,}")
    print("="*40)