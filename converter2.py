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

print("--- SOLAR DATA OPTIMIZER (5-min to Hourly) ---")

# 1. Select the file
root = tk.Tk(); root.withdraw()
print("Select 'Training_Data_Fixed.csv'...")
file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])

if file_path:
    print("Reading large file (this may take a moment)...")
    df = pd.read_csv(file_path)
    
    # 2. Convert Timestamp
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # 3. Resample to HOURLY averages
    # This takes the average of the 12 readings per hour (e.g., 10:00, 10:05... 10:55)
    print("Converting 5-minute data to Hourly averages...")
    df.set_index('Timestamp', inplace=True)
    df_hourly = df.resample('h').mean()
    
    # 4. Clean Missing Data (NaNs)
    # The resampling might leave gaps, so we drop them to prevent training errors
    before = len(df_hourly)
    df_hourly.dropna(inplace=True)
    after = len(df_hourly)
    print(f"Removed {before - after} empty rows.")
    
    # 5. Reset index to get 'Timestamp' back as a column
    df_hourly.reset_index(inplace=True)
    
    # 6. Save
    output = "Training_Data_Hourly.csv"
    df_hourly.to_csv(output, index=False)
    
    print("\n" + "="*40)
    print(f"DONE! Saved as: {output}")
    print(f"Original Rows: {len(df):,}")
    print(f"New Rows:      {len(df_hourly):,}")
    print("="*40)
    print("Use this new file in your App.")