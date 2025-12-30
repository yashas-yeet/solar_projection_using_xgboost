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

import customtkinter as ctk
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import requests
import datetime
import os
import sys
import threading
import joblib  # Required for Saving/Loading the AI Model

# =============================================================================
#   SYSTEM CONFIGURATION & GLOBAL STYLING
# =============================================================================
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# --- Typography Constants ---
FONT_MAIN = ("Bahnschrift", 14)
FONT_BOLD = ("Bahnschrift", 14, "bold")
FONT_HEADER = ("Bahnschrift", 20, "bold")
FONT_MONO = ("Consolas", 12)

# =============================================================================
#   HELPER CLASS: MODERN DATE SELECTOR
#   (Manages the complex dropdown logic for calendar dates)
# =============================================================================
class DateSelector(ctk.CTkFrame):
    def __init__(self, parent, default_date=None):
        super().__init__(parent, fg_color="transparent")
        
        if default_date is None:
            default_date = datetime.date.today()

        # --- Arrays for dropdown values ---
        self.days = [str(i).zfill(2) for i in range(1, 32)]
        self.months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        
        # Extended range for historical data (2000 - 2031)
        self.years = [str(i) for i in range(2000, 2031)]

        # --- Day Dropdown ---
        self.combo_day = ctk.CTkComboBox(self, values=self.days, width=60, font=("Bahnschrift", 12))
        self.combo_day.set(str(default_date.day).zfill(2))
        self.combo_day.pack(side="left", padx=(0, 5))

        # --- Month Dropdown ---
        self.combo_month = ctk.CTkComboBox(self, values=self.months, width=70, font=("Bahnschrift", 12))
        self.combo_month.set(self.months[default_date.month - 1])
        self.combo_month.pack(side="left", padx=(0, 5))

        # --- Year Dropdown ---
        self.combo_year = ctk.CTkComboBox(self, values=self.years, width=80, font=("Bahnschrift", 12))
        self.combo_year.set(str(default_date.year))
        self.combo_year.pack(side="left")

    def get_date(self):
        """
        Reconstructs the date object from the dropdowns safely.
        Returns None if the date is invalid (e.g., Feb 31st).
        """
        try:
            d = int(self.combo_day.get())
            m_str = self.combo_month.get()
            m = self.months.index(m_str) + 1
            y = int(self.combo_year.get())
            return datetime.date(y, m, d)
        except ValueError:
            return None

# =============================================================================
#   WINDOW: SETTINGS
#   (Controls AI Complexity and Physical Scaling - No Financials)
# =============================================================================
class SettingsWindow(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title("System Configuration & AI Settings")
        self.geometry("500x550")
        self.attributes("-topmost", True)
        
        # --- Section 1: AI Model Complexity ---
        lbl_ai = ctk.CTkLabel(self, text="AI Model Complexity (Trees)", font=("Bahnschrift", 18, "bold"), text_color="#3498db")
        lbl_ai.pack(pady=(20, 5))
        
        self.lbl_trees = ctk.CTkLabel(self, text=f"{parent.n_estimators} Trees", font=("Bahnschrift", 14))
        self.lbl_trees.pack(pady=(0, 10))
        
        self.slider_trees = ctk.CTkSlider(self, from_=10, to=500, number_of_steps=49, command=self.update_tree_label)
        self.slider_trees.set(parent.n_estimators)
        self.slider_trees.pack(pady=5, padx=40, fill="x")
        
        hint_ai = ctk.CTkLabel(self, text="Higher = More Accuracy, Slower Speed", text_color="gray", font=("Consolas", 10))
        hint_ai.pack(pady=(0, 20))

        # --- Section 2: Physical System Scaling ---
        lbl_scale = ctk.CTkLabel(self, text="System Scaling Factor", font=("Bahnschrift", 18, "bold"), text_color="#2ecc71")
        lbl_scale.pack(pady=(10, 5))
        
        self.lbl_scale = ctk.CTkLabel(self, text=f"{parent.scaling_factor}x Size", font=("Bahnschrift", 14))
        self.lbl_scale.pack(pady=(0, 10))
        
        self.slider_scale = ctk.CTkSlider(self, from_=0.1, to=5.0, number_of_steps=49, command=self.update_scale_label)
        self.slider_scale.set(parent.scaling_factor)
        self.slider_scale.pack(pady=5, padx=40, fill="x")
        
        hint_scale = ctk.CTkLabel(self, text="Use 1.0x for exact CSV match.\nUse 0.5x if CSV is for 2 panels but you only have 1.", 
                                  justify="left", text_color="gray", font=("Consolas", 11))
        hint_scale.pack(pady=(0, 20))
        
        # --- Save Button ---
        btn_save = ctk.CTkButton(self, text="Save Configuration", command=self.save_settings, width=200, height=40,
                                 fg_color="#27ae60", font=("Bahnschrift", 14, "bold"))
        btn_save.pack(pady=30)

    def update_tree_label(self, value):
        self.lbl_trees.configure(text=f"{int(value)} Trees")

    def update_scale_label(self, value):
        self.lbl_scale.configure(text=f"{value:.1f}x Size")

    def save_settings(self):
        self.parent.n_estimators = int(self.slider_trees.get())
        self.parent.scaling_factor = round(float(self.slider_scale.get()), 2)
        self.parent.log(f"CONFIGURATION UPDATED:\n   > AI Trees: {self.parent.n_estimators}\n   > Scaling Factor: {self.parent.scaling_factor}x")
        self.destroy()

# =============================================================================
#   WINDOW: ANALYSIS DASHBOARD
#   (Detailed Engineering Metrics)
# =============================================================================
class AnalysisWindow(ctk.CTkToplevel):
    def __init__(self, parent, df):
        super().__init__(parent)
        self.title("Model Accuracy Analysis")
        self.geometry("1100x650")
        self.attributes("-topmost", True)
        
        # Filter for valid data only
        valid_df = df[(df['Actual_Output'] > 0) & (df['Physical_Pred'] > 0)].copy()
        
        if valid_df.empty:
            ctk.CTkLabel(self, text="Insufficient data for analysis.", font=FONT_HEADER).pack(pady=20)
            return

        # Calculate Metrics
        y_true = valid_df['Actual_Output']
        y_phys = valid_df['Physical_Pred']
        y_stat = valid_df['Stat_Pred']

        r2_phys = r2_score(y_true, y_phys)
        r2_stat = r2_score(y_true, y_stat)
        
        rmse_phys = np.sqrt(mean_squared_error(y_true, y_phys))
        rmse_stat = np.sqrt(mean_squared_error(y_true, y_stat))

        # --- Header Section ---
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.pack(fill="x", padx=20, pady=10)
        
        title = ctk.CTkLabel(header_frame, text="Prediction vs. Actual Analysis", font=FONT_HEADER, text_color="white")
        title.pack(side="top", pady=(0,10))

        metrics_txt = (f"PHYSICAL MODEL:\n"
                       f"  R² Score: {r2_phys:.3f}\n"
                       f"  RMSE: {rmse_phys:.3f} kWh\n\n"
                       f"AI MODEL (Scaled {parent.scaling_factor}x):\n"
                       f"  R² Score: {r2_stat:.3f}\n"
                       f"  RMSE: {rmse_stat:.3f} kWh")
        
        ctk.CTkLabel(header_frame, text=metrics_txt, font=("Consolas", 12), justify="left", 
                     fg_color="#333333", corner_radius=6, padx=20, pady=10).pack()

        # --- Plotting Section ---
        plot_frame = ctk.CTkFrame(self)
        plot_frame.pack(fill="both", expand=True, padx=20, pady=10)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=100)
        fig.patch.set_facecolor('#2b2b2b')

        # Plot 1: Physical
        ax1.set_facecolor('#242424')
        ax1.scatter(y_true, y_phys, alpha=0.5, s=10, color='#3498db')
        lims_p = [0, max(y_true.max(), y_phys.max())]
        ax1.plot(lims_p, lims_p, 'r--', alpha=0.7, label="Ideal")
        ax1.set_title("Physical Model Performance", color='white')
        ax1.set_xlabel("Actual Output (kWh)", color='white')
        ax1.set_ylabel("Predicted Output (kWh)", color='white')
        ax1.tick_params(colors='white')
        ax1.grid(True, alpha=0.2)

        # Plot 2: AI
        ax2.set_facecolor('#242424')
        ax2.scatter(y_true, y_stat, alpha=0.5, s=10, color='#2ecc71')
        lims_s = [0, max(y_true.max(), y_stat.max())]
        ax2.plot(lims_s, lims_s, 'r--', alpha=0.7, label="Ideal")
        ax2.set_title("AI Model Performance", color='white')
        ax2.set_xlabel("Actual Output (kWh)", color='white')
        ax2.tick_params(colors='white')
        ax2.grid(True, alpha=0.2)

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

# =============================================================================
#   WINDOW: REPORT (ENGINEERING ONLY - NO FINANCIALS)
#   (Shows pure energy stats, strictly kWh)
# =============================================================================
class ReportWindow(ctk.CTkToplevel):
    def __init__(self, parent, phys_24h, stat_24h):
        super().__init__(parent)
        self.title("Engineering Energy Report")
        self.geometry("900x650")
        self.resizable(True, True)
        self.attributes("-topmost", True)
        
        ctk.CTkLabel(self, text="Energy Output Projections", font=("Bahnschrift", 24, "bold"), text_color="#3498db").pack(pady=20)
        ctk.CTkLabel(self, text="Values in Kilowatt-Hours (kWh)", font=("Bahnschrift", 12), text_color="gray").pack(pady=(0, 20))
        
        grid_frame = ctk.CTkFrame(self, fg_color="transparent")
        grid_frame.pack(fill="both", expand=True, padx=20, pady=10)

        periods = [("Next 24 Hours", 1), ("Next 7 Days", 7), ("Next 30 Days", 30), ("Next 1 Year", 365)]
        colors = ["#2ecc71", "#27ae60", "#f1c40f", "#e67e22"] 

        for i, (label, multiplier) in enumerate(periods):
            row = i // 2
            col = i % 2
            
            p_val = phys_24h * multiplier
            s_val = stat_24h * multiplier
            
            card = ctk.CTkFrame(grid_frame, corner_radius=10, fg_color="#333333", border_width=1, border_color="#555555")
            card.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            grid_frame.grid_columnconfigure(col, weight=1)
            grid_frame.grid_rowconfigure(row, weight=1)

            ctk.CTkLabel(card, text=label, font=("Bahnschrift", 18, "bold"), text_color=colors[i]).pack(pady=(20, 10))
            
            ctk.CTkLabel(card, text=f"Physical Model:", font=("Bahnschrift", 12), text_color="gray").pack(pady=(5,0))
            ctk.CTkLabel(card, text=f"{p_val:,.0f} kWh", font=("Bahnschrift", 16, "bold"), text_color="lightgray").pack()
            
            ctk.CTkLabel(card, text=f"AI Model:", font=("Bahnschrift", 12), text_color="gray").pack(pady=(15,0))
            ctk.CTkLabel(card, text=f"{s_val:,.0f} kWh", font=("Bahnschrift", 22, "bold"), text_color="#2CC985").pack(pady=(0, 20))

        ctk.CTkButton(self, text="Close Report", command=self.destroy, fg_color="#c0392b", hover_color="#e74c3c", font=("Bahnschrift", 14)).pack(pady=20)

# =============================================================================
#   WINDOW: PREDICTION RANGE DIALOG
# =============================================================================
class PredictionRangeDialog(ctk.CTkToplevel):
    def __init__(self, parent, callback):
        super().__init__(parent)
        self.callback = callback
        self.title("Select Prediction Range")
        self.geometry("400x350")
        self.resizable(False, False)
        self.attributes("-topmost", True)
        
        ctk.CTkLabel(self, text="Long-Term Climate Projection", font=("Bahnschrift", 18, "bold")).pack(pady=(20, 5))
        ctk.CTkLabel(self, text="Uses Seasonal Averaging Algorithm", font=("Bahnschrift", 12), text_color="gray").pack(pady=(0, 20))

        row_start = ctk.CTkFrame(self, fg_color="transparent")
        row_start.pack(pady=10)
        ctk.CTkLabel(row_start, text="Start Date:", width=80).pack(side="left")
        self.sel_start = DateSelector(row_start, datetime.date.today() + datetime.timedelta(days=1))
        self.sel_start.pack(side="left")

        row_end = ctk.CTkFrame(self, fg_color="transparent")
        row_end.pack(pady=10)
        ctk.CTkLabel(row_end, text="End Date:", width=80).pack(side="left")
        self.sel_end = DateSelector(row_end, datetime.date.today() + datetime.timedelta(days=365))
        self.sel_end.pack(side="left")

        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(pady=30)
        ctk.CTkButton(btn_frame, text="Run Projection", command=self.on_confirm, width=120, fg_color="#2980b9").pack(side="left", padx=10)
        ctk.CTkButton(btn_frame, text="Cancel", command=self.destroy, width=100, fg_color="#c0392b").pack(side="left", padx=10)

    def on_confirm(self):
        start = self.sel_start.get_date()
        end = self.sel_end.get_date()
        if not start or not end:
             messagebox.showerror("Error", "Invalid date selection.")
             return
        if end < start:
            messagebox.showerror("Error", "End date cannot be before Start date.")
            return
        self.callback(start, end)
        self.destroy()

# =============================================================================
#   MAIN APPLICATION
# =============================================================================
class SolarForecastApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Solar Grid Output Forecaster (Enterprise Edition)")
        self.geometry("1280x800")
        
        # --- Application State ---
        self.df = None
        self.forecast_df = None 
        self.current_view_df = None
        self.model_stats = None
        self.n_estimators = 100 
        self.scaling_factor = 1.0 
        self.cid = None
        self.last_plot_type = None
        
        # --- Canvas State ---
        self.canvas = None
        self.toolbar = None
        self.toolbar_frame = None 
        self.ax = None
        self.v_line = None
        self.annot = None
        self.status_text = None 

        # --- Grid Layout ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Sidebar ---
        self.sidebar = ctk.CTkScrollableFrame(self, width=300, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        self.logo_label = ctk.CTkLabel(self.sidebar, text="SOLAR FORECASTER", font=FONT_HEADER)
        self.logo_label.pack(pady=(30, 20))

        # --- Inputs Section ---
        self.create_location_inputs()
        self.create_phys_inputs()

        # --- Section: Model Persistence (Save/Load) ---
        f_model = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        f_model.pack(fill="x", padx=20, pady=(20, 10))
        ctk.CTkLabel(f_model, text="Model Management", font=("Bahnschrift", 12, "bold")).pack(pady=5)
        
        row_btns = ctk.CTkFrame(f_model, fg_color="transparent")
        row_btns.pack(fill="x")
        ctk.CTkButton(row_btns, text="💾 Save Brain", width=110, command=self.save_model, fg_color="#27ae60", hover_color="#2ecc71").pack(side="left", padx=2)
        ctk.CTkButton(row_btns, text="📂 Load Brain", width=110, command=self.load_model, fg_color="#2980b9", hover_color="#3498db").pack(side="right", padx=2)

        # --- Section: Data Management ---
        f_data = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        f_data.pack(fill="x", padx=20, pady=10)
        ctk.CTkButton(f_data, text="Import CSV", width=110, command=self.import_csv, fg_color="#555555").pack(side="left", padx=2)
        ctk.CTkButton(f_data, text="Export Data", width=110, command=self.export_data, fg_color="#555555").pack(side="right", padx=2)
        
        ctk.CTkButton(self.sidebar, text="⚙ Settings", command=self.open_settings, font=FONT_MAIN, fg_color="#7f8c8d", hover_color="#95a5a6").pack(pady=10)
        
        # --- Section: Pipeline Buttons ---
        self.btn_load = ctk.CTkButton(self.sidebar, text="1. Fetch History Data", command=self.fetch_historical_data, font=FONT_MAIN, fg_color="#D35400", hover_color="#A04000")
        self.btn_load.pack(pady=10, padx=20)
        
        self.btn_compare = ctk.CTkButton(self.sidebar, text="2a. Train New Model", command=self.run_comparison, font=FONT_MAIN, fg_color="transparent", border_width=2)
        self.btn_compare.pack(pady=10, padx=20)

        # --- CRITICAL FEATURE: TEST IMPORTED MODEL ---
        self.btn_test = ctk.CTkButton(self.sidebar, text="2b. Test Imported Model", command=self.test_existing_model, font=FONT_MAIN, fg_color="#8e44ad", hover_color="#9b59b6")
        self.btn_test.pack(pady=10, padx=20)
        
        self.btn_forecast = ctk.CTkButton(self.sidebar, text="3. Real Forecast (24h)", command=self.extrapolate_data, font=FONT_MAIN, fg_color="#2CC985", text_color="white")
        self.btn_forecast.pack(pady=10, padx=20)
        
        ctk.CTkButton(self.sidebar, text="4. View Report", command=self.open_report, font=FONT_MAIN, fg_color="#8e44ad").pack(pady=10, padx=20)
        ctk.CTkButton(self.sidebar, text="5. Predict Custom Range", command=self.open_prediction_dialog, font=FONT_MAIN, fg_color="#2980b9").pack(pady=10, padx=20)
        ctk.CTkButton(self.sidebar, text="6. Analyze Accuracy", command=self.open_analysis, font=FONT_MAIN, fg_color="#c0392b").pack(pady=10, padx=20)

        # --- Right Side (Content) ---
        self.right_frame = ctk.CTkFrame(self, corner_radius=10, fg_color="transparent")
        self.right_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.right_frame.grid_rowconfigure(0, weight=1)
        self.right_frame.grid_columnconfigure(0, weight=1)
        
        self.tabview = ctk.CTkTabview(self.right_frame)
        self.tabview.pack(fill="both", expand=True)
        
        self.tab_graphs = self.tabview.add("Graphs")
        self.tab_logs = self.tabview.add("System Log")
        
        self.tab_graphs.grid_columnconfigure(0, weight=1)
        self.tab_graphs.grid_rowconfigure(1, weight=1)
        self.tab_logs.grid_columnconfigure(0, weight=1)
        self.tab_logs.grid_rowconfigure(0, weight=1)

        # --- Graph Controls ---
        controls = ctk.CTkFrame(self.tab_graphs, height=40)
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        ctk.CTkLabel(controls, text="View Duration:", font=("Bahnschrift", 12)).pack(side="left", padx=10)
        
        self.view_selector = ctk.CTkSegmentedButton(controls, 
                                                    values=["Full Data", "Last 7 Days", "Last 3 Days", "Last 24h"],
                                                    command=self.update_view_duration)
        self.view_selector.set("Full Data")
        self.view_selector.pack(side="left", padx=10, pady=5)

        self.graph_frame = ctk.CTkFrame(self.tab_graphs, corner_radius=10)
        self.graph_frame.grid(row=1, column=0, sticky="nsew")
        
        self.placeholder_lbl = ctk.CTkLabel(self.graph_frame, text="1. Select Dates\n2. Fetch Data\n3. Use Toolbar to Zoom/Pan", font=FONT_HEADER, text_color="gray")
        self.placeholder_lbl.place(relx=0.5, rely=0.5, anchor="center")

        # --- Log Box ---
        self.textbox = ctk.CTkTextbox(self.tab_logs, font=FONT_MONO)
        self.textbox.pack(fill="both", expand=True, padx=10, pady=10)

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        plt.close('all')
        self.destroy()
        sys.exit(0)

    def open_settings(self):
        SettingsWindow(self)

    def open_prediction_dialog(self):
        PredictionRangeDialog(self, self.run_long_term_projection)

    def create_location_inputs(self):
        f = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        f.pack(fill="x", padx=20, pady=5)
        ctk.CTkLabel(f, text="Location & Date Range", font=FONT_BOLD).pack(anchor="w")
        
        self.loc_entries = {}
        for lbl, val in [("Lat", "12.97"), ("Lon", "79.16")]:
            r = ctk.CTkFrame(f, fg_color="transparent")
            r.pack(fill="x", pady=2)
            ctk.CTkLabel(r, text=lbl, width=60, anchor="w").pack(side="left")
            e = ctk.CTkEntry(r); e.insert(0, val); e.pack(side="right", fill="x", expand=True)
            self.loc_entries[lbl] = e
            
        today = datetime.date.today()
        # Dates setup
        ctk.CTkLabel(f, text="Start Date:", width=80, anchor="w").pack(side="top", anchor="w", pady=(10,0))
        self.sel_start = DateSelector(f, today - datetime.timedelta(days=14))
        self.sel_start.pack(anchor="w")
        
        ctk.CTkLabel(f, text="End Date:", width=80, anchor="w").pack(side="top", anchor="w", pady=(5,0))
        
        # --- FIX: Default End Date changed to 2 days ago (Aggressive Lag) ---
        self.sel_end = DateSelector(f, today - datetime.timedelta(days=2))
        self.sel_end.pack(anchor="w")

    def create_phys_inputs(self):
        f = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        f.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(f, text="Physical Params", font=FONT_BOLD).pack(anchor="w")
        self.phys_entries = {}
        for lbl, val in [("Area (m²)", "50"), ("Eff (η)", "0.18"), ("Coeff (α)", "0.004"), ("Loss (L)", "0.14")]:
            r = ctk.CTkFrame(f, fg_color="transparent")
            r.pack(fill="x", pady=2)
            ctk.CTkLabel(r, text=lbl).pack(side="left")
            e = ctk.CTkEntry(r, width=80); e.insert(0, val); e.pack(side="right")
            self.phys_entries[lbl] = e

    def log(self, msg):
        print(msg)
        self.textbox.insert("end", f"> {msg}\n"); self.textbox.see("end")

    def get_weather_status(self, row):
        rain = row.get('Precipitation', 0)
        clouds = row.get('Cloud_Cover', 0)
        ghi = row.get('GHI', 0)
        is_daytime = ghi > 0.05 
        if rain > 0.1: return "Rainy 🌧"
        elif clouds > 70: return "Cloudy ☁"
        elif clouds > 30: return "Partly Cloudy ⛅" if is_daytime else "Partly Cloudy 🌙"
        else: return "Sunny ☀" if is_daytime else "Clear Night 🌙"

    # =========================================================================
    #   MODEL MANAGEMENT
    # =========================================================================
    def save_model(self):
        if self.model_stats is None:
            messagebox.showwarning("No Model", "You haven't trained a model yet!\nRun Step 2 first.")
            return
        
        file_path = filedialog.asksaveasfilename(defaultextension=".joblib", filetypes=[("AI Model", "*.joblib")])
        if file_path:
            try:
                joblib.dump(self.model_stats, file_path)
                self.log(f"Model saved successfully to: {os.path.basename(file_path)}")
                messagebox.showinfo("Saved", "AI Model saved successfully!")
            except Exception as e:
                messagebox.showerror("Save Error", str(e))

    def load_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("AI Model", "*.joblib")])
        if file_path:
            try:
                self.model_stats = joblib.load(file_path)
                self.log(f"Model loaded: {os.path.basename(file_path)}")
                self.log("You can now run forecasts without re-training!")
                self.btn_compare.configure(text="2a. Model Loaded (Ready)", fg_color="#2ecc71")
            except Exception as e:
                messagebox.showerror("Load Error", str(e))

    # =========================================================================
    #   DATA IMPORT / EXPORT
    # =========================================================================
    def import_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file_path: return
        
        try:
            self.log(f"Importing data from {os.path.basename(file_path)}...")
            df = pd.read_csv(file_path)
            
            required = ['Timestamp', 'GHI', 'Temperature', 'Actual_Output']
            if not all(col in df.columns for col in required):
                raise ValueError(f"CSV missing required columns: {required}")
            
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df = df.sort_values(by='Timestamp')
            
            self.df = df
            self.last_plot_type = "fetch"
            self.log(f"SUCCESS! {len(df)} rows imported.")
            self.plot_graph(self.df, f"Imported Data", is_forecast=False)
            
        except Exception as e:
            self.log(f"Import Error: {e}")
            messagebox.showerror("Import Failed", str(e))

    def export_data(self):
        if self.df is None and self.forecast_df is None:
            messagebox.showwarning("No Data", "Nothing to export yet.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel Files", "*.xlsx"), ("CSV Files", "*.csv")]
        )
        if not file_path: return
        
        try:
            self.log(f"Exporting data...")
            target_df = self.df
            
            if self.forecast_df is not None and self.df is not None:
                history = self.df.copy()
                for col in ['Physical_Forecast', 'Stat_Forecast']:
                    if col not in history.columns: history[col] = np.nan
                
                combined = pd.concat([history, self.forecast_df], ignore_index=True).sort_values(by='Timestamp')
                target_df = combined
            elif self.forecast_df is not None:
                target_df = self.forecast_df

            if file_path.endswith('.csv'):
                target_df.to_csv(file_path, index=False)
            else:
                with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                    target_df.to_excel(writer, sheet_name='SolarData', index=False)
            self.log(f"SUCCESS! Data saved to {os.path.basename(file_path)}")
            
        except Exception as e:
            self.log(f"Export Error: {e}")
            messagebox.showerror("Export Failed", f"Error: {str(e)}\n\n(Did you run 'pip install openpyxl'?)")

    # =========================================================================
    #   CORE DATA FETCHING
    # =========================================================================
    def fetch_historical_data(self, start_date=None, end_date=None):
        try:
            self.log("Connecting to Open-Meteo Archive...")
            lat = self.loc_entries["Lat"].get()
            lon = self.loc_entries["Lon"].get()
            
            if start_date and end_date:
                start_str = start_date.strftime("%Y-%m-%d")
                end_str = end_date.strftime("%Y-%m-%d")
                self.log(f"Auto-Fetching Missing Data: {start_str} to {end_str}")
            else:
                user_start = self.sel_start.get_date()
                user_end = self.sel_end.get_date()
                if not user_start or not user_end:
                    messagebox.showerror("Date Error", "Invalid date selection")
                    return
                start_str = user_start.strftime("%Y-%m-%d")
                end_str = user_end.strftime("%Y-%m-%d")
                self.log(f"Fetching Historical Data: {start_str} to {end_str}")
            
            # --- FIX: AGGRESSIVE 2-DAY LAG ---
            today = datetime.date.today()
            cutoff = today - datetime.timedelta(days=2) 
            
            if not start_date and user_end > cutoff:
                 messagebox.showerror("API Warning", "Archive API lags by 2-5 days. Pick an older date.")
                 return

            url = "https://archive-api.open-meteo.com/v1/archive"
            params = {
                "latitude": lat, "longitude": lon,
                "start_date": start_str, "end_date": end_str,
                "hourly": "temperature_2m,cloud_cover,precipitation,shortwave_radiation",
                "timezone": "auto"
            }
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 400:
                 messagebox.showerror("API Error", "Data unavailable for this range.")
                 return
            
            response.raise_for_status()
            data = response.json()
            
            hourly = data['hourly']
            timestamps = pd.to_datetime(hourly['time'])
            new_df = pd.DataFrame({
                'Timestamp': timestamps, 'Temperature': hourly['temperature_2m'],
                'Cloud_Cover': hourly['cloud_cover'], 'Precipitation': hourly['precipitation'],
                'GHI_W': hourly['shortwave_radiation']
            })
            new_df['GHI'] = new_df['GHI_W'] / 1000.0
            
            # --- REAL WORLD SIMULATION (Dirty/Broken Panels) ---
            # We simulate a "Systematic Defect" that the Physics model (which assumes 0.18) doesn't know about.
            # Efficiency 0.15 (vs 0.18 Input) | Loss 0.20 (vs 0.14 Input)
            # This creates a gap that the AI can "learn" and fix.
            base_energy = (50 * new_df['GHI'] * 0.15 * (1 - 0.004 * (new_df['Temperature'] - 25))) * 0.80
            
            day_mask = new_df['GHI'] > 0.05
            # Low noise to make the pattern clear for the AI
            noise = np.random.normal(0, 0.3, day_mask.sum())
            new_df['Actual_Output'] = base_energy
            new_df.loc[day_mask, 'Actual_Output'] += noise
            new_df['Actual_Output'] = np.maximum(0, new_df['Actual_Output'])
            
            if self.df is not None and start_date:
                self.df = pd.concat([self.df, new_df]).drop_duplicates(subset='Timestamp').sort_values(by='Timestamp')
                self.log(f"Merged {len(new_df)} new rows.")
            else:
                self.df = new_df
                self.last_plot_type = "fetch"
                self.log(f"SUCCESS! {len(self.df)} rows loaded.")
                self.plot_graph(self.df, f"Historical Data", is_forecast=False)

        except Exception as e:
            self.log(f"Fetch Error: {e}")
            tk.messagebox.showerror("Fetch Error", str(e))

    # =========================================================================
    #   AI TRAINING & TESTING
    # =========================================================================
    def run_comparison(self):
        if self.df is None or self.df.empty:
            tk.messagebox.showwarning("Order Error", "Click Button 1 first!")
            return
            
        self.btn_compare.configure(state="disabled", text="Training...", fg_color="#555555")
        self.log(f"Starting AI Training ({self.n_estimators} Trees)...")
        
        thread = threading.Thread(target=self._train_model_thread)
        thread.daemon = True 
        thread.start()

    def _train_model_thread(self):
        try:
            A = float(self.phys_entries["Area (m²)"].get())
            eta = float(self.phys_entries["Eff (η)"].get())
            alpha = float(self.phys_entries["Coeff (α)"].get())
            L = float(self.phys_entries["Loss (L)"].get())

            phys_calc = A * self.df['GHI'] * eta * (1 - alpha * (self.df['Temperature'] - 25)) * (1 - L)
            self.df['Physical_Pred'] = np.maximum(0, phys_calc)

            self.df['Hour'] = self.df['Timestamp'].dt.hour
            self.df['Month'] = self.df['Timestamp'].dt.month
            
            X = self.df[['GHI', 'Temperature', 'Hour', 'Month']]
            y = self.df['Actual_Output']
            
            # --- n_jobs=1 to prevent freezing ---
            self.model_stats = RandomForestRegressor(n_estimators=self.n_estimators, random_state=42, n_jobs=1)
            self.model_stats.fit(X, y)
            
            stat_pred = self.model_stats.predict(X)
            
            # --- APPLY SCALING FACTOR ---
            stat_pred = stat_pred * self.scaling_factor
            
            self.df['Stat_Pred'] = stat_pred
            self.df.loc[self.df['GHI'] < 0.05, 'Stat_Pred'] = 0
            self.df['Stat_Pred'] = np.maximum(0, self.df['Stat_Pred'])

            self.after(0, self._training_complete_success)
            
        except Exception as e:
            self.after(0, lambda: self._training_complete_error(str(e)))

    def _training_complete_success(self):
        y = self.df['Actual_Output']
        mae_p = mean_absolute_error(y, self.df['Physical_Pred'])
        mae_s = mean_absolute_error(y, self.df['Stat_Pred'])
        self.log(f"Physical MAE: {mae_p:.3f}")
        self.log(f"AI Model MAE: {mae_s:.3f}")
        
        self.last_plot_type = "compare"
        self.plot_graph(self.df, "Model Comparison", comparison=True)
        self.btn_compare.configure(state="normal", text="2a. Train New Model", fg_color="transparent")

    def _training_complete_error(self, error_msg):
        tk.messagebox.showerror("Model Error", error_msg)
        self.btn_compare.configure(state="normal", text="2a. Train New Model", fg_color="transparent")

    # --- FEATURE: Test Imported Model (No Training) ---
    def test_existing_model(self):
        if self.df is None: return messagebox.showwarning("Error", "Import CSV first.")
        if self.model_stats is None: return messagebox.showwarning("Error", "Load a Model first.")
        
        self.log("Testing Imported Model on Current Data (No Training)...")
        try:
            A = float(self.phys_entries["Area (m²)"].get())
            eta = float(self.phys_entries["Eff (η)"].get())
            alpha = float(self.phys_entries["Coeff (α)"].get())
            L = float(self.phys_entries["Loss (L)"].get())

            self.df['Physical_Pred'] = np.maximum(0, A * self.df['GHI'] * eta * (1 - alpha * (self.df['Temperature'] - 25)) * (1 - L))
            
            self.df['Hour'] = self.df['Timestamp'].dt.hour
            self.df['Month'] = self.df['Timestamp'].dt.month
            X = self.df[['GHI', 'Temperature', 'Hour', 'Month']]
            
            # Predict using EXISTING brain
            self.df['Stat_Pred'] = np.maximum(0, self.model_stats.predict(X) * self.scaling_factor)
            
            y = self.df['Actual_Output']
            r2 = r2_score(y, self.df['Stat_Pred'])
            self.log(f"Test Result: R² = {r2:.3f}")
            
            self.last_plot_type = "compare"
            # FIX: Explicitly set comparison=True so it doesn't look for Forecast columns
            self.plot_graph(self.df, "Model Test (No Training)", comparison=True)
            
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # =========================================================================
    #   FORECASTING
    # =========================================================================
    def extrapolate_data(self):
        # ALGORITHM 1: REAL FORECAST (Uses Random Forest)
        try:
            self.log("Fetching Live Forecast...")
            if self.model_stats is None:
                tk.messagebox.showwarning("Order Error", "Click Button 2a or Load Model first!")
                return
                
            lat = self.loc_entries["Lat"].get()
            lon = self.loc_entries["Lon"].get()
            url = "https://api.open-meteo.com/v1/forecast"
            params = {"latitude": lat, "longitude": lon, "hourly": "temperature_2m,shortwave_radiation", "forecast_days": 16}
            data = requests.get(url, params=params).json()['hourly']
            
            future = pd.DataFrame({'Timestamp': pd.to_datetime(data['time']), 
                                   'Temperature': data['temperature_2m'], 
                                   'GHI_W': data['shortwave_radiation']})
            future['GHI'] = future['GHI_W'] / 1000.0
            
            A = float(self.phys_entries["Area (m²)"].get())
            eta = float(self.phys_entries["Eff (η)"].get())
            alpha = float(self.phys_entries["Coeff (α)"].get())
            L = float(self.phys_entries["Loss (L)"].get())
            future['Physical_Forecast'] = np.maximum(0, A * future['GHI'] * eta * (1 - alpha * (future['Temperature'] - 25)) * (1 - L))
            
            future['Hour'] = future['Timestamp'].dt.hour
            future['Month'] = future['Timestamp'].dt.month
            if self.model_stats:
                future['Stat_Forecast'] = np.maximum(0, self.model_stats.predict(future[['GHI', 'Temperature', 'Hour', 'Month']]) * self.scaling_factor)
            
            # --- FIX: TRAPEZOID DEPRECATION (SAFE MODE) ---
            if hasattr(np, 'trapezoid'):
                future.attrs['total_phys'] = np.trapezoid(future['Physical_Forecast'], dx=1)
                future.attrs['total_stat'] = np.trapezoid(future.get('Stat_Forecast', future['Physical_Forecast']), dx=1)
            else:
                future.attrs['total_phys'] = np.trapz(future['Physical_Forecast'], dx=1)
                future.attrs['total_stat'] = np.trapz(future.get('Stat_Forecast', future['Physical_Forecast']), dx=1)
            
            self.forecast_df = future
            self.last_plot_type = "forecast"
            self.plot_graph(future, "16-Day Forecast", is_forecast=True)
            
        except Exception as e: self.log(f"Forecast Error: {e}")

    def run_long_term_projection(self, start, end):
        # ALGORITHM 2: SEASONAL CLIMATOLOGY (The Architect)
        self.log(f"Running Seasonal Algorithm ({start} to {end})...")
        try:
            if self.df is None: return
            self.df['Month'] = self.df['Timestamp'].dt.month
            self.df['Hour'] = self.df['Timestamp'].dt.hour
            profile = self.df.groupby(['Month', 'Hour'])[['GHI', 'Temperature']].mean().reset_index()
            
            dates = pd.date_range(start, end, freq='H')
            future = pd.DataFrame({'Timestamp': dates})
            future['Month'] = future['Timestamp'].dt.month
            future['Hour'] = future['Timestamp'].dt.hour
            future = pd.merge(future, profile, on=['Month', 'Hour'], how='left').fillna(0)
            
            A = float(self.phys_entries["Area (m²)"].get())
            eta = float(self.phys_entries["Eff (η)"].get())
            alpha = float(self.phys_entries["Coeff (α)"].get())
            L = float(self.phys_entries["Loss (L)"].get())
            future['Physical_Forecast'] = np.maximum(0, A * future['GHI'] * eta * (1 - alpha * (future['Temperature'] - 25)) * (1 - L))
            
            if self.model_stats:
                future['Stat_Forecast'] = np.maximum(0, self.model_stats.predict(future[['GHI', 'Temperature', 'Hour', 'Month']]) * self.scaling_factor)
                future['Stat_Forecast'] = future['Stat_Forecast'].rolling(window=24*7, min_periods=1).mean()
            
            # --- FIX: TRAPEZOID DEPRECATION (SAFE MODE) ---
            if hasattr(np, 'trapezoid'):
                total_energy = np.trapezoid(future.get('Stat_Forecast', future['Physical_Forecast']), dx=1)
                future.attrs['total_phys'] = np.trapezoid(future['Physical_Forecast'], dx=1)
            else:
                total_energy = np.trapz(future.get('Stat_Forecast', future['Physical_Forecast']), dx=1)
                future.attrs['total_phys'] = np.trapz(future['Physical_Forecast'], dx=1)

            self.log(f"Projected Total: {total_energy:,.0f} kWh")
            
            future.attrs['total_stat'] = total_energy
            self.forecast_df = future
            self.last_plot_type = "forecast"
            self.plot_graph(future, f"Long-Term Projection", is_forecast=True)
            
        except Exception as e: self.log(f"Projection Error: {e}")

    def open_report(self):
        if self.forecast_df is not None:
            p = self.forecast_df.attrs.get('total_phys', 0)
            s = self.forecast_df.attrs.get('total_stat', 0)
            ReportWindow(self, p, s)
        else: messagebox.showwarning("Error", "Run forecast first.")

    def open_analysis(self):
        if self.df is not None and 'Physical_Pred' in self.df: AnalysisWindow(self, self.df)
        else: messagebox.showwarning("Error", "Run AI Models first.")

    def update_view_duration(self, _):
        # FIX: Ensure keyword arguments are used correctly here too
        if self.last_plot_type == "fetch" and self.df is not None:
            self.plot_graph(self.df, "Historical Data", comparison=False)
        elif self.last_plot_type == "compare" and self.df is not None:
            self.plot_graph(self.df, "Model Comparison", comparison=True)
        elif self.last_plot_type == "forecast" and self.forecast_df is not None:
            self.plot_graph(self.forecast_df, "Forecast", is_forecast=True)

    def slice_data_by_duration(self, data):
        selection = self.view_selector.get()
        if selection == "Full Data": return data.copy()
        if self.last_plot_type == "forecast":
             if selection == "Last 24h": return data.head(24).copy()
             elif selection == "Last 3 Days": return data.head(72).copy()
             elif selection == "Last 7 Days": return data.head(168).copy()
        else:
             if selection == "Last 24h": return data.tail(24).copy()
             elif selection == "Last 3 Days": return data.tail(72).copy()
             elif selection == "Last 7 Days": return data.tail(168).copy()
        return data.copy()

    # =========================================================================
    #   GRAPHING ENGINE (ROBUST & SAFE)
    # =========================================================================
    def plot_graph(self, data, title, is_forecast=False, comparison=False):
        # 1. NUCLEAR CLEANUP (Fixes shrinking graph bug)
        for widget in self.graph_frame.winfo_children(): widget.destroy()
        plt.close('all')

        fig, self.ax = plt.subplots(figsize=(6, 5), dpi=100)
        fig.patch.set_facecolor('#242424'); self.ax.set_facecolor('#2b2b2b')
        
        self.current_view_df = self.slice_data_by_duration(data)
        subset = self.current_view_df
        integrator = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz

        if is_forecast:
            t_p = integrator(subset['Physical_Forecast'], dx=1)
            self.ax.plot(subset['Timestamp'], subset['Physical_Forecast'], label=f"Physical ({t_p:.0f} kWh)", color='#3498db')
            if 'Stat_Forecast' in subset:
                t_s = integrator(subset['Stat_Forecast'], dx=1)
                self.ax.plot(subset['Timestamp'], subset['Stat_Forecast'], label=f"AI Model ({t_s:.0f} kWh)", color='#2ecc71')
                self.ax.fill_between(subset['Timestamp'], subset['Stat_Forecast'], color='#2ecc71', alpha=0.1)
            self.ax.set_ylabel("Energy (kWh)")
            
        elif comparison:
            self.ax.plot(subset['Timestamp'], subset['Actual_Output'], label='Actual', color='white', alpha=0.5)
            self.ax.plot(subset['Timestamp'], subset['Physical_Pred'], label='Physical', linestyle='--', color='#3498db')
            self.ax.plot(subset['Timestamp'], subset['Stat_Pred'], label='AI', linestyle='-.', color='#2ecc71')
            self.ax.set_ylabel("Energy (kWh)")
            
        else:
            self.ax.plot(subset['Timestamp'], subset['GHI'], label='Sun', color='orange')
            self.ax.plot(subset['Timestamp'], subset['Temperature'], label='Temp', color='cyan')
            self.ax.set_ylabel("Weather")

        self.ax.set_title(title, color='white'); self.ax.legend(loc='upper right'); self.ax.grid(True, alpha=0.3)
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        
        # 4. Render
        self.canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, ctk.CTkFrame(self.graph_frame))
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        # HOVER LOGIC
        self.v_line = self.ax.axvline(x=subset['Timestamp'].iloc[0], color='white', linestyle=':', alpha=0.8); self.v_line.set_visible(False)
        self.annot = self.ax.text(0.02, 0.95, "", transform=self.ax.transAxes, bbox=dict(boxstyle="round", fc="black", ec="white", alpha=0.8), color="white", verticalalignment='top'); self.annot.set_visible(False)
        
        def hover(event):
            if not event.inaxes: return
            try:
                click_date = mdates.num2date(event.xdata).replace(tzinfo=None)
                row = subset.loc[(subset['Timestamp'] - click_date).abs().idxmin()]
                self.v_line.set_xdata([row['Timestamp'], row['Timestamp']]); self.v_line.set_visible(True)
                
                msg = f"{row['Timestamp'].strftime('%H:%M %d %b')}\n"
                if is_forecast: msg += f"AI: {row.get('Stat_Forecast', 0):.2f}\nPhys: {row.get('Physical_Forecast', 0):.2f}"
                elif comparison: msg += f"Act: {row.get('Actual_Output', 0):.2f}\nAI: {row.get('Stat_Pred', 0):.2f}"
                else: msg += f"Sun: {row.get('GHI', 0):.2f}\nTemp: {row.get('Temperature', 0):.1f}"
                
                self.annot.set_text(msg); self.annot.set_visible(True); self.canvas.draw_idle()
            except: pass
        
        self.canvas.mpl_connect("motion_notify_event", hover)

if __name__ == "__main__":
    app = SolarForecastApp()
    app.mainloop()