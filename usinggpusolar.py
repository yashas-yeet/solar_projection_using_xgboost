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
import joblib

# --- IMPORT XGBOOST ---
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# --- Configuration ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")
FONT_MAIN = ("Bahnschrift", 14)
FONT_BOLD = ("Bahnschrift", 14, "bold")
FONT_HEADER = ("Bahnschrift", 20, "bold")

# --- HELPER: DATE SELECTOR ---
class DateSelector(ctk.CTkFrame):
    def __init__(self, parent, default_date=None):
        super().__init__(parent, fg_color="transparent")
        if default_date is None: default_date = datetime.date.today()
        self.days = [str(i).zfill(2) for i in range(1, 32)]
        self.months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        self.years = [str(i) for i in range(2000, 2031)]
        self.combo_day = ctk.CTkComboBox(self, values=self.days, width=60, font=("Bahnschrift", 12))
        self.combo_day.set(str(default_date.day).zfill(2))
        self.combo_day.pack(side="left", padx=(0, 5))
        self.combo_month = ctk.CTkComboBox(self, values=self.months, width=70, font=("Bahnschrift", 12))
        self.combo_month.set(self.months[default_date.month - 1])
        self.combo_month.pack(side="left", padx=(0, 5))
        self.combo_year = ctk.CTkComboBox(self, values=self.years, width=80, font=("Bahnschrift", 12))
        self.combo_year.set(str(default_date.year))
        self.combo_year.pack(side="left")

    def get_date(self):
        try:
            d = int(self.combo_day.get())
            m = self.months.index(self.combo_month.get()) + 1
            y = int(self.combo_year.get())
            return datetime.date(y, m, d)
        except: return None

# --- SETTINGS WINDOW ---
class SettingsWindow(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title("Settings")
        self.geometry("400x400")
        self.attributes("-topmost", True)
        ctk.CTkLabel(self, text="AI Complexity", font=FONT_BOLD).pack(pady=20)
        self.slider = ctk.CTkSlider(self, from_=10, to=1000, number_of_steps=99, command=self.update_label)
        self.slider.set(parent.n_estimators)
        self.slider.pack(pady=10)
        self.lbl = ctk.CTkLabel(self, text=f"{parent.n_estimators} ESTIMATORS / TREES")
        self.lbl.pack()
        ctk.CTkButton(self, text="Save", command=self.destroy, fg_color="#27ae60").pack(pady=30)

    def update_label(self, val):
        self.lbl.configure(text=f"{int(val)} ESTIMATORS / TREES")
        self.parent.n_estimators = int(val)

# --- ANALYSIS DASHBOARD ---
class AnalysisWindow(ctk.CTkToplevel):
    def __init__(self, parent, df):
        super().__init__(parent)
        self.title("Analysis")
        self.geometry("1100x650")
        valid = df[(df['Actual_Output'] > 0) & (df['Physical_Pred'] > 0)].copy()
        if valid.empty: return
        
        y_true, y_phys, y_stat = valid['Actual_Output'], valid['Physical_Pred'], valid['Stat_Pred']
        r2_p = r2_score(y_true, y_phys)
        r2_s = r2_score(y_true, y_stat)
        rmse_p = np.sqrt(mean_squared_error(y_true, y_phys))
        rmse_s = np.sqrt(mean_squared_error(y_true, y_stat))
        
        n_points = len(valid)
        
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(header, text="Prediction vs. Actual Analysis", font=("Bahnschrift", 22, "bold")).pack(pady=(0,10))
        
        stats_frame = ctk.CTkFrame(header, fg_color="#333333", corner_radius=8)
        stats_frame.pack(ipadx=20, ipady=10)
        
        txt = (f"DATA POINTS: {n_points:,}\n\n"
               f"PHYSICAL MODEL: R²={r2_p:.3f} | RMSE={rmse_p:.2f} kWh\n"
               f"AI MODEL: R²={r2_s:.3f} | RMSE={rmse_s:.2f} kWh")
        ctk.CTkLabel(stats_frame, text=txt, font=("Consolas", 14), text_color="#2ecc71").pack()
        
        plot_frame = ctk.CTkFrame(self)
        plot_frame.pack(fill="both", expand=True, padx=20, pady=10)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        fig.patch.set_facecolor('#2b2b2b')
        
        ax1.set_facecolor('#242424'); ax1.scatter(y_true, y_phys, s=10, alpha=0.5, color='#3498db')
        ax1.plot([0, y_true.max()], [0, y_true.max()], 'r--')
        ax1.set_title("Physical Model", color='white')
        
        ax2.set_facecolor('#242424'); ax2.scatter(y_true, y_stat, s=10, alpha=0.5, color='#2ecc71')
        ax2.plot([0, y_true.max()], [0, y_true.max()], 'r--')
        ax2.set_title("AI Model", color='white')
        
        for ax in [ax1, ax2]:
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('white'); ax.spines['top'].set_color('white')
            ax.spines['left'].set_color('white'); ax.spines['right'].set_color('white')
            ax.xaxis.label.set_color('white'); ax.yaxis.label.set_color('white')
            ax.set_xlabel("Actual (kWh)"); ax.set_ylabel("Predicted (kWh)")
            ax.grid(True, alpha=0.2)
        
        canvas = FigureCanvasTkAgg(fig, master=plot_frame); canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

# --- REPORT WINDOW ---
class ReportWindow(ctk.CTkToplevel):
    def __init__(self, parent, p_val, s_val):
        super().__init__(parent)
        self.title("Energy Report")
        self.geometry("900x650")
        
        ctk.CTkLabel(self, text="Energy Output Projections", font=("Bahnschrift", 24, "bold"), text_color="#3498db").pack(pady=20)
        
        grid = ctk.CTkFrame(self, fg_color="transparent")
        grid.pack(fill="both", expand=True, padx=20, pady=10)

        periods = [("Next 24 Hours", 1), ("Next 7 Days", 7), ("Next 30 Days", 30), ("Next 1 Year", 365)]
        colors = ["#2ecc71", "#27ae60", "#f1c40f", "#e67e22"] 

        for i, (label, multiplier) in enumerate(periods):
            row, col = i // 2, i % 2
            p = p_val * multiplier
            s = s_val * multiplier
            
            card = ctk.CTkFrame(grid, corner_radius=10, fg_color="#333333", border_width=1, border_color="#555555")
            card.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            grid.grid_columnconfigure(col, weight=1)
            grid.grid_rowconfigure(row, weight=1)

            ctk.CTkLabel(card, text=label, font=("Bahnschrift", 18, "bold"), text_color=colors[i]).pack(pady=(20, 10))
            ctk.CTkLabel(card, text=f"Physical: {p:,.0f} kWh", font=("Bahnschrift", 14), text_color="lightgray").pack()
            ctk.CTkLabel(card, text=f"AI Forecast: {s:,.0f} kWh", font=("Bahnschrift", 22, "bold"), text_color="#2CC985").pack(pady=10)
        
        ctk.CTkButton(self, text="Close", command=self.destroy, fg_color="#c0392b").pack(pady=20)

# --- PREDICTION RANGE POPUP ---
class PredictionRangeDialog(ctk.CTkToplevel):
    def __init__(self, parent, callback):
        super().__init__(parent)
        self.callback = callback
        self.title("Custom Range")
        self.geometry("300x250")
        self.s = DateSelector(self); self.s.pack(pady=10)
        self.e = DateSelector(self); self.e.pack(pady=10)
        ctk.CTkButton(self, text="Run", command=self.run).pack(pady=20)
    def run(self):
        s, e = self.s.get_date(), self.e.get_date()
        if s and e and e >= s: self.callback(s, e); self.destroy()

# --- MAIN APP ---
class SolarForecastApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Solar Grid Output Forecaster (By Yashas Vishwakarma)")
        self.geometry("1280x800")
        
        self.df = None
        self.forecast_df = None
        self.model_stats = None
        self.n_estimators = 100
        self.scaling_factor = 1.0
        self.last_plot_type = None
        self.canvas = None
        
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Sidebar
        self.sidebar_frame = ctk.CTkScrollableFrame(self, width=300, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        ctk.CTkLabel(self.sidebar_frame, text="SOLAR FORECASTER", font=FONT_HEADER).grid(row=0, column=0, padx=20, pady=(20,10))
        
        self.create_inputs()
        self.create_buttons()
        
        # Right Side
        self.right_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.right_frame.grid(row=0, column=1, sticky="nsew")
        self.right_frame.grid_rowconfigure(0, weight=1)
        self.right_frame.grid_columnconfigure(0, weight=1)
        
        self.tabview = ctk.CTkTabview(self.right_frame)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)
        self.tab_graphs = self.tabview.add("Graphs")
        self.tab_logs = self.tabview.add("Log")
        
        self.tab_graphs.grid_columnconfigure(0, weight=1)
        self.tab_graphs.grid_rowconfigure(1, weight=1)
        self.tab_logs.grid_columnconfigure(0, weight=1)
        self.tab_logs.grid_rowconfigure(0, weight=1)
        
        # Controls
        ctrl = ctk.CTkFrame(self.tab_graphs, height=40)
        ctrl.grid(row=0, column=0, sticky="ew", pady=5)
        self.view_selector = ctk.CTkSegmentedButton(ctrl, values=["Full Data", "Last 7 Days", "Last 3 Days", "Last 24h"], command=self.update_view_duration)
        self.view_selector.set("Full Data")
        self.view_selector.pack(pady=5)
        
        self.graph_frame = ctk.CTkFrame(self.tab_graphs)
        self.graph_frame.grid(row=1, column=0, sticky="nsew")
        
        # --- NEW: DEDICATED TOOLBAR FRAME ---
        # This sits below the graph frame to ensure the toolbar is always visible
        self.toolbar_frame = ctk.CTkFrame(self.tab_graphs, height=40)
        self.toolbar_frame.grid(row=2, column=0, sticky="ew")
        
        self.textbox = ctk.CTkTextbox(self.tab_logs, font=("Consolas", 12))
        self.textbox.pack(fill="both", expand=True)
        
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self): plt.close('all'); self.destroy(); sys.exit(0)
    def set_trees(self, n): self.n_estimators = n
    def log(self, m): self.textbox.insert("end", f"> {m}\n"); self.textbox.see("end")

    def create_inputs(self):
        f = ctk.CTkFrame(self.sidebar_frame)
        f.grid(row=1, column=0, padx=20, pady=5, sticky="ew")
        self.loc_entries = {}
        for l, v in [("Lat", "12.97"), ("Lon", "79.16")]:
            r = ctk.CTkFrame(f); r.pack(fill="x", pady=2)
            ctk.CTkLabel(r, text=l, width=50).pack(side="left")
            e = ctk.CTkEntry(r); e.insert(0, v); e.pack(side="right", expand=True, fill="x")
            self.loc_entries[l] = e
            
        today = datetime.date.today()
        self.sel_start = DateSelector(self.sidebar_frame, today - datetime.timedelta(days=14))
        self.sel_start.grid(row=2, column=0, padx=20, pady=2)
        self.sel_end = DateSelector(self.sidebar_frame, today - datetime.timedelta(days=2))
        self.sel_end.grid(row=3, column=0, padx=20, pady=2)
        
        f2 = ctk.CTkFrame(self.sidebar_frame)
        f2.grid(row=4, column=0, padx=20, pady=10, sticky="ew")
        self.phys_entries = {}
        for l, v in [("Area (m²)", "50"), ("Eff (η)", "0.18"), ("Coeff (α)", "0.004"), ("Loss (L)", "0.14")]:
            r = ctk.CTkFrame(f2); r.pack(fill="x", pady=2)
            ctk.CTkLabel(r, text=l).pack(side="left")
            e = ctk.CTkEntry(r, width=80); e.insert(0, v); e.pack(side="right")
            self.phys_entries[l] = e

    def create_buttons(self):
        btn_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        btn_frame.grid(row=5, column=0, padx=20, pady=5)
        self.btn_import = ctk.CTkButton(btn_frame, text="Import CSV", width=120, command=self.import_csv, fg_color="#555555", font=("Bahnschrift", 12))
        self.btn_import.pack(side="left", padx=5)
        
        self.btn_export = ctk.CTkButton(btn_frame, text="Export Data", width=120, command=self.export_data, fg_color="#555555", font=("Bahnschrift", 12))
        self.btn_export.pack(side="left", padx=5)

        self.btn_settings = ctk.CTkButton(self.sidebar_frame, text="⚙ Settings", command=self.open_settings, font=FONT_MAIN, fg_color="#7f8c8d")
        self.btn_settings.grid(row=6, column=0, padx=20, pady=10)

        self.btn_load = ctk.CTkButton(self.sidebar_frame, text="1. Fetch History Data", command=self.fetch_historical_data, font=FONT_MAIN, fg_color="#D35400")
        self.btn_load.grid(row=7, column=0, padx=20, pady=10)

        self.btn_train = ctk.CTkButton(self.sidebar_frame, text="2a. Train XGBoost", command=self.run_comparison, font=FONT_MAIN, fg_color="transparent", border_width=2)
        self.btn_train.grid(row=8, column=0, padx=20, pady=10)

        self.btn_test = ctk.CTkButton(self.sidebar_frame, text="2b. Test Model", command=self.test_existing_model, font=FONT_MAIN, fg_color="#8e44ad")
        self.btn_test.grid(row=9, column=0, padx=20, pady=10)

        self.btn_forecast = ctk.CTkButton(self.sidebar_frame, text="3. Real Forecast (24h)", command=self.extrapolate_data, font=FONT_MAIN, fg_color="#2CC985", text_color="white")
        self.btn_forecast.grid(row=10, column=0, padx=20, pady=10)

        self.btn_report = ctk.CTkButton(self.sidebar_frame, text="4. View Report", command=self.open_report_window, font=FONT_MAIN, fg_color="#8e44ad")
        self.btn_report.grid(row=11, column=0, padx=20, pady=10)

        self.btn_predict_custom = ctk.CTkButton(self.sidebar_frame, text="5. Custom Predict", command=self.open_prediction_dialog, font=FONT_MAIN, fg_color="#2980b9")
        self.btn_predict_custom.grid(row=12, column=0, padx=20, pady=10)

        self.btn_analysis = ctk.CTkButton(self.sidebar_frame, text="6. Analyze Accuracy", command=self.open_analysis_window, font=FONT_MAIN, fg_color="#c0392b")
        self.btn_analysis.grid(row=13, column=0, padx=20, pady=10)

        model_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        model_frame.grid(row=14, column=0, padx=20, pady=20)
        ctk.CTkButton(model_frame, text="💾 Save", width=110, command=self.save_model, fg_color="#27ae60").pack(side="left", padx=2)
        ctk.CTkButton(model_frame, text="📂 Load", width=110, command=self.load_model, fg_color="#2980b9").pack(side="right", padx=2)

    # --- CORE LOGIC ---
    def fetch_historical_data(self, s=None, e=None):
        try:
            self.log("Fetching history...")
            lat, lon = self.loc_entries["Lat"].get(), self.loc_entries["Lon"].get()
            if not s: s, e = self.sel_start.get_date(), self.sel_end.get_date()
            if e > (datetime.date.today() - datetime.timedelta(days=2)) and not s:
                return messagebox.showerror("Warning", "Archive API has 2-day lag. Pick older dates.")
            
            url = "https://archive-api.open-meteo.com/v1/archive"
            p = {"latitude": lat, "longitude": lon, "start_date": s, "end_date": e, "hourly": "temperature_2m,cloud_cover,precipitation,shortwave_radiation", "timezone": "auto"}
            r = requests.get(url, params=p); r.raise_for_status()
            d = r.json()['hourly']
            
            df = pd.DataFrame({'Timestamp': pd.to_datetime(d['time']), 'Temperature': d['temperature_2m'], 'GHI_W': d['shortwave_radiation'], 'Cloud_Cover': d['cloud_cover'], 'Precipitation': d['precipitation']})
            df['GHI'] = df['GHI_W'] / 1000.0
            
            # --- TUNED SIMULATION ---
            base = 50 * df['GHI'] * 0.14 * (1 - 0.004 * (df['Temperature'] - 25)) * 0.8
            df['Actual_Output'] = np.maximum(0, base + np.random.normal(0, 0.35, len(df)))
            df.loc[df['GHI'] < 0.05, 'Actual_Output'] = 0
            
            if self.df is not None and s: self.df = pd.concat([self.df, df]).drop_duplicates(subset='Timestamp').sort_values(by='Timestamp')
            else: self.df = df
            
            self.last_plot_type = "fetch"
            self.plot_graph(self.df, "Historical Data", False)
            self.log(f"Loaded {len(df)} rows.")
        except Exception as e: self.log(f"Error: {e}")

    def run_comparison(self):
        if self.df is None: return
        self.btn_train.configure(text="Training...")
        threading.Thread(target=self._train, daemon=True).start()

    def _train(self):
        try:
            A, eta, alpha, L = [float(self.phys_entries[k].get()) for k in ["Area (m²)", "Eff (η)", "Coeff (α)", "Loss (L)"]]
            self.df['Physical_Pred'] = np.maximum(0, A * self.df['GHI'] * eta * (1 - alpha * (self.df['Temperature'] - 25)) * (1 - L))
            
            self.df['Hour'], self.df['Month'] = self.df['Timestamp'].dt.hour, self.df['Timestamp'].dt.month
            
            X = self.df[['GHI', 'Temperature', 'Hour', 'Month']].astype(float)
            y = self.df['Actual_Output'].astype(float)

            # --- XGBOOST CPU ---
            if HAS_XGB:
                self.model_stats = xgb.XGBRegressor(n_estimators=self.n_estimators, n_jobs=-1, tree_method="hist")
                self.log("Training with XGBoost (Ryzen Multicore)")
            else:
                self.model_stats = RandomForestRegressor(n_estimators=self.n_estimators, n_jobs=-1, random_state=42)
                self.log("XGBoost not found. Using RandomForest.")

            self.model_stats.fit(X, y)
            self.df['Stat_Pred'] = np.maximum(0, self.model_stats.predict(X) * self.scaling_factor)
            self.after(0, self._train_done)
        except Exception as e: self.log(f"Train Err: {e}")

    def _train_done(self):
        self.btn_train.configure(text="2a. Train XGBoost")
        self.last_plot_type = "compare"
        self.plot_graph(self.df, "Model Comparison", comparison=True)

    def test_existing_model(self):
        if self.df is None or self.model_stats is None: return
        A, eta, alpha, L = [float(self.phys_entries[k].get()) for k in ["Area (m²)", "Eff (η)", "Coeff (α)", "Loss (L)"]]
        self.df['Physical_Pred'] = np.maximum(0, A * self.df['GHI'] * eta * (1 - alpha * (self.df['Temperature'] - 25)) * (1 - L))
        self.df['Hour'], self.df['Month'] = self.df['Timestamp'].dt.hour, self.df['Timestamp'].dt.month
        
        X = self.df[['GHI', 'Temperature', 'Hour', 'Month']].astype(float)
        self.df['Stat_Pred'] = np.maximum(0, self.model_stats.predict(X) * self.scaling_factor)
        
        self.last_plot_type = "compare"
        self.plot_graph(self.df, "Test Results", comparison=True)

    def extrapolate_data(self):
        try:
            self.log("Fetching Forecast...")
            lat, lon = self.loc_entries["Lat"].get(), self.loc_entries["Lon"].get()
            url = "https://api.open-meteo.com/v1/forecast"
            p = {"latitude": lat, "longitude": lon, "hourly": "temperature_2m,shortwave_radiation", "forecast_days": 3, "past_days": 7}
            r = requests.get(url, params=p); r.raise_for_status()
            d = r.json()['hourly']
            
            future = pd.DataFrame({'Timestamp': pd.to_datetime(d['time']), 'Temperature': d['temperature_2m'], 'GHI_W': d['shortwave_radiation']})
            future['GHI'] = future['GHI_W'] / 1000.0
            
            A, eta, alpha, L = [float(self.phys_entries[k].get()) for k in ["Area (m²)", "Eff (η)", "Coeff (α)", "Loss (L)"]]
            future['Physical_Forecast'] = np.maximum(0, A * future['GHI'] * eta * (1 - alpha * (future['Temperature'] - 25)) * (1 - L))
            
            future['Hour'], future['Month'] = future['Timestamp'].dt.hour, future['Timestamp'].dt.month
            if self.model_stats:
                X_fut = future[['GHI', 'Temperature', 'Hour', 'Month']].astype(float)
                future['Stat_Forecast'] = np.maximum(0, self.model_stats.predict(X_fut) * self.scaling_factor)
            else:
                future['Stat_Forecast'] = 0.0 
            
            trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
            # Calculate FUTURE totals only
            now = pd.Timestamp.now()
            future_only = future[future['Timestamp'] > now]
            future.attrs['total_phys'] = trapz(future_only['Physical_Forecast'], dx=1)
            future.attrs['total_stat'] = trapz(future_only.get('Stat_Forecast', future_only['Physical_Forecast']), dx=1)
            
            self.forecast_df = future
            self.last_plot_type = "forecast"
            self.plot_graph(future, "Live Forecast", is_forecast=True)
            
        except Exception as e: self.log(f"Err: {e}")

    # --- HELPERS ---
    def import_csv(self):
        f = filedialog.askopenfilename()
        if f:
            self.df = pd.read_csv(f); self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'])
            self.last_plot_type="fetch"; self.plot_graph(self.df, "Imported", False)
    def save_model(self):
        f = filedialog.asksaveasfilename()
        if f: joblib.dump(self.model_stats, f)
    def load_model(self):
        f = filedialog.askopenfilename()
        if f: self.model_stats = joblib.load(f)
    def export_data(self):
        f = filedialog.asksaveasfilename(defaultextension=".csv")
        if f and self.df is not None: self.df.to_csv(f)
    def open_report_window(self):
        if self.forecast_df is not None: ReportWindow(self, self.forecast_df.attrs.get('total_phys',0), self.forecast_df.attrs.get('total_stat',0))
    def open_analysis_window(self):
        if self.df is not None: AnalysisWindow(self, self.df)
    def open_settings(self): SettingsWindow(self)
    def open_prediction_dialog(self): PredictionRangeDialog(self, self.run_long_term_projection)
    
    def run_long_term_projection(self, s, e):
        self.extrapolate_data() 

    def update_view_duration(self, val):
        if self.last_plot_type == "fetch": self.plot_graph(self.df, "History", False)
        elif self.last_plot_type == "compare": self.plot_graph(self.df, "Comparison", is_forecast=False, comparison=True)
        elif self.last_plot_type == "forecast": 
            if self.df is not None:
                # Merge History + Forecast
                hist = self.df[['Timestamp', 'Actual_Output', 'Stat_Pred', 'GHI', 'Temperature']].copy()
                hist.rename(columns={'Actual_Output': 'Physical_Forecast', 'Stat_Pred': 'Stat_Forecast'}, inplace=True) 
                combined = pd.concat([hist, self.forecast_df], ignore_index=True).drop_duplicates('Timestamp').sort_values('Timestamp')
                self.plot_graph(combined, "Seamless Forecast", is_forecast=True)
            else:
                self.plot_graph(self.forecast_df, "Forecast", is_forecast=True)

    def slice_data_by_duration(self, data):
        selection = self.view_selector.get()
        if selection == "Full Data": return data.copy()
        
        n_hours = 24
        if "3 Days" in selection: n_hours = 72
        elif "7 Days" in selection: n_hours = 168
        
        # FIX: Backward slicing for Forecast View
        if self.last_plot_type == "forecast":
            now = pd.Timestamp.now()
            start_time = now - pd.Timedelta(hours=n_hours)
            return data[data['Timestamp'] >= start_time].copy()
        else:
            return data.tail(n_hours).copy()

    # --- PLOTTING ---
    def plot_graph(self, data, title, is_forecast=False, comparison=False):
        for w in self.graph_frame.winfo_children(): w.destroy()
        if hasattr(self, 'toolbar_frame'):
            for w in self.toolbar_frame.winfo_children(): w.destroy()
            
        plt.close('all')
        
        fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
        fig.patch.set_facecolor('#242424'); ax.set_facecolor('#2b2b2b')
        
        ax.spines['bottom'].set_color('white'); ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white'); ax.spines['right'].set_color('white')
        ax.xaxis.label.set_color('white'); ax.yaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white'); ax.tick_params(axis='y', colors='white')
        
        sub = self.slice_data_by_duration(data)
        trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
        
        if is_forecast:
            t_p = trapz(sub['Physical_Forecast'].fillna(0), dx=1)
            ax.plot(sub['Timestamp'], sub['Physical_Forecast'], label=f"Physical ({t_p:.0f} kWh)", color='#3498db')
            
            if 'Stat_Forecast' in sub:
                t_s = trapz(sub['Stat_Forecast'].fillna(0), dx=1)
                ax.plot(sub['Timestamp'], sub['Stat_Forecast'], label=f"XGBoost AI ({t_s:.0f} kWh)", color='#2ecc71')
                ax.fill_between(sub['Timestamp'], sub['Stat_Forecast'], color='#2ecc71', alpha=0.1)
                
        elif comparison:
            ax.plot(sub['Timestamp'], sub['Actual_Output'], label="Actual", color="white", alpha=0.5)
            ax.plot(sub['Timestamp'], sub['Physical_Pred'], label="Physical", linestyle="--", color='#3498db')
            ax.plot(sub['Timestamp'], sub['Stat_Pred'], label="XGBoost AI", linestyle="-.", color='#2ecc71')
        else:
            ax.plot(sub['Timestamp'], sub['GHI'], label="Sun", color="orange")
            ax.plot(sub['Timestamp'], sub['Temperature'], label="Temp", color="cyan")
        
        ax.legend(facecolor='#2b2b2b', labelcolor='white'); ax.grid(True, alpha=0.3); ax.set_title(title, color="white")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        
        self.annot = ax.text(0.02, 0.95, "", transform=ax.transAxes, bbox=dict(boxstyle="round", fc="black", ec="white"), color="white", visible=False)
        self.line = ax.axvline(x=sub['Timestamp'].iloc[0], color='white', linestyle=':', visible=False)
        
        def hover(e):
            if not e.inaxes: return
            try:
                dt = mdates.num2date(e.xdata).replace(tzinfo=None)
                row = sub.iloc[(sub['Timestamp'] - dt).abs().argsort()[:1]]
                self.line.set_xdata([row['Timestamp'], row['Timestamp']]); self.line.set_visible(True)
                t = row['Timestamp'].item().strftime('%H:%M %d %b')
                v = row['Stat_Forecast'].item() if 'Stat_Forecast' in row else row['GHI'].item()
                self.annot.set_text(f"{t}\nVal: {v:.2f}"); self.annot.set_visible(True)
                canvas.draw_idle()
            except: pass

        canvas = FigureCanvasTkAgg(fig, self.graph_frame); canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.mpl_connect("motion_notify_event", hover)
        
        # --- TOOLBAR (PLACED IN DEDICATED FRAME) ---
        toolbar = NavigationToolbar2Tk(canvas, self.toolbar_frame)
        toolbar.update()
        toolbar.pack(fill="x")

if __name__ == "__main__":
    app = SolarForecastApp()
    app.mainloop()