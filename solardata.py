import customtkinter as ctk
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import requests
import datetime
from tkcalendar import DateEntry
import os

# --- Configuration ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")
FONT_MAIN = ("Bahnschrift", 14)
FONT_BOLD = ("Bahnschrift", 14, "bold")
FONT_HEADER = ("Bahnschrift", 20, "bold")

# --- CUSTOM DARK CALENDAR STYLE ---
def setup_calendar_style():
    style = ttk.Style()
    style.theme_use('clam') 
    bg_dark = "#2b2b2b"      
    bg_darker = "#242424"    
    fg_white = "white"       
    accent = "#1f538d"       
    style.configure('my.DateEntry', fieldbackground=bg_darker, background=bg_dark, foreground=fg_white, arrowcolor=fg_white, bordercolor="#3E3E3E", lightcolor="#3E3E3E", darkcolor="#3E3E3E")
    style.configure('TCalendar', background=accent, foreground=fg_white, arrowcolor=fg_white, fieldbackground=bg_dark, bordercolor=bg_dark, headersbackground=bg_darker, headersforeground=fg_white, selectbackground=accent, selectforeground=fg_white, weekendbackground=bg_dark, weekendforeground=fg_white, othermonthwebackground=bg_darker, othermonthweforeground='gray')
    style.map('TCalendar', background=[('selected', accent)], foreground=[('selected', fg_white)])

# --- ANALYSIS DASHBOARD ---
class AnalysisWindow(ctk.CTkToplevel):
    def __init__(self, parent, df):
        super().__init__(parent)
        self.title("Model Accuracy Analysis")
        self.geometry("1000x600")
        self.attributes("-topmost", True)
        
        valid_df = df[(df['Actual_Output'] > 0) & (df['Physical_Pred'] > 0)].copy()
        
        if valid_df.empty:
            ctk.CTkLabel(self, text="Not enough data for analysis.", font=FONT_HEADER).pack(pady=20)
            return

        y_true = valid_df['Actual_Output']
        y_phys = valid_df['Physical_Pred']
        y_stat = valid_df['Stat_Pred']

        r2_phys = r2_score(y_true, y_phys)
        r2_stat = r2_score(y_true, y_stat)
        
        rmse_phys = np.sqrt(mean_squared_error(y_true, y_phys))
        rmse_stat = np.sqrt(mean_squared_error(y_true, y_stat))

        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.pack(fill="x", padx=20, pady=10)
        
        title = ctk.CTkLabel(header_frame, text="Prediction vs. Actual Analysis", font=FONT_HEADER, text_color="white")
        title.pack(side="top", pady=(0,10))

        metrics_txt = (f"PHYSICAL MODEL:\n  R² Score: {r2_phys:.3f}\n  RMSE: {rmse_phys:.3f} kWh\n\n"
                       f"STATISTICAL MODEL:\n  R² Score: {r2_stat:.3f}\n  RMSE: {rmse_stat:.3f} kWh")
        
        ctk.CTkLabel(header_frame, text=metrics_txt, font=("Consolas", 12), justify="left", 
                     fg_color="#333333", corner_radius=6, padx=10, pady=10).pack()

        plot_frame = ctk.CTkFrame(self)
        plot_frame.pack(fill="both", expand=True, padx=20, pady=10)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=100)
        fig.patch.set_facecolor('#2b2b2b')

        ax1.set_facecolor('#242424')
        ax1.scatter(y_true, y_phys, alpha=0.5, s=10, color='#3498db')
        lims = [0, max(y_true.max(), y_phys.max())]
        ax1.plot(lims, lims, 'r--', alpha=0.7, label="Ideal")
        ax1.set_title("Physical Model Performance", color='white')
        ax1.set_xlabel("Actual Output (kWh)", color='white')
        ax1.set_ylabel("Predicted Output (kWh)", color='white')
        ax1.tick_params(colors='white')
        ax1.grid(True, alpha=0.2)

        ax2.set_facecolor('#242424')
        ax2.scatter(y_true, y_stat, alpha=0.5, s=10, color='#2ecc71')
        ax2.plot(lims, lims, 'r--', alpha=0.7, label="Ideal")
        ax2.set_title("Statistical Model Performance", color='white')
        ax2.set_xlabel("Actual Output (kWh)", color='white')
        ax2.tick_params(colors='white')
        ax2.grid(True, alpha=0.2)

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

# --- POPUP: CUSTOM PREDICTION RANGE ---
class PredictionRangeDialog(ctk.CTkToplevel):
    def __init__(self, parent, callback):
        super().__init__(parent)
        self.callback = callback
        self.title("Select Prediction Range")
        self.geometry("400x300")
        self.resizable(False, False)
        self.attributes("-topmost", True)
        
        ctk.CTkLabel(self, text="Select Future Time Period", font=("Bahnschrift", 18, "bold")).pack(pady=(20, 5))
        ctk.CTkLabel(self, text="The AI will project historical patterns onto these dates.", font=("Bahnschrift", 12), text_color="gray").pack(pady=(0, 20))

        row_start = ctk.CTkFrame(self, fg_color="transparent")
        row_start.pack(pady=5)
        ctk.CTkLabel(row_start, text="Start:", width=60).pack(side="left")
        self.entry_start = DateEntry(row_start, width=12, background='#1f538d', foreground='white', borderwidth=2, date_pattern='dd/mm/yyyy')
        self.entry_start.set_date(datetime.date.today() + datetime.timedelta(days=1)) 
        self.entry_start.pack(side="left", padx=10)

        row_end = ctk.CTkFrame(self, fg_color="transparent")
        row_end.pack(pady=5)
        ctk.CTkLabel(row_end, text="End:", width=60).pack(side="left")
        self.entry_end = DateEntry(row_end, width=12, background='#1f538d', foreground='white', borderwidth=2, date_pattern='dd/mm/yyyy')
        self.entry_end.set_date(datetime.date.today() + datetime.timedelta(days=365)) 
        self.entry_end.pack(side="left", padx=10)

        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(pady=30)
        ctk.CTkButton(btn_frame, text="Predict", command=self.on_confirm, width=100, fg_color="#2980b9").pack(side="left", padx=10)
        ctk.CTkButton(btn_frame, text="Cancel", command=self.destroy, width=100, fg_color="#c0392b").pack(side="left", padx=10)

    def on_confirm(self):
        start = self.entry_start.get_date()
        end = self.entry_end.get_date()
        if end < start:
            messagebox.showerror("Error", "End date cannot be before Start date.")
            return
        self.callback(start, end)
        self.destroy()

# --- REPORT WINDOW CLASS ---
class ReportWindow(ctk.CTkToplevel):
    def __init__(self, parent, phys_24h, stat_24h):
        super().__init__(parent)
        self.title("Long-Term Energy Projections")
        self.geometry("800x600")
        self.resizable(True, True)
        self.attributes("-topmost", True)
        
        ctk.CTkLabel(self, text="Predicted Energy Output", font=("Bahnschrift", 24, "bold"), text_color="#3498db").pack(pady=20)
        ctk.CTkLabel(self, text="Projections based on 24h Forecast Performance", font=("Bahnschrift", 12), text_color="gray").pack(pady=(0, 20))

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

            ctk.CTkLabel(card, text=label, font=("Bahnschrift", 16, "bold"), text_color=colors[i]).pack(pady=(15, 5))
            ctk.CTkLabel(card, text=f"{p_val:,.1f} kWh", font=("Bahnschrift", 20, "bold"), text_color="white").pack()
            ctk.CTkLabel(card, text="Physical Model", font=("Bahnschrift", 10), text_color="gray").pack(pady=(0, 5))
            ctk.CTkLabel(card, text=f"{s_val:,.1f} kWh", font=("Bahnschrift", 20, "bold"), text_color="#2CC985").pack()
            ctk.CTkLabel(card, text="Statistical Model", font=("Bahnschrift", 10), text_color="gray").pack(pady=(0, 15))

        ctk.CTkButton(self, text="Close Report", command=self.destroy, fg_color="#c0392b", hover_color="#e74c3c", font=("Bahnschrift", 14)).pack(pady=20)


# --- MAIN APPLICATION CLASS ---
class SolarForecastApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        setup_calendar_style()

        self.title("Solar Grid Output Forecaster (Final)")
        self.geometry("1300x900")
        
        self.df = None
        self.forecast_df = None 
        self.current_view_df = None
        self.model_stats = None
        self.canvas = None
        self.toolbar = None
        self.toolbar_frame = None 
        self.ax = None
        self.v_line = None
        self.annot = None
        self.status_text = None 
        self.cid = None 
        self.last_plot_type = None 

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Sidebar ---
        self.sidebar_frame = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(12, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="SOLAR\nFORECASTER", font=FONT_HEADER)
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.create_location_inputs()
        self.create_phys_inputs()

        # BUTTONS
        self.btn_load = ctk.CTkButton(self.sidebar_frame, text="1. Fetch History Data", command=self.fetch_historical_data, font=FONT_MAIN, fg_color="#D35400", hover_color="#A04000")
        self.btn_load.grid(row=5, column=0, padx=20, pady=10)

        self.btn_compare = ctk.CTkButton(self.sidebar_frame, text="2. Run Models", command=self.run_comparison, font=FONT_MAIN, fg_color="transparent", border_width=2)
        self.btn_compare.grid(row=6, column=0, padx=20, pady=10)

        self.btn_forecast = ctk.CTkButton(self.sidebar_frame, text="3. Real Forecast (24h)", command=self.extrapolate_data, font=FONT_MAIN, fg_color="#2CC985", text_color="white")
        self.btn_forecast.grid(row=7, column=0, padx=20, pady=10)
        
        self.btn_report = ctk.CTkButton(self.sidebar_frame, text="4. View Report", command=self.open_report_window, font=FONT_MAIN, fg_color="#8e44ad", hover_color="#9b59b6")
        self.btn_report.grid(row=8, column=0, padx=20, pady=10)

        self.btn_predict_custom = ctk.CTkButton(self.sidebar_frame, text="5. Predict Custom Range", command=self.open_prediction_dialog, font=FONT_MAIN, fg_color="#2980b9", hover_color="#3498db")
        self.btn_predict_custom.grid(row=9, column=0, padx=20, pady=10)

        self.btn_analysis = ctk.CTkButton(self.sidebar_frame, text="6. Analyze Accuracy", command=self.open_analysis_window, font=FONT_MAIN, fg_color="#c0392b", hover_color="#e74c3c")
        self.btn_analysis.grid(row=10, column=0, padx=20, pady=10)

        # Removed Log Label and Textbox from Sidebar

        # --- Main Area (Right Side) ---
        self.right_frame = ctk.CTkFrame(self, corner_radius=10, fg_color="transparent")
        self.right_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.right_frame.grid_rowconfigure(0, weight=1)
        self.right_frame.grid_columnconfigure(0, weight=1)

        # --- TAB VIEW ---
        self.tabview = ctk.CTkTabview(self.right_frame)
        self.tabview.pack(fill="both", expand=True)
        
        self.tab_graphs = self.tabview.add("Graphs")
        self.tab_logs = self.tabview.add("System Log")
        
        # Configure Grid for Tabs
        self.tab_graphs.grid_columnconfigure(0, weight=1)
        self.tab_graphs.grid_rowconfigure(1, weight=1)
        
        self.tab_logs.grid_columnconfigure(0, weight=1)
        self.tab_logs.grid_rowconfigure(0, weight=1)

        # --- TAB 1: GRAPHS CONTENT ---
        self.view_controls_frame = ctk.CTkFrame(self.tab_graphs, height=40)
        self.view_controls_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        ctk.CTkLabel(self.view_controls_frame, text="View Duration:", font=("Bahnschrift", 12)).pack(side="left", padx=10)
        
        self.view_selector = ctk.CTkSegmentedButton(self.view_controls_frame, 
                                                    values=["Full Data", "Last 7 Days", "Last 3 Days", "Last 24h"],
                                                    command=self.update_view_duration)
        self.view_selector.set("Full Data")
        self.view_selector.pack(side="left", padx=10, pady=5)

        self.graph_frame = ctk.CTkFrame(self.tab_graphs, corner_radius=10)
        self.graph_frame.grid(row=1, column=0, sticky="nsew")
        
        self.placeholder_lbl = ctk.CTkLabel(self.graph_frame, text="1. Select Dates\n2. Fetch Data\n3. Use Toolbar to Zoom/Pan", font=FONT_HEADER, text_color="gray")
        self.placeholder_lbl.place(relx=0.5, rely=0.5, anchor="center")

        # --- TAB 2: SYSTEM LOG CONTENT ---
        self.textbox = ctk.CTkTextbox(self.tab_logs, font=("Consolas", 13))
        self.textbox.pack(fill="both", expand=True, padx=10, pady=10)

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        plt.close('all')
        self.destroy()

    def create_location_inputs(self):
        frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        frame.grid(row=1, column=0, padx=20, pady=5, sticky="ew")
        ctk.CTkLabel(frame, text="Location & Date Range", font=FONT_BOLD).pack(anchor="w")
        
        self.loc_entries = {}
        coords = [("Lat", "12.97"), ("Lon", "79.16")]
        for lbl, val in coords:
            row = ctk.CTkFrame(frame, fg_color="transparent")
            row.pack(fill="x", pady=2)
            ctk.CTkLabel(row, text=lbl, width=60, anchor="w").pack(side="left")
            entry = ctk.CTkEntry(row, height=25)
            entry.insert(0, val)
            entry.pack(side="right", expand=True, fill="x")
            self.loc_entries[lbl] = entry

        today = datetime.date.today()
        default_start = today - datetime.timedelta(days=10)
        default_end = today

        row_start = ctk.CTkFrame(frame, fg_color="transparent")
        row_start.pack(fill="x", pady=(10, 2))
        ctk.CTkLabel(row_start, text="Start Date:", width=80, anchor="w").pack(side="left")
        self.date_start_entry = DateEntry(row_start, width=12, background='#1f538d', foreground='white', borderwidth=2, date_pattern='dd/mm/yyyy')
        self.date_start_entry.set_date(default_start)
        self.date_start_entry.pack(side="left", padx=10)

        row_end = ctk.CTkFrame(frame, fg_color="transparent")
        row_end.pack(fill="x", pady=2)
        ctk.CTkLabel(row_end, text="End Date:", width=80, anchor="w").pack(side="left")
        self.date_end_entry = DateEntry(row_end, width=12, background='#1f538d', foreground='white', borderwidth=2, date_pattern='dd/mm/yyyy')
        self.date_end_entry.set_date(default_end)
        self.date_end_entry.pack(side="left", padx=10)

    def create_phys_inputs(self):
        frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        ctk.CTkLabel(frame, text="Physical Params", font=FONT_BOLD).pack(anchor="w")
        self.phys_entries = {}
        params = [("Area (m²)", "50"), ("Eff (η)", "0.18"), ("Coeff (α)", "0.004"), ("Loss (L)", "0.14")]
        for lbl, val in params:
            row = ctk.CTkFrame(frame, fg_color="transparent")
            row.pack(fill="x", pady=2)
            ctk.CTkLabel(row, text=lbl).pack(side="left")
            entry = ctk.CTkEntry(row, width=80, height=25)
            entry.insert(0, val)
            entry.pack(side="right")
            self.phys_entries[lbl] = entry

    def log(self, message):
        print(message) 
        self.textbox.insert("end", f"> {message}\n")
        self.textbox.see("end")

    def get_weather_status(self, row):
        rain = row.get('Precipitation', 0)
        clouds = row.get('Cloud_Cover', 0)
        ghi = row.get('GHI', 0)
        is_daytime = ghi > 0.05 
        if rain > 0.1: return "Rainy 🌧"
        elif clouds > 70: return "Cloudy ☁"
        elif clouds > 30: return "Partly Cloudy ⛅" if is_daytime else "Partly Cloudy 🌙"
        else: return "Sunny ☀" if is_daytime else "Clear Night 🌙"

    # --- VIEW LOGIC ---
    def update_view_duration(self, value):
        if self.last_plot_type == "fetch" and self.df is not None:
            self.plot_graph(self.df, f"Historical Data ({value})", is_forecast=False, comparison=False)
        elif self.last_plot_type == "compare" and self.df is not None:
            self.plot_graph(self.df, f"Model Comparison ({value})", is_forecast=False, comparison=True)
        elif self.last_plot_type == "forecast" and self.forecast_df is not None:
            if value == "Last 24h":
                self.plot_graph(self.forecast_df, "24h Forecast (Open-Meteo)", is_forecast=True, comparison=False)
            else:
                history = self.df.copy()
                history['Physical_Forecast'] = history.get('Physical_Pred', np.nan)
                history['Stat_Forecast'] = history.get('Stat_Pred', np.nan)
                
                combined = pd.concat([history, self.forecast_df], ignore_index=True)
                combined = combined.sort_values(by='Timestamp')
                
                # LINE BREAKER
                combined['delta'] = combined['Timestamp'].diff()
                mask = combined['delta'] > pd.Timedelta(hours=2)
                if mask.any():
                    gap_rows = []
                    for idx in combined.index[mask]:
                        gap_row = combined.loc[idx].copy()
                        gap_row[:] = np.nan
                        gap_row['Timestamp'] = combined.loc[idx]['Timestamp'] - pd.Timedelta(minutes=1)
                        gap_rows.append(gap_row)
                    if gap_rows:
                        gap_df = pd.DataFrame(gap_rows)
                        combined = pd.concat([combined, gap_df], ignore_index=True)
                        combined = combined.sort_values(by='Timestamp')

                self.plot_graph(combined, f"History + Forecast ({value})", is_forecast=True, comparison=False)

    def slice_data_by_duration(self, data):
        selection = self.view_selector.get()
        if selection == "Full Data": return data.copy()
        elif selection == "Last 7 Days": return data.tail(7 * 24).copy()
        elif selection == "Last 3 Days": return data.tail(3 * 24).copy()
        elif selection == "Last 24h": return data.tail(24).copy()
        return data.copy()

    # --- FETCH DATA ---
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
                user_start = self.date_start_entry.get_date()
                user_end = self.date_end_entry.get_date()
                start_str = user_start.strftime("%Y-%m-%d")
                end_str = user_end.strftime("%Y-%m-%d")
                self.log(f"Fetching Historical Data: {start_str} to {end_str}")
            
            url = "https://archive-api.open-meteo.com/v1/archive"
            params = {
                "latitude": lat, "longitude": lon,
                "start_date": start_str, "end_date": end_str,
                "hourly": "temperature_2m,cloud_cover,precipitation,shortwave_radiation",
                "timezone": "auto"
            }
            response = requests.get(url, params=params, timeout=15)
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
            
            if new_df.empty:
                tk.messagebox.showerror("No Data", "Open-Meteo returned no data.")
                return

            true_energy = (50 * new_df['GHI'] * 0.17 * (1 - 0.004 * (new_df['Temperature'] - 25))) * 0.9
            noise = np.random.normal(0, 0.2, len(new_df))
            new_df['Actual_Output'] = np.maximum(0, true_energy + noise)
            
            if self.df is not None and start_date:
                self.df = pd.concat([self.df, new_df]).drop_duplicates(subset='Timestamp').sort_values(by='Timestamp')
                self.log(f"Merged {len(new_df)} new rows into training data.")
            else:
                self.df = new_df
                self.last_plot_type = "fetch"
                self.log(f"SUCCESS! {len(self.df)} rows loaded.")
                self.plot_graph(self.df, f"Historical Data", is_forecast=False)

        except Exception as e:
            self.log(f"Fetch Error: {e}")
            tk.messagebox.showerror("Fetch Error", str(e))

    def run_comparison(self):
        self.log("Running Models...")
        if self.df is None or self.df.empty:
            tk.messagebox.showwarning("Order Error", "Click Button 1 first!")
            return
        try:
            A = float(self.phys_entries["Area (m²)"].get())
            eta = float(self.phys_entries["Eff (η)"].get())
            alpha = float(self.phys_entries["Coeff (α)"].get())
            L = float(self.phys_entries["Loss (L)"].get())

            phys_calc = A * self.df['GHI'] * eta * (1 - alpha * (self.df['Temperature'] - 25)) * (1 - L)
            self.df['Physical_Pred'] = np.maximum(0, phys_calc)

            X = self.df[['GHI', 'Temperature']]
            y = self.df['Actual_Output']
            self.model_stats = LinearRegression()
            self.model_stats.fit(X, y)
            
            stat_pred = self.model_stats.predict(X)
            self.df['Stat_Pred'] = np.maximum(0, stat_pred)

            mae_p = mean_absolute_error(y, self.df['Physical_Pred'])
            mae_s = mean_absolute_error(y, self.df['Stat_Pred'])
            self.log(f"Physical MAE: {mae_p:.3f}")
            self.log(f"Statistical MAE: {mae_s:.3f}")

            self.last_plot_type = "compare"
            self.plot_graph(self.df, "Model Comparison", comparison=True)
        except Exception as e:
            tk.messagebox.showerror("Model Error", str(e))

    def extrapolate_data(self):
        self.log("Fetching Real-Time Forecast (Open-Meteo)...")
        if self.model_stats is None:
            tk.messagebox.showwarning("Order Error", "Click Button 2 first!")
            return
        try:
            lat = self.loc_entries["Lat"].get()
            lon = self.loc_entries["Lon"].get()
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": lat, "longitude": lon,
                "hourly": "temperature_2m,cloud_cover,precipitation,shortwave_radiation",
                "timezone": "auto", "forecast_days": 3 
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            hourly = data['hourly']
            timestamps = pd.to_datetime(hourly['time'])
            forecast_raw = pd.DataFrame({
                'Timestamp': timestamps, 'Temperature': hourly['temperature_2m'],
                'Cloud_Cover': hourly['cloud_cover'], 'Precipitation': hourly['precipitation'],
                'GHI_W': hourly['shortwave_radiation']
            })

            now = pd.Timestamp.now()
            future_df = forecast_raw[forecast_raw['Timestamp'] >= now].head(24).copy()
            if future_df.empty: future_df = forecast_raw.head(24).copy()

            future_df['GHI'] = future_df['GHI_W'] / 1000.0
            A = float(self.phys_entries["Area (m²)"].get())
            eta = float(self.phys_entries["Eff (η)"].get())
            alpha = float(self.phys_entries["Coeff (α)"].get())
            L = float(self.phys_entries["Loss (L)"].get())

            phys_forecast = A * future_df['GHI'] * eta * (1 - alpha * (future_df['Temperature'] - 25)) * (1 - L)
            future_df['Physical_Forecast'] = np.maximum(0, phys_forecast)
            stat_forecast = self.model_stats.predict(future_df[['GHI', 'Temperature']])
            future_df['Stat_Forecast'] = np.maximum(0, stat_forecast)

            if hasattr(np, 'trapezoid'):
                total_phys = np.trapezoid(future_df['Physical_Forecast'], dx=1)
                total_stat = np.trapezoid(future_df['Stat_Forecast'], dx=1)
            else:
                total_phys = np.trapz(future_df['Physical_Forecast'], dx=1)
                total_stat = np.trapz(future_df['Stat_Forecast'], dx=1)
            
            self.log(f">> REAL FORECAST (Next 24h):")
            self.log(f"   Physical: {total_phys:.2f} kWh")
            self.log(f"   Statistical: {total_stat:.2f} kWh")

            future_df.attrs['total_phys'] = total_phys
            future_df.attrs['total_stat'] = total_stat
            self.forecast_df = future_df 
            self.last_plot_type = "forecast"
            self.view_selector.set("Last 24h")
            self.update_view_duration("Last 24h")
        except Exception as e:
            self.log(f"Forecast Error: {e}")

    # --- CUSTOM RANGE PREDICTION LOGIC WITH AUTO-FETCH ---
    def open_prediction_dialog(self):
        PredictionRangeDialog(self, self.run_custom_prediction)

    def run_custom_prediction(self, start_date, end_date):
        self.log(f"Analyzing Prediction Range: {start_date} to {end_date}...")
        
        try:
            req_dates = pd.date_range(start=start_date, end=end_date, freq='MS')
            required_months = set(req_dates.month)
            required_months.add(end_date.month)
            
            if self.df is None or self.df.empty:
                existing_months = set()
            else:
                existing_months = set(self.df['Timestamp'].dt.month.unique())
            
            missing_months = required_months - existing_months
            
            if missing_months:
                self.log(f"Missing data for months: {missing_months}. Auto-learning...")
                today = datetime.date.today()
                fetch_year = today.year - 1
                fetch_start = datetime.date(fetch_year, 1, 1)
                fetch_end = datetime.date(fetch_year, 12, 31)
                
                self.fetch_historical_data(fetch_start, fetch_end)
                self.run_comparison()

            self.df['Month'] = self.df['Timestamp'].dt.month
            self.df['Hour'] = self.df['Timestamp'].dt.hour
            monthly_profile = self.df.groupby(['Month', 'Hour'])[['GHI', 'Temperature']].mean().reset_index()

            s_dt = datetime.datetime.combine(start_date, datetime.time.min)
            e_dt = datetime.datetime.combine(end_date, datetime.time.max)
            future_dates = pd.date_range(start=s_dt, end=e_dt, freq="h")
            
            future_df = pd.DataFrame({'Timestamp': future_dates})
            future_df['Month'] = future_df['Timestamp'].dt.month
            future_df['Hour'] = future_df['Timestamp'].dt.hour

            future_df = pd.merge(future_df, monthly_profile, on=['Month', 'Hour'], how='left').fillna(0)
            
            A = float(self.phys_entries["Area (m²)"].get())
            eta = float(self.phys_entries["Eff (η)"].get())
            alpha = float(self.phys_entries["Coeff (α)"].get())
            L = float(self.phys_entries["Loss (L)"].get())

            phys_forecast = A * future_df['GHI'] * eta * (1 - alpha * (future_df['Temperature'] - 25)) * (1 - L)
            future_df['Physical_Forecast'] = np.maximum(0, phys_forecast)
            
            try:
                stat_forecast = self.model_stats.predict(future_df[['GHI', 'Temperature']])
                future_df['Stat_Forecast'] = np.maximum(0, stat_forecast)
            except:
                future_df['Stat_Forecast'] = 0

            if hasattr(np, 'trapezoid'):
                total_energy = np.trapezoid(future_df['Physical_Forecast'], dx=1)
            else:
                total_energy = np.trapz(future_df['Physical_Forecast'], dx=1)

            self.log(f">> PREDICTION COMPLETE")
            self.log(f"   Total: {total_energy:,.0f} kWh")
            
            future_df.attrs['total_phys'] = total_energy
            future_df.attrs['total_stat'] = 0 
            
            self.forecast_df = future_df
            self.last_plot_type = "forecast"
            self.view_selector.set("Full Data")
            
            days_count = (end_date - start_date).days
            title = f"Prediction: {days_count} Days (Total: {total_energy:,.0f} kWh)"
            self.plot_graph(future_df, title, is_forecast=True)

        except Exception as e:
            self.log(f"Prediction Error: {e}")

    def open_report_window(self):
        if self.forecast_df is None:
            tk.messagebox.showwarning("No Forecast", "Please run Step 3 first.")
            return
        p_total = self.forecast_df.attrs.get('total_phys', 0)
        s_total = self.forecast_df.attrs.get('total_stat', 0)
        ReportWindow(self, p_total, s_total)

    def open_analysis_window(self):
        if self.df is None or 'Physical_Pred' not in self.df:
            tk.messagebox.showwarning("No Model", "Run Step 2 (Run Models) first.")
            return
        AnalysisWindow(self, self.df)
    #  PLOTTING FUNCTION
    def plot_graph(self, data, title, is_forecast=False, comparison=False):
        if self.canvas:
            if self.cid: self.canvas.mpl_disconnect(self.cid)
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        
        if hasattr(self, 'toolbar_frame') and self.toolbar_frame:
            self.toolbar_frame.destroy()
            self.toolbar_frame = None
        
        if self.toolbar:
            self.toolbar.destroy()
            self.toolbar = None
            
        plt.close('all')

        font_path = 'C:/Windows/Fonts/seguiemj.ttf'
        if os.path.exists(font_path):
            emoji_font = fm.FontProperties(fname=font_path)
        else:
            emoji_font = fm.FontProperties()

        plt.style.use('dark_background')
        fig, self.ax = plt.subplots(figsize=(6, 5), dpi=100)
        fig.patch.set_facecolor('#242424')
        self.ax.set_facecolor('#2b2b2b')
        fig.subplots_adjust(bottom=0.25)

        self.current_view_df = self.slice_data_by_duration(data)
        if self.current_view_df.empty:
            self.current_view_df = data.copy()

        # DYNAMIC CALCULATION
        if is_forecast:
            if hasattr(np, 'trapezoid'):
                t_phys = np.trapezoid(self.current_view_df['Physical_Forecast'], dx=1)
            else:
                t_phys = np.trapz(self.current_view_df['Physical_Forecast'], dx=1)
            
            if 'Stat_Forecast' in self.current_view_df and self.current_view_df['Stat_Forecast'].sum() > 0:
                if hasattr(np, 'trapezoid'):
                    t_stat = np.trapezoid(self.current_view_df['Stat_Forecast'], dx=1)
                else:
                    t_stat = np.trapz(self.current_view_df['Stat_Forecast'], dx=1)
            else:
                t_stat = 0

            use_fill = len(self.current_view_df) < 1000 
            line_width = 1 if not use_fill else 2

            self.ax.plot(self.current_view_df['Timestamp'], self.current_view_df['Physical_Forecast'], 
                         label=f'Physical ({t_phys:,.1f} kWh)', 
                         marker='o' if use_fill else None, 
                         linewidth=line_width,
                         color='#3498db', 
                         markersize=3)
            
            if t_stat > 0:
                self.ax.plot(self.current_view_df['Timestamp'], self.current_view_df['Stat_Forecast'], 
                             label=f'Statistical ({t_stat:,.1f} kWh)', 
                             marker='x' if use_fill else None, 
                             linewidth=line_width,
                             color='#2ecc71', 
                             markersize=3)
            
            if use_fill:
                self.ax.fill_between(self.current_view_df['Timestamp'], self.current_view_df['Physical_Forecast'], alpha=0.1, color='#3498db')
                if t_stat > 0:
                    self.ax.fill_between(self.current_view_df['Timestamp'], self.current_view_df['Stat_Forecast'], alpha=0.1, color='#2ecc71')
            
            self.ax.set_ylabel("Energy (kWh)") 

        elif comparison:
            subset = self.current_view_df
            self.ax.plot(subset['Timestamp'], subset['Actual_Output'], label='Actual', color='white', alpha=0.6, linewidth=2)
            self.ax.plot(subset['Timestamp'], subset['Physical_Pred'], label='Physical', linestyle='--', color='#3498db')
            self.ax.plot(subset['Timestamp'], subset['Stat_Pred'], label='Statistical', linestyle='-.', color='#2ecc71')
            self.ax.set_ylabel("Energy (kWh)")
        else:
            subset = self.current_view_df
            self.ax.plot(subset['Timestamp'], subset['GHI'], label='GHI', color='orange', linewidth=1)
            self.ax.plot(subset['Timestamp'], subset['Temperature'], label='Temp', color='cyan', linewidth=1)
            self.ax.set_ylabel("Values (kW/m² or °C)")

        self.v_line = self.ax.axvline(x=self.current_view_df['Timestamp'].iloc[0], color='white', linestyle=':', alpha=0.8)
        self.v_line.set_visible(False)
        
        self.annot = self.ax.text(0.02, 0.95, "", transform=self.ax.transAxes,
                                  bbox=dict(boxstyle="round", fc="black", ec="white", alpha=0.8),
                                  color="white", verticalalignment='top')
        self.annot.set_visible(False)

        self.status_text = self.ax.text(0.5, -0.25, "Hover over graph for status", 
                                        transform=self.ax.transAxes,
                                        ha='center', va='top', 
                                        color='#2CC985', fontsize=12,
                                        fontproperties=emoji_font)

        self.ax.set_xlabel("Date / Time")
        self.ax.set_title(title, color='white', pad=10)
        self.ax.legend(loc='upper right')
        self.ax.grid(True, alpha=0.3)
        
        if len(self.current_view_df) > 8000: 
             date_fmt = mdates.DateFormatter('%b')
        elif len(self.current_view_df) > 120:
             date_fmt = mdates.DateFormatter('%d %b') 
        else:
             date_fmt = mdates.DateFormatter('%d %b\n%H:%M') 
        self.ax.xaxis.set_major_formatter(date_fmt)
        plt.xticks(rotation=0)

        def hover(event):
            if not event or not event.inaxes or event.inaxes != self.ax:
                if self.v_line: self.v_line.set_visible(False)
                if self.annot: self.annot.set_visible(False)
                if self.status_text: self.status_text.set_text("Hover over graph for status")
                if self.canvas: self.canvas.draw_idle()
                return
            if event.xdata is None: return
            try:
                click_date = mdates.num2date(event.xdata).replace(tzinfo=None)
                df = self.current_view_df
                if df is None or df.empty: return
                closest_idx = (df['Timestamp'] - click_date).abs().idxmin()
                row = df.loc[closest_idx]
                
                if pd.isna(row.get('GHI', np.nan)) and pd.isna(row.get('Physical_Forecast', np.nan)):
                    return

                self.v_line.set_xdata([row['Timestamp'], row['Timestamp']])
                self.v_line.set_visible(True)
                t_str = row['Timestamp'].strftime('%H:%M\n%d %b')
                weather = self.get_weather_status(row) 
                self.status_text.set_text(f"Status: {weather}")
                msg = f"{t_str}\n"
                if is_forecast:
                    if 'Physical_Forecast' in row and not pd.isna(row['Physical_Forecast']): 
                         msg += f"Phys: {row['Physical_Forecast']:.2f} kWh\n"
                    if 'Stat_Forecast' in row and not pd.isna(row['Stat_Forecast']): 
                         msg += f"Stat: {row['Stat_Forecast']:.2f} kWh\n"
                elif comparison:
                    if 'Actual_Output' in row and not pd.isna(row['Actual_Output']): 
                         msg += f"Actual: {row['Actual_Output']:.2f} kWh\n"
                    if 'Physical_Pred' in row and not pd.isna(row['Physical_Pred']): 
                         msg += f"Phys: {row['Physical_Pred']:.2f} kWh\n"
                    if 'Stat_Pred' in row and not pd.isna(row['Stat_Pred']): 
                         msg += f"Stat: {row['Stat_Pred']:.2f} kWh\n"
                else:
                    if 'GHI' in row: msg += f"GHI: {row['GHI']:.2f} kW/m²\n"
                    if 'Temperature' in row: msg += f"Temp: {row['Temperature']:.1f} °C"
                self.annot.set_text(msg)
                self.annot.set_visible(True)
                self.canvas.draw_idle()
            except Exception: pass

        self.canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        self.cid = self.canvas.mpl_connect("motion_notify_event", hover)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(10, 0))

        self.toolbar_frame = ctk.CTkFrame(self.graph_frame, fg_color="transparent")
        self.toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.graph_frame.update_idletasks()

if __name__ == "__main__":
    app = SolarForecastApp()
    app.mainloop()
    
    