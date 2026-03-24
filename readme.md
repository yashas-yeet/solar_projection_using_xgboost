# Hybrid Solar Grid Output Forecaster

**A Comparative Analysis of Gradient Boosting Machines (XGBoost) vs. Deterministic Physical Models**

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-Regression-orange?style=for-the-badge&logo=xgboost)
![License](https://img.shields.io/badge/License-GPLv3-red?style=for-the-badge)
![Platform](https://img.shields.io/badge/Platform-Win%20%7C%20Linux-lightgrey?style=for-the-badge)

## 📌 Overview

This repository contains the source code for the **"Hybrid Solar Grid Output Forecaster,"** a desktop application designed to benchmark predictive accuracy on legacy photovoltaic (PV) infrastructure.

The system performs a head-to-head comparison between:
1.  **A Deterministic Physical Model:** Based on the isotropic sky model ($P_{phys} = GHI \times A \times \eta$).
2.  **A Stochastic AI Model:** Utilizing **XGBoost** (Extreme Gradient Boosting) to learn non-linear degradation patterns.

The software was validated using 15 years of high-fidelity telemetry data from the **BP Solar Site 12 (5.1kW)** array at the Desert Knowledge Australia Solar Centre (DKASC).

---

## ⚡ Key Findings (The "Efficiency Gap")

| Metric | Physical Model | XGBoost (AI) | Improvement |
| :--- | :--- | :--- | :--- |
| **$R^2$ Score** | 0.875 | **0.973** | +11.2% |
| **RMSE** | 0.50 kWh | **0.27 kWh** | **-46.0%** |
| **Fault Detection** | No | **Yes** | Automated |

> **Critical Insight:** The Physical Model consistently over-predicted power during peak thermal loading, failing to account for documented **intermittent inverter dropouts** (SMA SMC 6000A). The XGBoost model implicitly "learned" these fault signatures, correctly predicting zero output during overheating events.

To achieve 97%+ accuracy with noisy satellite data, the pipeline implements three critical signal-processing layers:

**Thermal Lag Integration:** Incorporates Temp_Lag (T-1) to account for the PV module's heat-soak capacity.

**Satellite Anomaly Filtering:** A 95th-percentile filter that identifies and removes "Satellite Mismatches" (where space sensors see sun but ground sensors see local micro-clouds).

**Temporal Encoding:** Uses DayOfYear and Hour to map the non-linear solar arc of the Southern Hemisphere.
---

## 🚀 Hardware Optimization (Edge Computing)

Contrary to the industry standard of using heavy GPU acceleration, our benchmarks prove that **modern high-IPC CPUs** are superior for single-site solar telemetry forecasting ($N < 10^6$ rows).

| Hardware | Device | Training Time (100 Trees) | Speedup |
| :--- | :--- | :--- | :--- |
| **CPU** | **AMD Ryzen 7 7840HS** | **0.42s** | **4.4x** |
| **GPU** | NVIDIA RTX 4060 | 1.85s | 1.0x |

*GPU training suffered from PCIe data transfer latency which outweighed the benefits of parallel compute for tabular data of this size.*

---

## 🛠️ Installation

### Prerequisites
* Python 3.10 or higher
* Git

### Setup
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yashas-yeet/solar_projection_using_xgboost.git](https://github.com/yashas-yeet/solar_projection_using_xgboost.git)
    cd solar_projection_using_xgboost
    ```

2.  **Create a virtual environment (Optional but Recommended):**
    ```bash
    python -m venv .venv
    # Windows:
    .venv\Scripts\activate
    # Mac/Linux:
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 💻 Usage

1.  **Download Data:**
    * Visit the [DKASC Data Download](https://dkasolarcentre.com.au/download).
    * Select **Alice Springs** -> **Site 12 (BP Solar 5.1kW Mono-Si)**.
    * Download the CSV (ensure it includes Weather Data).

2.  **Prepare Data:**
    * Run the included master scripts to clean the raw CSV and compress it into hourly averages:
    ```bash
    data_converter.py
    ```
    * If user only has power output data they can use
    ```bash
    ghi_temp.py
    ```
    to get those data sets for themselves

3.  **Run the App:**
    ```bash
    python usinggpusolar.py
    ```

4.  **Configure Simulation:**
    * **Area:** `37.8` (Site 12 Official Specs)
    * **Efficiency:** `0.14` (BP 4170N Panel Specs)
    * Click **"Import CSV"** -> Select your cleaned `Final_Hourly_Ready.csv`.
    * Click **"Train XGBoost"** to view the results.

---

## 📚 Methodology

The application implements a **Hybrid Splicing Architecture**:
* **Historical Layer:** Visualizes actual generation (`Active_Power`) from the CSV.
* **Forecast Layer:** Uses the trained XGBoost regressor to predict future generation based on `GHI` (Global Horizontal Irradiance) and `Ambient_Temperature`.
* **Physics Layer:** Calculates theoretical maximums using:
    $$P_{phys} = GHI \times Area \times \eta \times (1 - Loss_{thermal})$$

---

## ⚖️ Licensing & Commercial Use

This project utilizes a **Dual Licensing** model to ensure availability for academic research while supporting commercial viability.

**1. Open Source Use (GPL v3):**
This software is free for academic research, open-source projects, and personal use under the **GNU General Public License v3 (GPLv3)**.
* ✅ You can modify and distribute this code.
* ⚠️ **Condition:** If you distribute an app that uses this code, **your entire app must also be open-source under GPLv3.**

**2. Commercial / Proprietary Use:**
If you wish to use this software in a proprietary (closed-source) commercial product **without** releasing your source code, you must purchase a **Commercial License**.

Benefits of the Commercial License:
* Release your software as proprietary/closed-source.
* Receive priority technical support.
* Legal indemnification.

📩 **Contact for Commercial Licensing:** [yashasakvish@gmail.com]

---

## 🔗 Citation

*to be updated