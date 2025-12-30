# Hybrid Solar Grid Output Forecaster

**A Comparative Analysis of Gradient Boosting Machines (XGBoost) vs. Deterministic Physical Models**

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-Regression-orange?style=for-the-badge&logo=xgboost)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
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
| **$R^2$ Score** | 0.875 | **0.995** | +13.7% |
| **RMSE** | 0.44 kWh | **0.10 kWh** | **-77.2%** |
| **Fault Detection** | No | **Yes** | Automated |

> **Critical Insight:** The Physical Model consistently over-predicted power during peak thermal loading, failing to account for documented **intermittent inverter dropouts** (SMA SMC 6000A). The XGBoost model implicitly "learned" these fault signatures, correctly predicting zero output during overheating events.

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
    git clone [https://github.com/your-username/solar-grid-forecaster.git](https://github.com/your-username/solar-grid-forecaster.git)
    cd solar-grid-forecaster
    ```

2.  **Install dependencies:**
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
    * Run the included master script to clean the raw CSV and compress it into hourly averages:
    ```bash
    python data_prep_master.py
    ```

3.  **Run the App:**
    ```bash
    python main.py
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

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
