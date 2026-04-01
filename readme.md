# Hybrid Solar Grid Output Forecaster

**Predicting solar output by upgrading satellite data with XGBoost.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-Regression-orange?style=for-the-badge&logo=xgboost)
![License](https://img.shields.io/badge/License-GPLv3-red?style=for-the-badge)
![Platform](https://img.shields.io/badge/Platform-Win%20%7C%20Linux%20%7C%20Mac-lightgrey?style=for-the-badge)

## What is this?
If you want to accurately predict how much power a solar farm will generate, you usually need expensive, on-site weather stations. Free satellite weather data exists, but it's too coarse—a 10km satellite grid can't see the specific cloud shading your 5-meter solar panel.

This desktop app fixes that. It's a machine learning pipeline that uses **XGBoost** to act as a "Virtual Sensor." It takes low-res satellite data, engineers some clever features (like thermal lag and scattered light), and trains an AI to predict power output with the accuracy of expensive ground sensors.

Built and validated using 15 years of real telemetry data from the **BP Solar Site 12 (5.1kW)** array at the Desert Knowledge Australia Solar Centre (DKASC).

---

## The Results
Standard math equations using satellite data are pretty terrible at this (hitting around 46% accuracy because of timezone mismatches and missing localized data). By feeding advanced satellite features into XGBoost, we pushed that accuracy to **93.3%**—getting incredibly close to the theoretical limit of having an actual physical sensor on the roof.

| Model | Data Source | R² Score | RMSE (kWh) |
| :--- | :--- | :--- | :--- |
| Basic Math (HWB Equations) | Satellite Data | 0.460 | 1.03 |
| Basic Math (HWB Equations) | Ground Sensors | 0.812 | 0.94 |
| **This Project (XGBoost)** | **Satellite Data** | **0.933** | **0.41** |
| Theoretical Max (XGBoost) | Ground Sensors | 0.973 | 0.24 |

---

## How it works under the hood
To get satellite data to perform this well, the code does a few specific things to clean and prep the data before the AI even sees it:
1. **Grabs better weather data:** It doesn't just look at temperature and standard sunlight (GHI). It pulls Direct Normal Irradiance (DNI), Diffuse Horizontal Irradiance (DHI), and Cloud Cover so the AI knows exactly if the light is direct or scattered.
2. **Auto-Aligns Timezones:** Satellite data is often in UTC; your inverter logs are in local time. The script runs a cross-correlation algorithm to perfectly align the sun's peak with the inverter's peak automatically.
3. **Drops Dead Data:** If the sun is blazing but the inverter was turned off for maintenance, the script drops those rows. We don't want the AI learning that "perfect sun = zero power."
4. **Accounts for Heat Soak:** Panels get hot and lose efficiency. We feed the AI the previous hour's temperature so it understands thermal lag.

---

## CPU > GPU for this
You don't need a massive GPU to run this. In fact, our benchmarks showed that modern CPUs crush GPUs for this specific tabular dataset because of PCIe transfer latency.

| Hardware | Device | Training Time (100 Trees) |
| :--- | :--- | :--- |
| **CPU** | **AMD Ryzen 7 7840HS** | **0.42s** |
| GPU | NVIDIA RTX 4060 | 1.85s |

---

## 🛠️ Installation

**Prerequisites:** Python 3.10+ and Git.

1.  **Clone the repo:**
    ```bash
    git clone [https://github.com/yashas-yeet/solar_projection_using_xgboost.git](https://github.com/yashas-yeet/solar_projection_using_xgboost.git)
    cd solar_projection_using_xgboost
    ```

2.  **Create a virtual environment (Recommended):**
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

## 💻 Usage & Workflows

There are two ways to use this depending on what data you have.

### Workflow A: "I only have my inverter's power data"
Use the Automated Weather Backcaster. Point it to your CSV, enter your coordinates, and it will ping the Open-Meteo API, download the advanced weather data, auto-align the timezones, and merge it all together.
```bash
python ghi_temp.py
