# Hybrid Solar Grid Output Forecaster

**Predicting solar grid output: XGBoost vs. Standard Physics Models**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-Regression-orange?style=for-the-badge&logo=xgboost)
![License](https://img.shields.io/badge/License-GPLv3-red?style=for-the-badge)
![Platform](https://img.shields.io/badge/Platform-Win%20%7C%20Linux%20%7C%20Mac-lightgrey?style=for-the-badge)

## 📌 What is this?

Predicting how much power a solar farm will generate is usually done with basic physics math ($Sunlight \times Area \times Efficiency$). But real life is messy. Panels suffer from thermal heat-soak, and inverters occasionally drop out. 

This is a desktop app I built to test whether a machine learning model (**XGBoost**) can outperform standard math by learning those hardware quirks. I validated the system using 15 years of telemetry data from the BP Solar Site 12 (5.1kW) array at the Desert Knowledge Australia Solar Centre (DKASC).

---

## ⚡ The Results

Here is how the two models stacked up against each other:

| Metric | Basic Math (Physics) | XGBoost (AI) | Improvement |
| :--- | :--- | :--- | :--- |
| **$R^2$ Score** | 0.875 | **0.973** | +11.2% |
| **RMSE** | 0.50 kWh | **0.27 kWh** | **-46.0%** |
| **Fault Detection** | Nope | **Automated** | |

**Why did XGBoost win?** The standard math model kept over-predicting power on super hot days. It didn't know that the physical inverter (SMA SMC 6000A) was occasionally dropping out from overheating. The XGBoost model actually learned this hardware fault signature from the data and correctly predicted zero output during heat spikes.

To get the AI up to 97%+ accuracy, I added a few signal-processing tricks under the hood:
* **Thermal Lag:** Fed the AI the previous hour's temperature so it understands "heat soak."
* **Anomaly Filtering:** Wrote a script to strip out bad data (e.g., when a satellite sees clear skies but a local micro-cloud is actually covering the panel).
* **Time Encodings:** Added "Day of Year" and "Hour" features so the model understands the sun's actual physical arc in the sky.

---

## 🚀 Fun Fact: CPU > GPU for this

You'd think a GPU would crush this training, but for tabular telemetry data of this size (< 1 million rows), PCIe transfer latency actually slows things down. A modern CPU is significantly faster here.

| Hardware | Device | Training Time (100 Trees) | Speedup |
| :--- | :--- | :--- | :--- |
| **CPU** | **AMD Ryzen 7 7840HS** | **0.42s** | **4.4x** |
| GPU | NVIDIA RTX 4060 | 1.85s | 1.0x |

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

There are two ways to use this depending on what kind of data you have.

### Workflow A: I have the raw DKASC Ground Sensor data
1. Grab the dataset from the [DKASC Data Download](https://dkasolarcentre.com.au/download) page (Select Alice Springs -> Site 12).
2. Run the cleaner script to compress the raw CSV into hourly averages:
    ```bash
    python data_converter.py
    ```
3. Launch the dashboard:
    ```bash
    python usinggpusolar.py
    ```

### Workflow B: I only have my inverter's power logs
If you don't have ground weather sensors, you can use satellite data. 
1. Run the fetcher script. It will ping the Open-Meteo API, download historical weather data for your coordinates, and automatically merge it with your power logs.
    ```bash
    python satelite_data_fetcher.py
    ```
2. Launch the satellite-specific dashboard:
    ```bash
    python usinggpusolar2.py
    ```

### Inside the App:
* Enter your panel specs on the left (For Site 12, Area is `37.8` and Efficiency is `0.14`).
* Click **"Import CSV"** and load your processed data.
* Click **"Train XGBoost"** to generate the prediction graphs.

---

## ⚖️ License

This project uses a **Dual Licensing** model.

**1. Open Source (GPLv3):**
Free for personal, academic, and open-source projects. 
* *Catch:* If you distribute an app that uses this code, your app must also be completely open-source.

**2. Commercial License:**
Required if you want to use this in a closed-source/proprietary commercial product. 
* Keeps your source code private.
* Includes priority technical support and legal indemnification.

For commercial licensing, reach out at: [yashasakvish@gmail.com]

---

## 🔗 Citation
*(To be updated)*
