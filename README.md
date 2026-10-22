# Centrifugal Pump Predictive Maintenance 💧⚙️

## Project Overview
This project simulates sensor data from a centrifugal pump in a smart manufacturing environment. The goal is to detect anomalies (such as pipeline leakage) by analyzing time-series data from IoT sensors, eventually building a predictive maintenance model.

## Folder Structure
- `data/`: Contains the generated dataset (`pump_sensor_data.csv`) and EDA plots.
- `src/`: Contains the Python source code.
  - `data_generator.py`: Simulates normal and leakage conditions for Pressure, Flow, and Vibration sensors.
  - `eda_analysis.py`: Performs Exploratory Data Analysis (EDA) and visualizes the sensor signals.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Generate Data (optional, as data is already provided):
   ```bash
   python src/data_generator.py
   ```
3. Run EDA Analysis:
   ```bash
   python src/eda_analysis.py
   ```
