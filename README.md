# Centrifugal Pump Predictive Maintenance & SCADA Dashboard ⚙️💧

A Smart Manufacturing / Industry 4.0 portfolio project that simulates centrifugal pump telemetry and presents sensor trends, asset status, and AI-assisted diagnostic guidance in an interactive Streamlit dashboard.

> **Project status:** Portfolio demonstration using CSV-based historical telemetry and simulated manual overrides. It is not connected to a live PLC/SCADA system and is not a safety-rated monitoring or control application.

## 🚀 Live Dashboard

**[Open the Centrifugal Pump Predictive Maintenance Dashboard](https://centrifugal-pump-predictive-maintenance.streamlit.app/)**

Explore the interactive SCADA dashboard featuring real-time-style sensor monitoring, pressure/flow/vibration trends, Random Forest classification, and AI-assisted diagnostic recommendations.

[![Open Live Dashboard](https://img.shields.io/badge/Streamlit-Open%20Live%20Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://centrifugal-pump-predictive-maintenance.streamlit.app/)

## Project Overview

Centrifugal pumps are widely used in industrial processes, and changes in pressure, flow, or vibration can indicate abnormal operating conditions. This project demonstrates a basic condition-monitoring workflow: load historical sensor data, visualize time-series signals, classify the current sensor profile, and display engineering checks that can guide further investigation.

The dashboard is built with Python, Streamlit, Plotly, Pandas, NumPy, and Scikit-learn.

## Key Features

- **SCADA-inspired dark interface** with a responsive dashboard layout.
- **Four KPI cards:** discharge pressure, flow rate, vibration velocity, and asset status.
- **Simulation controls:** historical CSV replay and manual sensor-value overrides.
- **Three interactive trend charts:** pressure, flow, and vibration, with hover tooltips, reference values, and illustrative operating bands.
- **Random Forest classification:** trains on historical rows labelled `Normal` and `Leakage` when sufficient examples are available.
- **AI-assisted diagnostic panel:** displays sensor observations, actionable engineering checks, and a heuristic Asset Health Index.
- **Data validation:** parses timestamps, converts sensor columns to numeric values, handles invalid rows, and checks required columns.
- **Fallback diagnostics:** uses configurable threshold rules if a Random Forest model cannot be trained.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application logic and data processing |
| Streamlit | Interactive web dashboard |
| Plotly | Interactive time-series charts |
| Pandas | CSV ingestion and data cleaning |
| NumPy | Numeric operations and health-score calculations |
| Scikit-learn | Random Forest classification and model evaluation |

## Repository Structure

```text
centrifugal-pump-predictive-maintenance/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── pump_sensor_data.csv
├── src/
│   ├── data_generator.py
│   ├── eda_analysis.py
│   └── train_model.py
└── docs/
    └── dashboard-screenshot.png
```

The tree above describes the intended project layout. Keep only files and folders that actually exist in your repository. The dashboard expects the dataset at `data/pump_sensor_data.csv`.

## Dataset

The application expects a CSV file at:

```text
data/pump_sensor_data.csv
```

Required columns:

| Column | Description | Example / unit |
|---|---|---|
| `timestamp` | Telemetry timestamp | ISO timestamp |
| `pressure_psi` | Discharge pressure | `120` psi |
| `flow_gpm` | Flow rate | `300` GPM |
| `vibration_mms` | Vibration velocity | `2.0` mm/s |
| `status` | Historical condition label | `Normal` or `Leakage` |

Example:

```csv
timestamp,pressure_psi,flow_gpm,vibration_mms,status
2026-01-01T08:00:00Z,120,300,2.0,Normal
2026-01-01T08:01:00Z,119,298,2.1,Normal
2026-01-01T08:02:00Z,88,180,6.2,Leakage
```

The example rows are illustrative. Use your actual dataset for the dashboard. The CSV needs enough valid rows from both classes to train the Random Forest; otherwise, the app uses threshold-based fallback diagnostics.

## Getting Started

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd centrifugal-pump-predictive-maintenance
```

Replace `<YOUR_GITHUB_REPOSITORY_URL>` with your repository's clone URL.

### 2. Create and activate a virtual environment (recommended)

**Windows PowerShell**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

Ensure `requirements.txt` includes:

```text
streamlit
pandas
numpy
plotly
scikit-learn
```

### 4. Check the dataset

Confirm that `data/pump_sensor_data.csv` exists and contains all five required columns listed above.

If your repository includes a data generator and you want to regenerate the CSV, run:

```bash
python src/data_generator.py
```

Only run this command if that script exists and writes to the expected dataset path; it may replace your current generated data.

### 5. Launch the dashboard

```bash
streamlit run app.py
```

Open the local URL printed in the terminal, typically `http://localhost:8501`.

## How to Use the Dashboard

1. Select **Live Stream** to inspect the latest historical telemetry represented in the CSV.
2. Adjust the **trend window** to display more or fewer recent samples.
3. Select **Manual Override** to explore simulated pressure, flow, and vibration values.
4. Review the KPI cards and all three trend charts for values outside the configured illustrative bands.
5. Read the diagnostic findings and recommended engineering checks.
6. Open **Telemetry details & data quality** to inspect recent rows and the source data summary.

Manual Override changes simulated readings only. It does not issue commands to industrial equipment.

## Model and Diagnostic Approach

The application uses a Scikit-learn `RandomForestClassifier` to classify sensor profiles using the historical `status` labels.

- The model uses `pressure_psi`, `flow_gpm`, and `vibration_mms` as features.
- A chronological 80/20 split is used for a holdout accuracy estimate when the dataset and both partitions contain the required classes.
- The final demo classifier is then fitted on the available valid historical rows.
- If there are too few valid rows or the data does not contain both classes, a threshold-based fallback is used.
- The diagnostic panel combines the selected classification with sensor-band checks to produce observations and recommended inspections.

### Important interpretation notes

- A classification of `Leakage` is a model estimate based on historical labels, not proof that a physical leak exists.
- The Asset Health Index is a **heuristic demo score**, not a certified reliability metric or probability of failure.
- Holdout accuracy alone does not establish model quality. For a stronger evaluation, report a confusion matrix, precision, recall, F1-score, class balance, and performance on genuinely later operating periods.
- If data contains adjacent samples from the same operating episode, chronological splitting may still overstate generalization. Evaluate by time period, pump, or operating episode where possible.

## Illustrative Operating Bands

The current application uses these default bands as examples:

| Sensor | Illustrative band | Reference value |
|---|---:|---:|
| Discharge pressure | 100–140 psi | 120 psi |
| Flow rate | 240–360 GPM | 300 GPM |
| Vibration velocity | 0–4.5 mm/s | 2.0 mm/s |

**These are not universal pump limits.** Replace them with values justified by the pump OEM documentation, pump curve, installation, process conditions, instrumentation, and site alarm philosophy before using the dashboard for engineering decisions.

## Deployment on Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, and the required dataset to your GitHub repository.
2. In Streamlit Community Cloud, create or open the app associated with the repository.
3. Select the correct branch and set the main file path to `app.py`.
4. Deploy or reboot the app after committing updates.
5. Review the app logs if dependency, path, or data-loading errors occur.

**Data note:** Do not commit confidential plant telemetry, sensitive operational data, credentials, or customer information to a public repository. Use synthetic or appropriately anonymized data for a public portfolio.

## Current Limitations and Future Work

### Current limitations

- CSV-based telemetry replay rather than live industrial data ingestion.
- Manual Override is a simulation control only.
- Thresholds are illustrative and require site-specific validation.
- Classification depends on the quality and representativeness of historical labels.
- No direct PLC, OPC UA, or MQTT connection is included in the current dashboard.
- The model does not forecast the time until failure.

### Potential next steps

- [ ] Add MQTT or OPC UA telemetry ingestion with robust connection handling.
- [ ] Add rolling-window features and time-aware model validation.
- [ ] Report confusion matrix, precision, recall, F1-score, and class-specific performance.
- [ ] Add alert history and event logging.
- [ ] Add data freshness, missing-data, and sensor-quality indicators.
- [ ] Add model versioning and reproducible training scripts.
- [ ] Validate operating thresholds with pump OEM guidance and site engineers.
- [ ] Add automated tests for data validation, diagnostics, and model inference.

## Safety and Intended Use

This project is for learning, demonstration, and portfolio purposes. It is not a replacement for a validated industrial control system, an OEM maintenance procedure, or a qualified engineering assessment. Do not use its output as the sole basis for operating, shutting down, or maintaining equipment.

## Author

**Muat01 — Mutaasem Hadi**

- GitHub: [Muat01](https://github.com/Muat01)

---

If you find this project useful, feel free to explore the code, suggest improvements, or adapt it for your own condition-monitoring experiments.
