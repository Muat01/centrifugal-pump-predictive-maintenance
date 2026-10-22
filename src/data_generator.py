import pandas as pd
import numpy as np
import os

def generate_sensor_data(output_path):
    print("Generating simulated pump data...")
    np.random.seed(42)
    
    # Normal Data
    time_index_normal = pd.date_range(start='2026-10-22 08:00:00', periods=100, freq='1min')
    df_normal = pd.DataFrame({
        'timestamp': time_index_normal,
        'pressure_psi': np.random.normal(loc=120, scale=2, size=100),
        'flow_gpm': np.random.normal(loc=300, scale=5, size=100),
        'vibration_mms': np.random.normal(loc=2.0, scale=0.2, size=100),
        'status': 'Normal'
    })
    
    # Anomaly Data (Leakage)
    time_index_anomaly = pd.date_range(start='2026-10-22 09:40:00', periods=50, freq='1min')
    df_anomaly = pd.DataFrame({
        'timestamp': time_index_anomaly,
        'pressure_psi': np.random.normal(loc=90, scale=5, size=50),
        'flow_gpm': np.random.normal(loc=220, scale=10, size=50),
        'vibration_mms': np.random.normal(loc=3.5, scale=0.5, size=50),
        'status': 'Leakage'
    })
    
    # Combine and Save
    df_combined = pd.concat([df_normal, df_anomaly]).reset_index(drop=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_combined.to_csv(output_path, index=False)
    print(f"Data saved successfully to {output_path}")

if __name__ == "__main__":
    generate_sensor_data("../data/pump_sensor_data.csv")
