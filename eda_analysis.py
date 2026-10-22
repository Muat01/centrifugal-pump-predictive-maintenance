import pandas as pd
import matplotlib.pyplot as plt
import os

def perform_eda(data_path):
    print("Loading data for EDA...")
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    leakage_start_time = df[df['status'] == 'Leakage']['timestamp'].iloc[0]
    
    # Pressure Plot
    ax1.plot(df['timestamp'], df['pressure_psi'], color='blue', label='Pressure (psi)')
    ax1.axvline(x=leakage_start_time, color='red', linestyle='--', label='Leakage Starts')
    ax1.set_ylabel('Pressure (psi)')
    ax1.set_title('Centrifugal Pump Sensor Data Over Time')
    ax1.legend()
    ax1.grid(True)
    
    # Flow Plot
    ax2.plot(df['timestamp'], df['flow_gpm'], color='green', label='Flow (gpm)')
    ax2.axvline(x=leakage_start_time, color='red', linestyle='--')
    ax2.set_ylabel('Flow (gpm)')
    ax2.legend()
    ax2.grid(True)
    
    # Vibration Plot
    ax3.plot(df['timestamp'], df['vibration_mms'], color='orange', label='Vibration (mm/s)')
    ax3.axvline(x=leakage_start_time, color='red', linestyle='--')
    ax3.set_ylabel('Vibration (mm/s)')
    ax3.set_xlabel('Time')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    # Save the plot instead of just showing it
    output_image = "../data/eda_plot.png"
    plt.savefig(output_image)
    print(f"EDA Plot saved to {output_image}")
    plt.show()

if __name__ == "__main__":
    perform_eda("../data/pump_sensor_data.csv")
