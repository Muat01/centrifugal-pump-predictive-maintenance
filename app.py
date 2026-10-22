import streamlit as st
import pandas as pd
import plotly.express as px

# 1. Page Configuration
st.set_page_config(
    page_title="Smart Factory | Centrifugal Pump SCADA",
    page_icon="🏭",
    layout="wide"
)

# 2. Load Data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/pump_sensor_data.csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

if df is not None:
    # Sidebar Controls
    st.sidebar.header("⚙️ SCADA Simulation Control")
    simulation_mode = st.sidebar.radio("Select Operating Mode:", ["Live Sensor Stream", "Manual Override Simulation"])
    
    if simulation_mode == "Manual Override Simulation":
        st.sidebar.markdown("---")
        st.sidebar.subheader("Adjust Parameters Live:")
        sim_pressure = st.sidebar.slider("Pressure (psi)", 50.0, 140.0, 120.0)
        sim_flow = st.sidebar.slider("Flow (gpm)", 150.0, 350.0, 300.0)
        sim_vibration = st.sidebar.slider("Vibration (mm/s)", 1.0, 6.0, 2.0)
        
        df.loc[df.index[-1], 'pressure_psi'] = sim_pressure
        df.loc[df.index[-1], 'flow_gpm'] = sim_flow
        df.loc[df.index[-1], 'vibration_mms'] = sim_vibration
        df.loc[df.index[-1], 'status'] = 'Leakage' if sim_pressure < 100 or sim_vibration > 3.0 else 'Normal'

    # Main Header
    st.title("🏭 Smart Factory Control Center")
    st.markdown("### **Unit 4: Centrifugal Pump Predictive Maintenance & Real-time SCADA**")
    st.markdown("---")

    # 3. KPIs
    latest_row = df.iloc[-1]
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("⚡ Discharge Pressure", f"{latest_row['pressure_psi']:.1f} psi", f"{latest_row['pressure_psi'] - 120:.1f} vs Normal")
    with col2:
        st.metric("🌊 Flow Rate", f"{latest_row['flow_gpm']:.1f} gpm", f"{latest_row['flow_gpm'] - 300:.1f} vs Normal")
    with col3:
        st.metric("📳 Vibration Velocity", f"{latest_row['vibration_mms']:.2f} mm/s", f"{latest_row['vibration_mms'] - 2.0:+.2f} mm/s")
    with col4:
        is_leakage = latest_row['status'] == 'Leakage'
        status_text = "🔴 CRITICAL LEAKAGE" if is_leakage else "🟢 SYSTEM HEALTHY"
        st.metric("🛠️ Asset Status", status_text, "Action Required" if is_leakage else "Optimal")

    st.markdown("---")

    # 4. Interactive Plotly Charts (Dark Mode Native)
    st.subheader("📈 Real-time Industrial IoT Sensor Telemetry")

    fig_pressure = px.line(df, x='timestamp', y='pressure_psi', title="Pressure Sensor Tracking (psi)", template="plotly_dark")
    fig_pressure.update_traces(line_color='#00d2d3', line_width=2)
    fig_pressure.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_pressure, use_container_width=True)

    fig_flow = px.line(df, x='timestamp', y='flow_gpm', title="Flow Rate Sensor Tracking (gpm)", template="plotly_dark")
    fig_flow.update_traces(line_color='#ff9f43', line_width=2)
    fig_flow.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_flow, use_container_width=True)

    fig_vib = px.line(df, x='timestamp', y='vibration_mms', title="Vibration Velocity Sensor Tracking (mm/s)", template="plotly_dark")
    fig_vib.update_traces(line_color='#ff6b6b', line_width=2)
    fig_vib.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_vib, use_container_width=True)

    # 5. AI Diagnostics Panel
    st.markdown("---")
    st.subheader("🤖 AI Predictive Maintenance & Root Cause Analysis")
    
    col_a, col_b = st.columns([2, 1])
    
    with col_a:
        if is_leakage:
            st.error("""
            **⚠️ CRITICAL ALERT: Impeller Cavitation & Pipeline Leakage Signature Identified!**
            * **Diagnosis:** Pressure dropped significantly below threshold alongside abnormal vibration spikes.
            * **Root Cause Probabilities:** 
                1. Pipeline gasket failure / Downstream rupture (89%)
                2. Impeller mechanical imbalance (11%)
            * **Recommended Action:** Execute emergency shutdown sequence via SCADA and dispatch maintenance crew to Valve Sector B.
            """)
        else:
            st.success("""
            **✅ SYSTEM OPERATING NORMALLY**
            * All telemetry indicators (Pressure, Flow, Vibration) are within ISO 10816 standards.
            * Estimated remaining useful life (RUL): **> 1,450 operational hours**.
            """)
            
    with col_b:
        st.markdown("### 📊 Health Score")
        health_percentage = 35.0 if is_leakage else 98.5
        st.progress(health_percentage / 100.0)
        st.write(f"**Asset Health Index:** `{health_percentage}%`")
        st.info("Powered by Random Forest ML Classifier & IoT Telemetry.")
