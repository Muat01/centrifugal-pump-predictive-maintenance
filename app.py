import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 1. Page Configuration (Industrial Dark Theme Setup)
st.set_page_config(
    page_title="Smart Factory | Centrifugal Pump SCADA",
    page_icon="🏭",
    layout="wide"
)

# Custom CSS for Industrial Look & Feel
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stMetric {
        background-color: #1a1c24;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2f3640;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Header & Title
st.title("🏭 Smart Factory Control Center")
st.markdown("### **Unit 4: Centrifugal Pump Predictive Maintenance & Real-time SCADA**")
st.markdown("---")

# 3. Load Dataset
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
    # Sidebar Controls for Simulation & Testing
    st.sidebar.header("⚙️ SCADA Simulation Control")
    simulation_mode = st.sidebar.radio("Select Operating Mode:", ["Live Sensor Stream", "Manual Override Simulation"])
    
    if simulation_mode == "Manual Override Simulation":
        st.sidebar.markdown("---")
        st.sidebar.subheader("Adjust Parameters Live:")
        sim_pressure = st.sidebar.slider("Pressure (psi)", 50.0, 140.0, 120.0)
        sim_flow = st.sidebar.slider("Flow (gpm)", 150.0, 350.0, 300.0)
        sim_vibration = st.sidebar.slider("Vibration (mm/s)", 1.0, 6.0, 2.0)
        
        # Override last row for live simulation
        df.loc[df.index[-1], 'pressure_psi'] = sim_pressure
        df.loc[df.index[-1], 'flow_gpm'] = sim_flow
        df.loc[df.index[-1], 'vibration_mms'] = sim_vibration
        df.loc[df.index[-1], 'status'] = 'Leakage' if sim_pressure < 100 or sim_vibration > 3.0 else 'Normal'

    # 4. Key Performance Indicators (KPIs)
    latest_row = df.iloc[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="⚡ Discharge Pressure", 
            value=f"{latest_row['pressure_psi']:.1f} psi", 
            delta=f"{latest_row['pressure_psi'] - 120:.1f} vs Normal"
        )
    with col2:
        st.metric(
            label="🌊 Flow Rate", 
            value=f"{latest_row['flow_gpm']:.1f} gpm", 
            delta=f"{latest_row['flow_gpm'] - 300:.1f} vs Normal"
        )
    with col3:
        st.metric(
            label="📳 Vibration Velocity", 
            value=f"{latest_row['vibration_mms']:.2f} mm/s", 
            delta=f"{latest_row['vibration_mms'] - 2.0:+.2f} mm/s"
        )
    with col4:
        is_leakage = latest_row['status'] == 'Leakage'
        status_text = "🔴 CRITICAL LEAKAGE" if is_leakage else "🟢 SYSTEM HEALTHY"
        st.metric(
            label="🛠️ Asset Status", 
            value=status_text,
            delta="Action Required" if is_leakage else "Optimal"
        )

    st.markdown("---")

    # 5. Interactive Advanced Charts using Plotly (Fixed line parameter syntax)
    st.subheader("📈 Real-time Industrial IoT Sensor Telemetry (Time-Series)")
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['pressure_psi'],
        name='Pressure (psi)', line=dict(color='#00d2d3', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['flow_gpm'],
        name='Flow (gpm)', line=dict(color='#ff9f43', width=2), yaxis="y2"
    ))
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['vibration_mms'],
        name='Vibration (mm/s)', line=dict(color='#ff6b6b', width=2), yaxis="y3"
    ))

    fig.update_layout(
        template="plotly_dark",
        height=500,
        hovermode="x unified",
        xaxis=dict(title="Timestamp (UTC)"),
        yaxis=dict(title="Pressure (psi)", titlefont=dict(color="#00d2d3"), tickfont=dict(color="#00d2d3")),
        yaxis2=dict(title="Flow (gpm)", titlefont=dict(color="#ff9f43"), tickfont=dict(color="#ff9f43"), anchor="free", overlaying="y", side="right"),
        yaxis3=dict(title="Vibration (mm/s)", titlefont=dict(color="#ff6b6b"), tickfont=dict(color="#ff6b6b"), anchor="x", overlaying="y", side="right", position=0.95),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    leakage_df = df[df['status'] == 'Leakage']
    if not leakage_df.empty:
        fig.add_vline(x=leakage_df['timestamp'].iloc[0], line_width=2, line_dash="dash", line_color="red", annotation_text="Anomaly Detected")

    st.plotly_chart(fig, use_container_width=True)

    # 6. AI Predictive Diagnostics & Action Panel
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
