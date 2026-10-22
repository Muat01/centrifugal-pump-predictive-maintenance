"""
Centrifugal Pump Predictive Maintenance & SCADA Dashboard
--------------------------------------------------------
Run:
    pip install streamlit pandas numpy plotly scikit-learn
    streamlit run app.py

Expected CSV:
    data/pump_sensor_data.csv

Required columns:
    timestamp, pressure_psi, flow_gpm, vibration_mms, status

Notes:
- This is a portfolio/demo dashboard, not a safety-rated control system.
- Manual override changes simulated values only; it does not write to a PLC.
- Random Forest results are classification estimates based on historical labels,
  not guaranteed predictions of future failure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "pump_sensor_data.csv"

SENSOR_COLUMNS = ["pressure_psi", "flow_gpm", "vibration_mms"]
REQUIRED_COLUMNS = ["timestamp", *SENSOR_COLUMNS, "status"]

# Illustrative operating bands for this portfolio demo. Confirm against the
# pump curve, OEM limits, process conditions, and site alarm philosophy.
NORMAL_BANDS = {
    "pressure_psi": (100.0, 140.0),
    "flow_gpm": (240.0, 360.0),
    "vibration_mms": (0.0, 4.5),
}
REFERENCE_VALUES = {
    "pressure_psi": 120.0,
    "flow_gpm": 300.0,
    "vibration_mms": 2.0,
}
FEATURE_NAMES = SENSOR_COLUMNS


# ---------------------------------------------------------------------------
# Streamlit page and theme
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PumpSense | Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root { color-scheme: dark; }
      .stApp {
        background: radial-gradient(ellipse at top left, #142337 0%, #0b1220 48%, #080d16 100%);
        color: #e5edf7;
      }
      [data-testid="stSidebar"] {
        background: #0b1321;
        border-right: 1px solid #26354a;
      }
      [data-testid="stHeader"] { background: rgba(8, 13, 22, 0.88); }
      .block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1600px; }
      .hero {
        padding: 1.15rem 1.4rem;
        border: 1px solid #263a51;
        border-radius: 16px;
        background: linear-gradient(115deg, rgba(18,37,58,.96), rgba(13,23,39,.96));
        margin-bottom: 1.2rem;
      }
      .eyebrow {
        color: #5eead4; font-size: .72rem; font-weight: 800;
        letter-spacing: .16em; text-transform: uppercase;
      }
      .hero h1 { margin: .25rem 0 .25rem 0; font-size: 2rem; color: #f1f5f9; }
      .hero p { margin: 0; color: #9fb0c5; }
      .section-title {
        color: #e8eef7; font-weight: 750; font-size: 1.05rem;
        padding: .2rem 0 .65rem 0;
      }
      .kpi {
        border: 1px solid #283a50; border-radius: 14px;
        background: linear-gradient(145deg, #142235, #0e1827);
        padding: 1rem 1.05rem; min-height: 126px;
      }
      .kpi-label { color: #9fb0c5; font-size: .78rem; font-weight: 700; letter-spacing: .04em; }
      .kpi-value { color: #f8fafc; font-size: 1.8rem; font-weight: 800; margin-top: .35rem; line-height: 1.25; }
      .kpi-foot { color: #91a5bb; font-size: .75rem; margin-top: .35rem; }
      .panel {
        border: 1px solid #283a50; border-radius: 14px;
        background: rgba(15, 26, 42, .94); padding: 1rem 1.15rem;
      }
      .status-pill {
        display: inline-block; padding: .28rem .65rem; border-radius: 999px;
        font-weight: 800; font-size: .76rem; letter-spacing: .03em;
      }
      .status-good { color: #6ee7b7; background: rgba(16,185,129,.12); border: 1px solid rgba(16,185,129,.35); }
      .status-warn { color: #fbbf24; background: rgba(245,158,11,.12); border: 1px solid rgba(245,158,11,.35); }
      .status-bad { color: #fca5a5; background: rgba(239,68,68,.12); border: 1px solid rgba(239,68,68,.35); }
      .small-muted { color: #93a6bc; font-size: .82rem; }
      div[data-testid="stMetric"] {
        background: #111d2d; border: 1px solid #26384d;
        padding: .8rem; border-radius: 12px;
      }
      div[data-testid="stMetricLabel"] { color: #a9b8ca; }
      div[data-testid="stMetricValue"] { color: #f8fafc; }
      .stButton button {
        border-radius: 9px; border: 1px solid #2b536a;
        background: #123149; color: #e5f6ff; font-weight: 700;
      }
      hr { border-color: #26364b; }
      footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data and model helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_dataset(path_string: str, modified_time: float) -> tuple[pd.DataFrame, str | None]:
    """Load and validate the source CSV. modified_time invalidates cache on file changes."""
    del modified_time
    path = Path(path_string)
    if not path.exists():
        return pd.DataFrame(columns=REQUIRED_COLUMNS), f"Dataset not found: {path}"
    try:
        raw = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        return pd.DataFrame(columns=REQUIRED_COLUMNS), f"Could not read CSV: {exc}"

    missing = [col for col in REQUIRED_COLUMNS if col not in raw.columns]
    if missing:
        return pd.DataFrame(columns=REQUIRED_COLUMNS), (
            "CSV is missing required column(s): " + ", ".join(missing)
        )

    df = raw[REQUIRED_COLUMNS].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    for col in SENSOR_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    df["status"] = df["status"].astype("string").str.strip().str.lower()
    df["status"] = df["status"].map(
        lambda x: "Leakage" if isinstance(x, str) and x in {"leakage", "leak", "fault", "failure"}
        else ("Normal" if isinstance(x, str) and x in {"normal", "healthy", "ok"} else pd.NA)
    )
    df = df.dropna(subset=["timestamp", *SENSOR_COLUMNS, "status"])
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    if df.empty:
        return df, "The CSV contains no usable rows after validation."
    return df.reset_index(drop=True), None


@st.cache_resource(show_spinner=False)
def train_model(data_fingerprint: tuple[Any, ...], model_frame: pd.DataFrame) -> tuple[Any | None, float | None, str]:
    """Train a small RF classifier only when there are enough examples per class."""
    del data_fingerprint
    frame = model_frame.dropna(subset=[*FEATURE_NAMES, "status"]).copy()
    if frame["status"].nunique() < 2:
        return None, None, "Model unavailable: the dataset must contain both Normal and Leakage labels."
    counts = frame["status"].value_counts()
    if len(frame) < 20 or counts.min() < 5:
        return None, None, "Model unavailable: need at least 20 valid rows and 5 examples per class."

    X = frame[FEATURE_NAMES]
    y = frame["status"]
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", RandomForestClassifier(
                    n_estimators=250,
                    max_depth=10,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                )),
            ]
        )
        model.fit(X_train, y_train)
        accuracy = float(accuracy_score(y_test, model.predict(X_test)))
        # Fit the production demo model on all valid historical data after
        # computing a holdout estimate.
        model.fit(X, y)
        return model, accuracy, "Random Forest trained on historical labeled telemetry."
    except (ValueError, TypeError) as exc:
        return None, None, f"Model unavailable: {exc}"


def fallback_status(values: dict[str, float]) -> str:
    """Conservative threshold fallback for demo use when ML cannot be trained."""
    pressure = values["pressure_psi"]
    flow = values["flow_gpm"]
    vibration = values["vibration_mms"]
    if (
        pressure < NORMAL_BANDS["pressure_psi"][0]
        or pressure > NORMAL_BANDS["pressure_psi"][1]
        or flow < NORMAL_BANDS["flow_gpm"][0]
        or flow > NORMAL_BANDS["flow_gpm"][1]
        or vibration > NORMAL_BANDS["vibration_mms"][1]
        or vibration < NORMAL_BANDS["vibration_mms"][0]
    ):
        return "Leakage"
    return "Normal"


def diagnose(values: dict[str, float], predicted_status: str) -> tuple[int, str, list[str], list[str]]:
    """Return illustrative health score, severity, observations and actions."""
    pressure = values["pressure_psi"]
    flow = values["flow_gpm"]
    vibration = values["vibration_mms"]
    observations: list[str] = []
    actions: list[str] = []
    penalty = 0

    p_lo, p_hi = NORMAL_BANDS["pressure_psi"]
    f_lo, f_hi = NORMAL_BANDS["flow_gpm"]
    v_lo, v_hi = NORMAL_BANDS["vibration_mms"]

    if pressure < p_lo:
        observations.append("Discharge pressure is below the illustrative operating band.")
        actions.append("Check suction supply, priming, suction strainer restriction, and discharge pressure transmitter.")
        penalty += min(25, int((p_lo - pressure) * 0.8) + 8)
    elif pressure > p_hi:
        observations.append("Discharge pressure is above the illustrative operating band.")
        actions.append("Verify valve position, downstream restriction, pressure transmitter calibration, and pump operating point.")
        penalty += min(20, int((pressure - p_hi) * 0.5) + 6)

    if flow < f_lo:
        observations.append("Flow is below the illustrative operating band.")
        actions.append("Inspect for suction blockage, closed valves, impeller wear, or a flow-meter issue; compare against the pump curve.")
        penalty += min(25, int((f_lo - flow) * 0.12) + 7)
    elif flow > f_hi:
        observations.append("Flow is above the illustrative operating band.")
        actions.append("Confirm process demand, valve lineup, and flow-meter scaling; verify the pump is not operating outside its intended range.")
        penalty += min(18, int((flow - f_hi) * 0.08) + 5)

    if vibration > v_hi:
        observations.append("Vibration exceeds the illustrative high-vibration threshold.")
        actions.append("Inspect coupling alignment, bearing condition, looseness, cavitation, and foundation bolts; trend vibration by frequency if available.")
        penalty += min(35, int((vibration - v_hi) * 7) + 12)
    elif vibration > 3.0:
        observations.append("Vibration is elevated relative to the 2.0 mm/s reference.")
        actions.append("Trend vibration and inspect alignment, lubrication, mounting, and potential cavitation at the next safe opportunity.")
        penalty += 8

    if predicted_status == "Leakage":
        observations.append("The asset classifier indicates a Leakage condition.")
        actions.append("Inspect seals, flanges, drains, and connected piping for leakage; follow site isolation and permit procedures before physical inspection.")
        penalty += 18

    if not observations:
        observations.append("All displayed sensor values are within the configured illustrative operating bands.")
        actions.append("Continue routine monitoring and compare trends with the pump OEM limits and site baseline.")

    health = int(np.clip(100 - penalty, 0, 100))
    if health >= 80 and predicted_status == "Normal":
        severity = "HEALTHY"
    elif health >= 55:
        severity = "ATTENTION"
    else:
        severity = "CRITICAL"
    # Deduplicate repeated actions while preserving order.
    actions = list(dict.fromkeys(actions))
    return health, severity, observations, actions


def make_trend_chart(
    frame: pd.DataFrame,
    column: str,
    title: str,
    unit: str,
    color: str,
    latest_value: float,
    reference_value: float,
    threshold_band: tuple[float, float] | None = None,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame["timestamp"],
            y=frame[column],
            mode="lines",
            name=title,
            line=dict(color=color, width=2.3),
            hovertemplate=f"%{{x|%Y-%m-%d %H:%M:%S}}<br>{title}: %{{y:.2f}} {unit}<extra></extra>",
        )
    )
    fig.add_hline(
        y=reference_value,
        line_dash="dot",
        line_color="#91a4ba",
        opacity=0.8,
        annotation_text=f"Reference {reference_value:g}",
        annotation_position="bottom right",
        annotation_font_color="#aab8c9",
    )
    if threshold_band is not None:
        lower, upper = threshold_band
        fig.add_hrect(
            y0=lower, y1=upper, fillcolor="rgba(52,211,153,0.07)",
            line_width=0, layer="below",
        )
    fig.update_layout(
        template="plotly_dark",
        title=dict(text=title, x=0.02, font=dict(size=15, color="#e7eef7")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,14,24,0.45)",
        margin=dict(l=12, r=18, t=52, b=12),
        height=290,
        showlegend=False,
        hovermode="x unified",
        font=dict(family="Inter, Arial, sans-serif", color="#b9c7d8", size=11),
        xaxis=dict(title=None, showgrid=True, gridcolor="rgba(120,145,170,0.12)", zeroline=False),
        yaxis=dict(title=unit, showgrid=True, gridcolor="rgba(120,145,170,0.12)", zeroline=False),
    )
    # Mark current operating value as a separate point.
    if len(frame):
        fig.add_trace(
            go.Scatter(
                x=[frame["timestamp"].iloc[-1]],
                y=[latest_value],
                mode="markers",
                name="Current",
                marker=dict(color=color, size=8, line=dict(color="#f8fafc", width=1)),
                hovertemplate=f"Current: {latest_value:.2f} {unit}<extra></extra>",
            )
        )
    return fig


def status_class(severity: str) -> str:
    if severity == "HEALTHY":
        return "status-good"
    if severity == "ATTENTION":
        return "status-warn"
    return "status-bad"


# ---------------------------------------------------------------------------
# Sidebar: simulation controls
# ---------------------------------------------------------------------------
st.sidebar.markdown("## ⚙️ PumpSense")
st.sidebar.caption("INDUSTRIAL ASSET MONITOR")
st.sidebar.divider()
mode = st.sidebar.radio(
    "Operating mode",
    ["Live Stream", "Manual Override"],
    index=0,
    help="Live Stream replays the latest historical telemetry. Manual Override changes simulated sensor readings only.",
)

st.sidebar.markdown("### Simulation controls")
live_window = st.sidebar.select_slider(
    "Trend window",
    options=[30, 60, 120, 250, 500],
    value=120,
    format_func=lambda x: f"Last {x} samples",
)
st.sidebar.caption("Illustrative normal reference: 120 psi · 300 GPM · 2.0 mm/s")

if mode == "Manual Override":
    pressure_input = st.sidebar.slider("Discharge pressure (psi)", 0.0, 220.0, 120.0, 1.0)
    flow_input = st.sidebar.slider("Flow rate (GPM)", 0.0, 500.0, 300.0, 5.0)
    vibration_input = st.sidebar.slider("Vibration velocity (mm/s)", 0.0, 20.0, 2.0, 0.1)
else:
    pressure_input = flow_input = vibration_input = None

st.sidebar.divider()
st.sidebar.markdown("### System information")
st.sidebar.markdown(
    '<span class="small-muted">Model: Random Forest</span><br>'
    '<span class="small-muted">Telemetry: CSV replay / simulation</span><br>'
    '<span class="small-muted">Protocol: Not connected to live PLC</span>',
    unsafe_allow_html=True,
)
if st.sidebar.button("Refresh dashboard", use_container_width=True):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()


# ---------------------------------------------------------------------------
# Load data and train classifier
# ---------------------------------------------------------------------------
try:
    file_mtime = DATA_PATH.stat().st_mtime if DATA_PATH.exists() else 0.0
except OSError:
    file_mtime = 0.0

data, data_error = load_dataset(str(DATA_PATH), file_mtime)

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Industry 4.0 · Condition Monitoring</div>
      <h1>⚙️ Centrifugal Pump Intelligence</h1>
      <p>Predictive maintenance · Sensor telemetry · AI-assisted diagnostics</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if data_error:
    st.error(data_error)
    st.info(
        "Add `data/pump_sensor_data.csv` with columns `timestamp, pressure_psi, "
        "flow_gpm, vibration_mms, status`. The dashboard will load automatically when the file is available."
    )
    st.code(
        "timestamp,pressure_psi,flow_gpm,vibration_mms,status\n"
        "2026-01-01T08:00:00Z,120,300,2.0,Normal\n"
        "2026-01-01T08:01:00Z,119,298,2.1,Normal\n"
        "2026-01-01T08:02:00Z,88,180,6.2,Leakage",
        language="csv",
    )
    st.stop()

if data.empty:
    st.warning("No usable telemetry rows were found.")
    st.stop()

fingerprint = (
    len(data),
    str(data["timestamp"].min()),
    str(data["timestamp"].max()),
    int(pd.util.hash_pandas_object(data[REQUIRED_COLUMNS], index=True).sum()),
)
model, validation_accuracy, model_message = train_model(fingerprint, data)

# ---------------------------------------------------------------------------
# Select current operating point
# ---------------------------------------------------------------------------
trend_data = data.tail(int(live_window)).copy()
latest_row = data.iloc[-1]
current_values = {
    "pressure_psi": float(latest_row["pressure_psi"]),
    "flow_gpm": float(latest_row["flow_gpm"]),
    "vibration_mms": float(latest_row["vibration_mms"]),
}

if mode == "Manual Override":
    current_values = {
        "pressure_psi": float(pressure_input),
        "flow_gpm": float(flow_input),
        "vibration_mms": float(vibration_input),
    }
    current_time = pd.Timestamp.now(tz="UTC")
    # Append a synthetic current point to the trend view, clearly presented as
    # simulated telemetry and without mutating the source dataset.
    synthetic_row = {
        "timestamp": current_time,
        **current_values,
        "status": fallback_status(current_values),
    }
    trend_data = pd.concat([trend_data, pd.DataFrame([synthetic_row])], ignore_index=True)

if model is not None:
    predicted_status = str(model.predict(pd.DataFrame([current_values], columns=FEATURE_NAMES))[0])
    model_probability = None
    try:
        probabilities = model.predict_proba(pd.DataFrame([current_values], columns=FEATURE_NAMES))[0]
        class_labels = list(model.named_steps["classifier"].classes_)
        model_probability = float(probabilities[class_labels.index(predicted_status)])
    except (ValueError, IndexError):
        model_probability = None
    diagnosis_source = "Random Forest classifier"
else:
    predicted_status = fallback_status(current_values)
    model_probability = None
    diagnosis_source = "Threshold-based fallback"

health_index, severity, observations, recommendations = diagnose(current_values, predicted_status)

# The observed latest status is shown in live mode; the model/fallback output is
# the operational diagnostic for the current point.
if mode == "Live Stream":
    displayed_status = predicted_status
else:
    displayed_status = predicted_status

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">Live Asset Overview</div>', unsafe_allow_html=True)
k1, k2, k3, k4 = st.columns(4, gap="medium")

pressure_outside = not (NORMAL_BANDS["pressure_psi"][0] <= current_values["pressure_psi"] <= NORMAL_BANDS["pressure_psi"][1])
flow_outside = not (NORMAL_BANDS["flow_gpm"][0] <= current_values["flow_gpm"] <= NORMAL_BANDS["flow_gpm"][1])
vibration_outside = not (NORMAL_BANDS["vibration_mms"][0] <= current_values["vibration_mms"] <= NORMAL_BANDS["vibration_mms"][1])

with k1:
    st.markdown(
        f'<div class="kpi"><div class="kpi-label">DISCHARGE PRESSURE</div>'
        f'<div class="kpi-value">{current_values["pressure_psi"]:.1f} <span style="font-size:.9rem;color:#9fb0c5">psi</span></div>'
        f'<div class="kpi-foot">Reference 120 psi · {"Outside band" if pressure_outside else "Within band"}</div></div>',
        unsafe_allow_html=True,
    )
with k2:
    st.markdown(
        f'<div class="kpi"><div class="kpi-label">FLOW RATE</div>'
        f'<div class="kpi-value">{current_values["flow_gpm"]:.1f} <span style="font-size:.9rem;color:#9fb0c5">GPM</span></div>'
        f'<div class="kpi-foot">Reference 300 GPM · {"Outside band" if flow_outside else "Within band"}</div></div>',
        unsafe_allow_html=True,
    )
with k3:
    st.markdown(
        f'<div class="kpi"><div class="kpi-label">VIBRATION VELOCITY</div>'
        f'<div class="kpi-value">{current_values["vibration_mms"]:.2f} <span style="font-size:.9rem;color:#9fb0c5">mm/s</span></div>'
        f'<div class="kpi-foot">Reference 2.0 mm/s · {"Outside band" if vibration_outside else "Within band"}</div></div>',
        unsafe_allow_html=True,
    )
with k4:
    status_style = "status-good" if displayed_status == "Normal" else "status-bad"
    st.markdown(
        f'<div class="kpi"><div class="kpi-label">ASSET STATUS</div>'
        f'<div style="margin-top:.65rem"><span class="status-pill {status_style}">{displayed_status.upper()}</span></div>'
        f'<div class="kpi-foot">Diagnostic source: {diagnosis_source}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("")

# ---------------------------------------------------------------------------
# Trend charts
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">Sensor Trends</div>', unsafe_allow_html=True)
chart1, chart2, chart3 = st.columns(3, gap="medium")

with chart1:
    st.plotly_chart(
        make_trend_chart(
            trend_data, "pressure_psi", "Discharge Pressure", "psi", "#38bdf8",
            current_values["pressure_psi"], REFERENCE_VALUES["pressure_psi"],
            NORMAL_BANDS["pressure_psi"],
        ),
        use_container_width=True, config={"displaylogo": False, "responsive": True},
    )
with chart2:
    st.plotly_chart(
        make_trend_chart(
            trend_data, "flow_gpm", "Flow Rate", "GPM", "#a78bfa",
            current_values["flow_gpm"], REFERENCE_VALUES["flow_gpm"],
            NORMAL_BANDS["flow_gpm"],
        ),
        use_container_width=True, config={"displaylogo": False, "responsive": True},
    )
with chart3:
    st.plotly_chart(
        make_trend_chart(
            trend_data, "vibration_mms", "Vibration Velocity", "mm/s", "#fbbf24",
            current_values["vibration_mms"], REFERENCE_VALUES["vibration_mms"],
            NORMAL_BANDS["vibration_mms"],
        ),
        use_container_width=True, config={"displaylogo": False, "responsive": True},
    )

# ---------------------------------------------------------------------------
# AI diagnostics and health
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">AI Diagnostics & Root Cause Analysis</div>', unsafe_allow_html=True)
left, right = st.columns([1.05, 1.35], gap="medium")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("#### Asset Health Index")
    st.progress(health_index / 100.0, text=f"{health_index} / 100")
    st.markdown(
        f'<span class="status-pill {status_class(severity)}">{severity}</span>',
        unsafe_allow_html=True,
    )
    st.markdown("")
    st.markdown(f"**Diagnostic classification:** {displayed_status}")
    if model_probability is not None:
        st.markdown(f"**Model confidence for selected class:** {model_probability:.1%}")
        st.caption("Confidence is the model's class probability, not a calibrated failure probability.")
    if validation_accuracy is not None:
        st.markdown(f"**Holdout accuracy:** {validation_accuracy:.1%}")
    else:
        st.markdown("**Holdout accuracy:** Not available")
    st.caption(model_message)
    st.caption("Health score is a heuristic indicator for this demo, not a certified safety or reliability metric.")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("#### Findings")
    for item in observations:
        st.markdown(f"- {item}")
    st.markdown("#### Recommended engineering actions")
    for index, action in enumerate(recommendations, start=1):
        st.markdown(f"**{index}.** {action}")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data quality and context
# ---------------------------------------------------------------------------
with st.expander("Telemetry details & data quality"):
    st.write(f"**Source:** `{DATA_PATH.relative_to(APP_DIR)}`")
    st.write(f"**Validated rows:** {len(data):,}")
    st.write(f"**Time range (UTC):** {data['timestamp'].min()} → {data['timestamp'].max()}")
    st.write(f"**Source labels:** {data['status'].value_counts().to_dict()}")
    st.write(f"**Mode:** {mode}")
    st.write("**Thresholds are illustrative:** pressure 100–140 psi, flow 240–360 GPM, vibration 0–4.5 mm/s.")
    st.dataframe(trend_data.tail(20), use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown(
    '<div class="small-muted">PumpSense SCADA Demo · Historical CSV replay and simulated overrides only. '
    'Validate alarm thresholds with OEM documentation and site engineering standards before operational use.</div>',
    unsafe_allow_html=True,
)
