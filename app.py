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
    with st.container(border=True):
        st.markdown("#### Asset Health Index")
        st.progress(health_index / 100.0, text=f"{health_index} / 100")
        st.markdown(
            f'<span class="status-pill {status_class(severity)}">{severity}</span>',
            unsafe_allow_html=True,
        )
        st.markdown("")
        st.markdown(f"**Predicted classification:** {displayed_status}")
        if mode == "Live Stream":
            st.markdown(f"**Latest CSV label:** {observed_status}")
        if model_probability is not None:
            st.markdown(f"**Model confidence for selected class:** {model_probability:.1%}")
            st.caption("Confidence is the model's class probability, not a calibrated failure probability.")
        if validation_accuracy is not None:
            st.markdown(f"**Chronological holdout accuracy:** {validation_accuracy:.1%}")
        else:
            st.markdown("**Chronological holdout accuracy:** Not available")
        st.caption(model_message)
        st.caption("Health score is a heuristic indicator for this demo, not a certified safety or reliability metric.")

with right:
    with st.container(border=True):
        st.markdown("#### Findings")
        for item in observations:
            st.markdown(f"- {item}")
        st.markdown("#### Recommended engineering actions")
        for index, action in enumerate(recommendations, start=1):
            st.markdown(f"**{index}.** {action}")

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
