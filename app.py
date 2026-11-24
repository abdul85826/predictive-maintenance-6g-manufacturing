"""
Predictive Intelligence: 6G Smart Manufacturing Predictive Maintenance Console

Author: Syed Abdul Rehman (Register No: 230071601233)
Department of Computer Science and Engineering
B.S. Abdur Rahman Crescent Institute of Science and Technology
Internship Organization: Unified Mentor Pvt. Ltd.
Framework: Isolation Forest Unsupervised Anomaly Detection & 6G URLLC Telemetry
"""

import os
import sys
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.runtime

# Base project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "processed_dataset.csv")
RAW_DATA_PATH = os.path.join(BASE_DIR, "dataset.csv")
FAVICON_PATH = os.path.join(BASE_DIR, "assets", "branding", "favicon.svg")
LOGO_PATH = os.path.join(BASE_DIR, "assets", "branding", "project_logo.svg")

# Plotly Natural Industrial Global Layout Theme
INDUSTRIAL_LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, -apple-system, Segoe UI, sans-serif", color="#202A2E", size=12),
    margin=dict(l=40, r=30, t=45, b=40),
    xaxis=dict(
        gridcolor="#E2E7E4",
        linecolor="#E2E7E4",
        tickcolor="#66747A",
        tickfont=dict(color="#66747A", size=11),
        title_font=dict(color="#202A2E", size=12, family="Inter, sans-serif"),
    ),
    yaxis=dict(
        gridcolor="#E2E7E4",
        linecolor="#E2E7E4",
        tickcolor="#66747A",
        tickfont=dict(color="#66747A", size=11),
        title_font=dict(color="#202A2E", size=12, family="Inter, sans-serif"),
    ),
    legend=dict(
        font=dict(color="#202A2E", size=11),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="#E2E7E4",
        borderwidth=1,
    ),
)


@st.cache_data
def load_telemetry_data():
    """
    Loads telemetry dataset safely with fallback handling.
    Prefers processed dataset with ML anomaly scores.
    """
    if os.path.exists(PROCESSED_DATA_PATH):
        df = pd.read_csv(PROCESSED_DATA_PATH)
        is_processed = True
    elif os.path.exists(RAW_DATA_PATH):
        df = pd.read_csv(RAW_DATA_PATH)
        is_processed = False
    else:
        df = pd.DataFrame()
        is_processed = False
    return df, is_processed


def run_dashboard():
    """
    Renders the complete Natural Industrial Predictive Intelligence dashboard.
    """
    # Page Configuration
    st.set_page_config(
        page_title="Predictive Intelligence | 6G Smart Manufacturing",
        page_icon=FAVICON_PATH if os.path.exists(FAVICON_PATH) else "⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Natural Industrial Design System Custom CSS
    st.markdown("""
    <style>
        /* Global Workspace Styling */
        .stApp {
            background-color: #F5F6F2;
            color: #202A2E;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        }
        
        /* Main container padding */
        .block-container {
            padding-top: 1.8rem;
            padding-bottom: 3rem;
            padding-left: 2rem;
            padding-right: 2rem;
            max-width: 1440px;
        }

        /* Sidebar Clean Surface */
        [data-testid="stSidebar"] {
            background-color: #FFFFFF;
            border-right: 1px solid #E2E7E4;
            padding-top: 1rem;
        }

        /* Natural Industrial Card Surface */
        .industrial-card {
            background-color: #FFFFFF;
            border: 1px solid #E2E7E4;
            border-radius: 8px;
            padding: 20px 24px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
            margin-bottom: 20px;
        }

        /* KPI Metric Tiles */
        .kpi-tile {
            background-color: #FFFFFF;
            border: 1px solid #E2E7E4;
            border-radius: 8px;
            padding: 16px 20px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            min-height: 104px;
        }

        .kpi-label {
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: #66747A;
            margin-bottom: 4px;
        }

        .kpi-value {
            font-size: 1.95rem;
            font-weight: 700;
            line-height: 1.1;
            color: #176B62;
        }

        .kpi-subtext {
            font-size: 0.78rem;
            color: #66747A;
            margin-top: 4px;
        }

        /* Header Context Container */
        .header-box {
            background-color: #FFFFFF;
            border: 1px solid #E2E7E4;
            border-radius: 8px;
            padding: 22px 28px;
            margin-bottom: 24px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
        }

        .header-title {
            font-size: 1.7rem;
            font-weight: 700;
            color: #202A2E;
            letter-spacing: -0.4px;
            margin: 0;
        }

        .header-caption {
            font-size: 0.92rem;
            color: #66747A;
            margin-top: 6px;
            margin-bottom: 12px;
        }

        /* Pills & Badges */
        .badge-status {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 4px;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }

        .badge-high {
            background-color: #FDE8E8;
            color: #C44F4F;
            border: 1px solid #F8B4B4;
        }

        .badge-med {
            background-color: #FEF3C7;
            color: #C58A32;
            border: 1px solid #FCD34D;
        }

        .badge-low {
            background-color: #DEF7EC;
            color: #39845A;
            border: 1px solid #BCF0DA;
        }

        .badge-info {
            background-color: #E2E7E4;
            color: #176B62;
            border: 1px solid #D1D5DB;
        }

        /* Clean Section Headings */
        .section-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: #202A2E;
            margin-top: 24px;
            margin-bottom: 14px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .section-title::before {
            content: "";
            display: inline-block;
            width: 4px;
            height: 18px;
            background-color: #176B62;
            border-radius: 2px;
        }

        /* Plotly Chart Container */
        .chart-surface {
            background-color: #FFFFFF;
            border: 1px solid #E2E7E4;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 16px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
        }
    </style>
    """, unsafe_allow_html=True)

    df_raw, has_ml_scores = load_telemetry_data()

    # ==============================================================================
    # SIDEBAR NAVIGATION & FILTER CONTROLS
    # ==============================================================================

    # Render original local SVG brand logo
    if os.path.exists(LOGO_PATH):
        with open(LOGO_PATH, "r", encoding="utf-8") as f:
            svg_content = f.read()
        st.sidebar.markdown(f'<div style="margin-bottom: 20px; padding: 0 4px;">{svg_content}</div>', unsafe_allow_html=True)
    else:
        st.sidebar.title("PREDICTIVE INTELLIGENCE")
        st.sidebar.caption("6G Smart Manufacturing Analytics")

    st.sidebar.markdown("<hr style='border: none; border-top: 1px solid #E2E7E4; margin: 12px 0 20px 0;'>", unsafe_allow_html=True)

    if df_raw.empty:
        st.sidebar.error("Telemetry data unavailable. Please execute generate_dataset.py and ml_pipeline.py.")
        st.error("No telemetry dataset found. Please run generate_dataset.py followed by ml_pipeline.py.")
        return

    # Prepare dates for temporal filtering
    df_raw["Datetime"] = pd.to_datetime(df_raw["Date"] + " " + df_raw["Time"])
    min_date = df_raw["Datetime"].dt.date.min()
    max_date = df_raw["Datetime"].dt.date.max()

    # Session State Initialization for Filter Reset
    if "selected_machine" not in st.session_state:
        st.session_state["selected_machine"] = "All Machines"
    if "selected_modes" not in st.session_state:
        st.session_state["selected_modes"] = sorted(df_raw["Operation_Mode"].unique().tolist())
    if "risk_threshold" not in st.session_state:
        st.session_state["risk_threshold"] = 35
    if "date_range" not in st.session_state:
        st.session_state["date_range"] = (min_date, max_date)

    # Machine ID Selector
    machine_list = ["All Machines"] + sorted(df_raw["Machine_ID"].unique().tolist())
    selected_machine = st.sidebar.selectbox(
        "Asset / Machine Selection",
        options=machine_list,
        index=machine_list.index(st.session_state["selected_machine"]) if st.session_state["selected_machine"] in machine_list else 0,
        help="Select an individual machine cell (M_001 to M_010) or view aggregated fleet telemetry."
    )

    # Date Range Filter
    date_range = st.sidebar.date_input(
        "Observation Date Range",
        value=st.session_state["date_range"],
        min_value=min_date,
        max_value=max_date,
        help="Filter telemetry recordings between specific calendar dates."
    )

    # Operating Mode Multiselect
    available_modes = sorted(df_raw["Operation_Mode"].unique().tolist())
    selected_modes = st.sidebar.multiselect(
        "Operating Regimes",
        options=available_modes,
        default=st.session_state["selected_modes"],
        help="Filter by machine operating mode: Normal (65%), High-Load (25%), Eco-Mode (10%)."
    )

    # Anomaly Risk Threshold Slider
    risk_threshold = st.sidebar.slider(
        "Anomaly Score Alert Cutoff",
        min_value=0,
        max_value=100,
        value=st.session_state["risk_threshold"],
        step=5,
        help="Telemetry records with Anomaly Score >= cutoff will populate the inspection queue."
    )

    # Reset Filters Button
    if st.sidebar.button("Reset All Filters", use_container_width=True):
        st.session_state["selected_machine"] = "All Machines"
        st.session_state["selected_modes"] = available_modes
        st.session_state["risk_threshold"] = 35
        st.session_state["date_range"] = (min_date, max_date)
        st.rerun()

    st.sidebar.markdown("<hr style='border: none; border-top: 1px solid #E2E7E4; margin: 24px 0 16px 0;'>", unsafe_allow_html=True)
    st.sidebar.markdown("""
    <div style="font-size: 0.8rem; color: #66747A; line-height: 1.5;">
        <strong>Academic Verification:</strong><br/>
        B.Tech CSE Internship Project<br/>
        B.S. Abdur Rahman Crescent Institute of Science and Technology<br/>
        Candidate: Syed Abdul Rehman (230071601233)<br/>
        Organization: Unified Mentor Pvt. Ltd.
    </div>
    """, unsafe_allow_html=True)

    # ==============================================================================
    # DATASET FILTERING ENGINE
    # ==============================================================================

    filtered_df = df_raw.copy()

    # Filter by Machine
    if selected_machine != "All Machines":
        filtered_df = filtered_df[filtered_df["Machine_ID"] == selected_machine]

    # Filter by Operating Mode
    if selected_modes:
        filtered_df = filtered_df[filtered_df["Operation_Mode"].isin(selected_modes)]
    else:
        st.warning("Please select at least one Operating Regime from the sidebar.")
        return

    # Filter by Date Range
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_d, end_d = date_range
        filtered_df = filtered_df[
            (filtered_df["Datetime"].dt.date >= start_d) & 
            (filtered_df["Datetime"].dt.date <= end_d)
        ]

    # ==============================================================================
    # HEADER BANNER & SYSTEM STATUS
    # ==============================================================================

    total_telemetry_count = len(filtered_df)
    monitored_machines_count = filtered_df["Machine_ID"].nunique()

    st.markdown(f"""
    <div class="header-box">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
            <div>
                <h1 class="header-title">Predictive Maintenance & Anomaly Detection</h1>
                <div class="header-caption">
                    6G-Enabled Smart Manufacturing Intelligence Console • Unsupervised Condition Monitoring System
                </div>
                <div style="display: flex; gap: 8px; flex-wrap: wrap; align-items: center;">
                    <span class="badge-status badge-info">Fleet Monitored: {monitored_machines_count} Assets</span>
                    <span class="badge-status badge-low">Telemetry: Synchronized</span>
                    <span class="badge-status {'badge-high' if total_telemetry_count == 0 else 'badge-info'}">Active Records: {total_telemetry_count:,}</span>
                    <span class="badge-status badge-info">Algorithm: Isolation Forest (150 iTrees)</span>
                </div>
            </div>
            <div style="text-align: right; font-size: 0.8rem; color: #66747A;">
                <strong>Simulation Window:</strong> June 1, 2026 – June 30, 2026<br/>
                <strong>Sampling Interval:</strong> 40 Minutes (36 Readings/Day)<br/>
                <strong>QoS Standard:</strong> 6G URLLC Sub-ms Simulation
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ==============================================================================
    # KEY PERFORMANCE INDICATOR (KPI) RIBBON
    # ==============================================================================

    if has_ml_scores and "Anomaly_Score" in filtered_df.columns:
        high_risk_records = len(filtered_df[filtered_df["Anomaly_Score"] >= 65.0])
        avg_score = filtered_df["Anomaly_Score"].mean() if total_telemetry_count > 0 else 0.0
        high_risk_pct = (high_risk_records / max(1, total_telemetry_count)) * 100.0
        fleet_health_pct = (len(filtered_df[filtered_df["Anomaly_Score"] < 35.0]) / max(1, total_telemetry_count)) * 100.0
    else:
        high_risk_records = 0
        avg_score = 0.0
        high_risk_pct = 0.0
        fleet_health_pct = 100.0

    kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)

    with kpi_c1:
        st.markdown(f"""
        <div class="kpi-tile">
            <div class="kpi-label">Active Telemetry Records</div>
            <div class="kpi-value">{total_telemetry_count:,}</div>
            <div class="kpi-subtext">Across {monitored_machines_count} machine cells in view</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_c2:
        st.markdown(f"""
        <div class="kpi-tile">
            <div class="kpi-label">Average Anomaly Score</div>
            <div class="kpi-value">{avg_score:.1f} <span style="font-size: 1rem; color: #66747A; font-weight: normal;">/ 100</span></div>
            <div class="kpi-subtext">Population baseline: 22.7 (Normal operational state)</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_c3:
        st.markdown(f"""
        <div class="kpi-tile">
            <div class="kpi-label">Critical High-Risk Spikes</div>
            <div class="kpi-value" style="color: {'#C44F4F' if high_risk_records > 0 else '#39845A'};">{high_risk_records} <span style="font-size: 1rem; font-weight: normal;">({high_risk_pct:.1f}%)</span></div>
            <div class="kpi-subtext">Score &ge; 65 (Immediate intervention required)</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_c4:
        st.markdown(f"""
        <div class="kpi-tile">
            <div class="kpi-label">Fleet Operational Health</div>
            <div class="kpi-value" style="color: #39845A;">{fleet_health_pct:.1f}%</div>
            <div class="kpi-subtext">Proportion of operations in Low Risk tier</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    # ==============================================================================
    # SECTION 1: FLEET MAINTENANCE RISK DISTRIBUTION & MACHINE BREAKDOWN
    # ==============================================================================

    st.markdown('<div class="section-title">Fleet Risk Profile & Machine Failure Breakdown</div>', unsafe_allow_html=True)

    row1_col1, row1_col2 = st.columns([1, 1])

    with row1_col1:
        st.markdown('<div class="chart-surface">', unsafe_allow_html=True)
        st.markdown("<strong style='font-size: 0.95rem; color: #202A2E;'>Asset Risk Tier Distribution (Donut Chart)</strong>", unsafe_allow_html=True)
        if has_ml_scores and "Maintenance_Risk_Level" in filtered_df.columns:
            risk_dist = filtered_df["Maintenance_Risk_Level"].value_counts().reset_index()
            risk_dist.columns = ["Risk_Level", "Count"]

            color_map = {
                "Low": "#39845A",      # Natural Green
                "Medium": "#C58A32",   # Muted Amber
                "High": "#C44F4F"      # Controlled Red
            }

            fig_pie = px.pie(
                risk_dist,
                values="Count",
                names="Risk_Level",
                color="Risk_Level",
                color_discrete_map=color_map,
                hole=0.48,
            )
            fig_pie.update_layout(**INDUSTRIAL_LAYOUT_DEFAULTS)
            fig_pie.update_layout(
                height=320,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
            )
            fig_pie.update_traces(
                textposition="inside",
                textinfo="percent+label",
                marker=dict(line=dict(color="#FFFFFF", width=2))
            )
            st.plotly_chart(fig_pie)
        else:
            st.info("Execute ml_pipeline.py to compute multi-tier risk distributions.")
        st.markdown('</div>', unsafe_allow_html=True)

    with row1_col2:
        st.markdown('<div class="chart-surface">', unsafe_allow_html=True)
        st.markdown("<strong style='font-size: 0.95rem; color: #202A2E;'>High-Risk Telemetry Incidents by Machine Asset</strong>", unsafe_allow_html=True)
        if has_ml_scores and "Maintenance_Risk_Level" in filtered_df.columns:
            high_risk_df = filtered_df[filtered_df["Maintenance_Risk_Level"] == "High"]
            if not high_risk_df.empty:
                machine_risk = high_risk_df["Machine_ID"].value_counts().reset_index()
                machine_risk.columns = ["Machine_ID", "High_Risk_Events"]
                machine_risk = machine_risk.sort_values(by="High_Risk_Events", ascending=True)

                fig_bar = px.bar(
                    machine_risk,
                    x="High_Risk_Events",
                    y="Machine_ID",
                    orientation="h",
                    text="High_Risk_Events",
                    labels={"High_Risk_Events": "Critical Events (Score >= 65)", "Machine_ID": "Machine Asset"},
                    color="High_Risk_Events",
                    color_continuous_scale=[[0.0, "#F8B4B4"], [1.0, "#C44F4F"]]
                )
                fig_bar.update_layout(
                    **INDUSTRIAL_LAYOUT_DEFAULTS,
                    height=320,
                    coloraxis_showscale=False
                )
                fig_bar.update_traces(textposition="outside", marker_line_color="#E2E7E4", marker_line_width=1)
                st.plotly_chart(fig_bar)
            else:
                st.success("Zero high-risk telemetry events identified under active filter criteria.")
        else:
            st.info("Run ml_pipeline.py to compute machine degradation rankings.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ==============================================================================
    # SECTION 2: ANOMALY TIMELINE & MULTIVARIATE SENSOR INSTABILITY
    # ==============================================================================

    st.markdown('<div class="section-title">Telemetry Anomaly Drift & Multivariate Correlations</div>', unsafe_allow_html=True)

    st.markdown('<div class="chart-surface">', unsafe_allow_html=True)
    st.markdown("<strong style='font-size: 0.95rem; color: #202A2E;'>Time-Series Anomaly Score Drift Across Machine Lifecycle</strong>", unsafe_allow_html=True)

    if has_ml_scores and "Anomaly_Score" in filtered_df.columns:
        fig_timeline = px.line(
            filtered_df,
            x="Datetime",
            y="Anomaly_Score",
            color="Machine_ID",
            labels={"Anomaly_Score": "Anomaly Score (0 - 100)", "Datetime": "Timestamp"},
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig_timeline.add_hline(
            y=risk_threshold,
            line_dash="dash",
            line_color="#C58A32",
            line_width=2,
            annotation_text=f"Active Alert Threshold ({risk_threshold})",
            annotation_position="top left",
            annotation_font=dict(color="#C58A32", size=10)
        )
        fig_timeline.add_hline(
            y=65.0,
            line_dash="solid",
            line_color="#C44F4F",
            line_width=2,
            annotation_text="Critical Action Threshold (65)",
            annotation_position="bottom left",
            annotation_font=dict(color="#C44F4F", size=10)
        )
        fig_timeline.update_layout(
            **INDUSTRIAL_LAYOUT_DEFAULTS,
            height=360,
            hovermode="x unified"
        )
        st.plotly_chart(fig_timeline)
    else:
        st.info("Anomaly Score timeline requires running ml_pipeline.py first.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Side-by-side scatter analysis
    scat_col1, scat_col2 = st.columns(2)

    with scat_col1:
        st.markdown('<div class="chart-surface">', unsafe_allow_html=True)
        st.markdown("<strong style='font-size: 0.95rem; color: #202A2E;'>Thermal vs. Vibration Instability Dynamics</strong>", unsafe_allow_html=True)
        if not filtered_df.empty:
            fig_scatter_phys = px.scatter(
                filtered_df,
                x="Temperature_C",
                y="Vibration_Hz",
                color="Anomaly_Score" if "Anomaly_Score" in filtered_df.columns else "Operation_Mode",
                size="Power_Consumption_kW",
                hover_data=["Machine_ID", "Operation_Mode", "Production_Speed_units_per_hr"],
                labels={
                    "Temperature_C": "Core Temperature (°C)",
                    "Vibration_Hz": "Structural Vibration (Hz)",
                    "Anomaly_Score": "Score",
                    "Power_Consumption_kW": "Power (kW)"
                },
                color_continuous_scale=[[0.0, "#39845A"], [0.5, "#C58A32"], [1.0, "#C44F4F"]]
            )
            fig_scatter_phys.update_layout(**INDUSTRIAL_LAYOUT_DEFAULTS, height=340)
            st.plotly_chart(fig_scatter_phys)
        st.markdown('</div>', unsafe_allow_html=True)

    with scat_col2:
        st.markdown('<div class="chart-surface">', unsafe_allow_html=True)
        st.markdown("<strong style='font-size: 0.95rem; color: #202A2E;'>Simulated 6G URLLC QoS: Latency vs. Packet Loss</strong>", unsafe_allow_html=True)
        if not filtered_df.empty:
            fig_scatter_net = px.scatter(
                filtered_df,
                x="Network_Latency_ms",
                y="Packet_Loss_%",
                color="Anomaly_Score" if "Anomaly_Score" in filtered_df.columns else "Operation_Mode",
                size="Error_Rate_%",
                hover_data=["Machine_ID", "Operation_Mode", "Error_Rate_%"],
                labels={
                    "Network_Latency_ms": "6G Latency (ms)",
                    "Packet_Loss_%": "Telemetry Packet Loss (%)",
                    "Anomaly_Score": "Score",
                    "Error_Rate_%": "Error Rate (%)"
                },
                color_continuous_scale=[[0.0, "#176B62"], [0.5, "#477D83"], [1.0, "#C44F4F"]]
            )
            fig_scatter_net.update_layout(**INDUSTRIAL_LAYOUT_DEFAULTS, height=340)
            st.plotly_chart(fig_scatter_net)
        st.markdown('</div>', unsafe_allow_html=True)

    # ==============================================================================
    # SECTION 3: PRIORITY MAINTENANCE ALERT QUEUE & PRESCRIPTIVE ACTIONS
    # ==============================================================================

    st.markdown('<div class="section-title">Priority Maintenance Inspection Queue</div>', unsafe_allow_html=True)
    st.markdown('<div class="industrial-card">', unsafe_allow_html=True)

    if has_ml_scores and "Anomaly_Score" in filtered_df.columns:
        alerts_df = filtered_df[filtered_df["Anomaly_Score"] >= risk_threshold].copy()
        
        if not alerts_df.empty:
            def get_prescriptive_action(score: float, temp: float, vib: float) -> str:
                if score >= 65.0:
                    if temp > 75.0 and vib > 55.0:
                        return "CRITICAL: Immediate spindle bearing overhaul & thermal sensor verification"
                    elif temp > 75.0:
                        return "CRITICAL: Coolant flushing & thermal runaway emergency inspection"
                    else:
                        return "CRITICAL: Urgent rotor re-alignment & bearing lubrication"
                else:
                    return "WARNING: Schedule vibration sensor calibration & lubricant check (Next Shift)"

            def get_lead_time_window(score: float) -> str:
                if score >= 75.0:
                    return "< 4 Hours"
                elif score >= 65.0:
                    return "4 - 12 Hours"
                else:
                    return "12 - 24 Hours"

            alerts_df["Prescriptive_Action"] = alerts_df.apply(
                lambda r: get_prescriptive_action(r["Anomaly_Score"], r["Temperature_C"], r["Vibration_Hz"]), axis=1
            )
            alerts_df["Estimated_Lead_Time"] = alerts_df["Anomaly_Score"].apply(get_lead_time_window)

            display_cols = [
                "Date", "Time", "Machine_ID", "Operation_Mode",
                "Anomaly_Score", "Maintenance_Risk_Level",
                "Temperature_C", "Vibration_Hz", "Estimated_Lead_Time", "Prescriptive_Action"
            ]

            alerts_sorted = alerts_df[display_cols].sort_values(by="Anomaly_Score", ascending=False)
            
            st.caption(f"Showing {len(alerts_sorted):,} telemetry instances exceeding active alert cutoff (Score >= {risk_threshold}).")
            st.dataframe(alerts_sorted)

            csv_data = alerts_sorted.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Active Maintenance Alerts (CSV)",
                data=csv_data,
                file_name=f"pdm_maintenance_alerts_{selected_machine}.csv",
                mime="text/csv",
                help="Download filtered alert queue for field technician work orders."
            )
        else:
            st.success(f"Fleet Clear: Zero machine telemetry instances currently exceed the active threshold (Score >= {risk_threshold}).")
    else:
        st.info("Execute ml_pipeline.py to populate automated prescriptive maintenance queues.")

    st.markdown('</div>', unsafe_allow_html=True)

    # ==============================================================================
    # SECTION 4: TEMPORAL RISK ESCALATION HEATMAP
    # ==============================================================================

    st.markdown('<div class="section-title">Fleet-Wide Temporal Degradation Heatmap</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-surface">', unsafe_allow_html=True)
    st.markdown("<strong style='font-size: 0.95rem; color: #202A2E;'>30-Day Mean Anomaly Score Escalation by Machine Asset</strong>", unsafe_allow_html=True)

    if has_ml_scores and "Anomaly_Score" in filtered_df.columns:
        daily_heatmap = filtered_df.groupby(["Date", "Machine_ID"])["Anomaly_Score"].mean().reset_index()
        fig_heatmap = px.density_heatmap(
            daily_heatmap,
            x="Date",
            y="Machine_ID",
            z="Anomaly_Score",
            labels={"Date": "Observation Date", "Machine_ID": "Machine Asset", "Anomaly_Score": "Mean Anomaly Score"},
            color_continuous_scale=[[0.0, "#DEF7EC"], [0.35, "#FEF3C7"], [0.65, "#FDE8E8"], [1.0, "#C44F4F"]]
        )
        fig_heatmap.update_layout(**INDUSTRIAL_LAYOUT_DEFAULTS, height=320)
        st.plotly_chart(fig_heatmap)
        st.caption("Visual confirmation of machine degradation inception. Machines M_009 and M_007 display notable thermal and mechanical drift past Day 15.")
    else:
        st.info("Run ml_pipeline.py to generate daily temporal escalation heatmaps.")

    st.markdown('</div>', unsafe_allow_html=True)

    # ==============================================================================
    # SECTION 5: METHODOLOGY & SCIENTIFIC DATA DISCLOSURE
    # ==============================================================================

    st.markdown('<div class="section-title">Methodology & Academic Data Disclosure</div>', unsafe_allow_html=True)

    st.markdown(r"""
    <div class="industrial-card" style="font-size: 0.88rem; color: #202A2E; line-height: 1.6;">
        <div style="font-weight: 700; color: #176B62; margin-bottom: 8px;">
            ENGINEERING SPECIFICATION & RESEARCH FRAMEWORK
        </div>
        <p>
            <strong>1. Unsupervised Anomaly Detection Principle:</strong> 
            In smart manufacturing, machine failures are rare and lack extensive historical training labels. 
            This platform employs an <strong>Isolation Forest</strong> algorithm ($n_{\text{estimators}} = 150$, $\text{contamination} = 0.08$) 
            which isolates anomalies by recursively constructing random binary search trees. Shorter path lengths denote higher isolation propensity. 
            Raw decision scores are inverted and normalized via <code>MinMaxScaler</code> to a continuous 0–100 Anomaly Score.
        </p>
        <p>
            <strong>2. Dynamic Rolling Baselines:</strong> 
            Static SCADA thresholding fails during operational regime transitions (e.g., High-Load vs. Eco-Mode). 
            This implementation extracts 10-period per-machine grouped rolling averages, computing transient deviations 
            ($\Delta T = T - \bar{T}_{\text{rolling}}$ and $\Delta V = V - \bar{V}_{\text{rolling}}$) alongside physical 
            vibration-to-power instability ratios ($\frac{\text{Vibration}}{\text{Power} + 10^{-5}}$).
        </p>
        <p>
            <strong>3. Transparent Simulation Boundaries:</strong>
            Telemetry represents synthetic industrial physics generated across 10 machines over 30 days (10,800 records). 
            The 6G communication metrics (sub-millisecond latency and packet loss) are modeled mathematically using 
            Gamma and Normal distributions to evaluate algorithmic viability in next-generation industrial IoT networks; 
            no physical 6G hardware, physical accelerometers, or write-back PLC controllers are connected to this software instance.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div style="text-align: center; font-size: 0.8rem; color: #66747A; margin-top: 30px; margin-bottom: 20px;">
        Predictive Intelligence Console • B.Tech CSE Internship Project (2026) • B.S. Abdur Rahman Crescent Institute of Science and Technology
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    if streamlit.runtime.exists():
        run_dashboard()
    else:
        # User executed python app.py directly: automatically launch via Streamlit CLI
        from streamlit.web import cli as stcli
        sys.argv = ["streamlit", "run", os.path.abspath(__file__)] + sys.argv[1:]
        sys.exit(stcli.main())
