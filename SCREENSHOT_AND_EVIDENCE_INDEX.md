# SCREENSHOT AND EVIDENCE INDEX SPECIFICATION
## PREDICTIVE MAINTENANCE FOR 6G-ENABLED SMART MANUFACTURING

This index defines every figure, visual evidence item, terminal log capture, and UI component required for the B.Tech academic internship report, specifying its exact caption, data state, source module, and recommended placement in the university report chapters.

---

## 1. EVIDENCE AND SCREENSHOT MASTER INVENTORY

| Evidence ID | Figure Title / Description | Source File / Execution State | Data State Represented | Recommended Report Chapter |
| :---: | :--- | :--- | :--- | :---: |
| **FIG-01** | Executive Control Tower: Glassmorphic Main Dashboard & KPI Banner | `app.py`: Top Banner & KPI Bar (Lines 105–148) | All 10 Machines, unfiltered. Shows Avg Score: 22.7, Spikes: 146, Lead Time: 18.5 hrs, Prevention: 96.1%. | **Chapter 5 (Section 5.10)** |
| **FIG-02** | Fleet Asset Maintenance Risk Distribution (Donut Chart) | `app.py`: Tab 1 Left Column (Lines 164–176) | Plotly Donut Chart showing Low (83.6%), Medium (15.1%), High (1.35%). | **Chapter 6 (Section 6.2)** |
| **FIG-03** | Machine Breakdown by High-Risk Telemetry Events (Bar Chart) | `app.py`: Tab 1 Right Column (Lines 181–192) | Bar chart displaying critical spikes per machine, highlighting `M_009` (78 spikes) and `M_007` (23 spikes). | **Chapter 6 (Section 6.3)** |
| **FIG-04** | Time-Series Anomaly Score Drift & Threshold Alert Boundaries | `app.py`: Tab 2 Line Chart (Lines 195–205) | Plotly line chart over 30 days with orange dashed line (35 alert threshold) and red solid line (65 critical threshold). | **Chapter 5 (Section 5.11)** |
| **FIG-05** | Multivariate Thermal vs. Vibration Instability Scatter Plot | `app.py`: Tab 2 Scatter Col A (Lines 209–218) | Scatter of Temperature (°C) vs. Vibration (Hz), color-scaled by Anomaly Score, sized by Power (kW). | **Chapter 6 (Section 6.2)** |
| **FIG-06** | 6G Communication Telemetry: Latency vs. Packet Loss Correlation | `app.py`: Tab 2 Scatter Col B (Lines 220–228) | Scatter of Latency (ms) vs. Packet Loss (%), color-scaled by Anomaly Score, sized by Error Rate (%). | **Chapter 4 (Section 4.5)** |
| **FIG-07** | Priority Maintenance Alert Queue & Prescriptive Inspection Table | `app.py`: Tab 3 Dataframe (Lines 229–247) | Filtered inspection table with Anomaly Score, Lead Time Window (< 6 hrs), and Prescriptive Actions. | **Chapter 5 (Section 5.10)** |
| **FIG-08** | Temporal Heatmap of Machine Risk Intensity over Days | `app.py`: Tab 4 Heatmap (Lines 249–265) | Plotly density heatmap (Date vs. Machine_ID vs. Anomaly Score) illustrating progressive wear degradation. | **Chapter 6 (Section 6.3)** |
| **FIG-09** | Interactive Sidebar Controls & Operation Mode Filtering | `app.py`: Sidebar (Lines 82–98) | Machine Selector dropdown, Operation Mode multiselect, and Anomaly Threshold slider. | **Chapter 5 (Section 5.10)** |
| **FIG-10** | Deterministic Dataset Generation Terminal Output | `generate_dataset.py` execution console | Terminal log showing: `Dataset successfully generated with 10800 rows and saved to dataset.csv.` | **Chapter 5 (Section 5.4)** |
| **FIG-11** | Machine Learning Pipeline Execution & Training Output | `ml_pipeline.py` execution console | Console log showing 10,800 records, 146 High-Risk instances (1.35%), and artifact persistence. | **Chapter 5 (Section 5.7)** |
| **FIG-12** | Git Repository Structure and Commit History Log | `git log --oneline` & `git remote -v` | Verification of student commit author (`SYED ABDUL REHMAN <230071601233@crescent.education>`). | **Chapter 2 (Section 2.2)** |

---

## 2. DETAILED SCREENSHOT CAPTURE SPECIFICATIONS

### Figure 1: Executive Control Tower Banner & KPI Metrics
- **Visual Composition:** Glassmorphic dark gradient banner featuring the lightning bolt icon (`⚡`), project header, Unified Mentor & Thales Group caption, and four centered glass metric cards:
  1. *Avg Anomaly Score:* `22.7` (Cyan font `#38bdf8`)
  2. *High-Risk Telemetry Spikes:* `146` (Crimson font `#ef4444`)
  3. *Early Lead Time Warning:* `18.5 hrs` (Emerald green font `#10b981`)
  4. *Downtime Prevention Index:* `96.1%` (Amber font `#f59e0b`)
- **Report Placement:** Chapter 5, Section 5.10 (Application Interface Design).
- **Caption:** *Figure 5.1: Real-time Executive Control Tower interface displaying fleet-wide KPI cards and glassmorphic telemetry cards.*

### Figure 2: Fleet Asset Maintenance Risk Distribution
- **Visual Composition:** A high-contrast Plotly donut pie chart with transparent background (`hole=0.4`) categorized into:
  - *Low Risk (Emerald Green #10b981):* 9,023 records (83.55%)
  - *Medium Risk (Amber #f59e0b):* 1,631 records (15.10%)
  - *High Risk (Red #ef4444):* 146 records (1.35%)
- **Report Placement:** Chapter 6, Section 6.2 (Observed Application Outputs).
- **Caption:** *Figure 6.1: Proportional distribution of telemetry records across the three maintenance risk tiers.*

### Figure 3: Machine Breakdown by High-Risk Telemetry
- **Visual Composition:** Vertical bar chart ranking machines along the x-axis (`Machine_ID`) against the frequency of High-Risk events along the y-axis, colored with a continuous `Reds` gradient.
- **Key Visual Focus:** The chart clearly reveals `M_009` as the critical outlier (78 events), followed by `M_007` (23 events), while `M_001` and `M_010` show minimal risk.
- **Report Placement:** Chapter 6, Section 6.3 (Machine Degradation and Failure Profiles).
- **Caption:** *Figure 6.2: Machine-specific distribution of critical high-risk telemetry spikes, identifying M_009 as the primary degraded asset.*

### Figure 4: Time-Series Anomaly Score Drift
- **Visual Composition:** Multi-line time-series graph plotting `Anomaly_Score` against `Datetime` (June 1 to June 30, 2026), colored by `Machine_ID`. Two horizontal reference lines are prominently displayed:
  - Orange dashed line at $y = 35$ ("Risk Alert Threshold").
  - Red solid line at $y = 65$ ("Urgent Critical Threshold").
- **Key Visual Focus:** The lines for healthy machines oscillate below 35, while `M_009` and `M_007` display progressive upward trajectories breaching the 65 threshold past day 20.
- **Report Placement:** Chapter 5, Section 5.11 (Visualization and Temporal Trend Analytics).
- **Caption:** *Figure 5.2: Multi-machine anomaly score drift over the 30-day operational lifecycle showing progressive threshold breaches.*

### Figure 5: Temperature vs. Vibration Instability Scatter
- **Visual Composition:** 2D scatter plot plotting `Temperature_C` (x-axis, 48°C to 90°C) against `Vibration_Hz` (y-axis, 36 Hz to 66 Hz). Data points are colored using the `Viridis` color scale representing `Anomaly_Score`, with marker diameter scaled by `Power_Consumption_kW`.
- **Key Visual Focus:** Normal data clusters tightly between 55–68°C and 40–55 Hz. Anomalous high-risk instances drift to the top-right quadrant (> 75°C, > 55 Hz).
- **Report Placement:** Chapter 6, Section 6.2 (Multivariate Failure Correlations).
- **Caption:** *Figure 6.3: Bivariate scatter plot of temperature versus vibration instability, illustrating multivariate anomaly clustering.*

### Figure 6: 6G Network QoS Impact (Latency vs. Packet Loss)
- **Visual Composition:** Scatter plot plotting `Network_Latency_ms` (x-axis) vs. `Packet_Loss_%` (y-axis), colored with the `Plasma` scale based on `Anomaly_Score`, with point size driven by `Error_Rate_%`.
- **Key Visual Focus:** Highlights the correlation between network performance degradation and operational error spikes under simulated 6G URLLC conditions.
- **Report Placement:** Chapter 4, Section 4.5 (6G Sensor Telemetry and QoS Analytics).
- **Caption:** *Figure 4.1: Simulated 6G network latency versus packet loss distribution and its correlation with machine anomaly severity.*

### Figure 7: Priority Maintenance Alert Queue
- **Visual Composition:** Streamlit interactive dataframe table filtered to high-risk events, highlighting:
  - Columns: `Date`, `Time`, `Machine_ID`, `Operation_Mode`, `Anomaly_Score`, `Maintenance_Risk_Level`, `Temperature_C`, `Vibration_Hz`, `Lead_Time_Window`, `Recommended_Action`.
  - Prescriptive tags: `🚨 Immediate Bearing Lubrication & Thermal Inspection` (< 6 Hours lead time).
- **Report Placement:** Chapter 5, Section 5.10 (Maintenance Alert Panel).
- **Caption:** *Figure 5.3: Prioritized maintenance dispatch table with automated prescriptive actions and lead-time windows.*

### Figure 8: Temporal Heatmap of Machine Risk Intensity
- **Visual Composition:** Plotly density heatmap with `Date` on the x-axis, `Machine_ID` on the y-axis, and `Anomaly_Score` on the color z-axis using the `Inferno` color scale.
- **Key Visual Focus:** Displays a visual transition from dark/cool colors during the first 15 days to intense yellow/white hot zones for `M_009` and `M_007` during the final 10 days of the month.
- **Report Placement:** Chapter 6, Section 6.3 (Temporal Risk Escalation).
- **Caption:** *Figure 6.4: 30-day temporal heatmap showing progressive risk escalation across machine assets.*

---

## 3. HOW TO CAPTURE HIGH-RESOLUTION SCREENSHOTS FOR THE REPORT

To capture high-resolution figures for your B.Tech report:
1. Open PowerShell and navigate to the project directory:
   ```powershell
   cd d:\Documents\NE_App\Nutrient_Earth\predictive-maintenance-6g-manufacturing
   .\venv\Scripts\Activate.ps1
   streamlit run app.py
   ```
2. Open your web browser at `http://localhost:8501`.
3. Set your browser zoom to 100% or 125% for optimal DPI rendering.
4. For each figure:
   - Use the Plotly in-chart **"Download plot as a png"** camera icon located in the top-right of every chart canvas to save high-resolution, lossless vector-like PNGs.
   - For UI banners, KPI cards, and data tables, use Windows Snipping Tool (`Win + Shift + S`) to capture clean rectangular crops.
5. Save screenshots into an `images/` or `figures/` folder with file names matching the Evidence IDs (`fig_01_dashboard_kpis.png`, `fig_02_risk_distribution.png`, etc.).
