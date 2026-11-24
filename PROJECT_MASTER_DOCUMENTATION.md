# PROJECT MASTER DOCUMENTATION: PREDICTIVE MAINTENANCE FOR 6G-ENABLED SMART MANUFACTURING USING MACHINE LEARNING

**Student Name:** Syed Abdul Rehman
**Register / Student ID:** 230071601233 (*Derived from Git Author Email: `230071601233@crescent.education`*)
**Degree:** Bachelor of Technology (B.Tech)
**Department:** Computer Science and Engineering (CSE)
**Institution:** B.S. Abdur Rahman Crescent Institute of Science and Technology, Vandalur, Chennai, Tamil Nadu, India
**Internship Organization:** Unified Mentor Pvt. Ltd. (Project files also reference: *Unified Mentor / Thales Group Initiative*)
**Designation:** Machine Learning Intern
**Internship Period:** 15 June 2026 – 15 July 2026 (Duration: 1 Month)
**Offer Letter Date:** 14 June 2026
**Certificate Issued Date:** 15 July 2026
**Project Repository:** `https://github.com/abdul85826/predictive-maintenance-6g-manufacturing`
**Repository Branch:** `main` (Commit: `0768e63`)
**Date of Technical Audit:** November 2026

---

## 1. PROJECT TITLE AND TERMINOLOGY VERIFICATION

### 1.1 Stated vs. Repository Titles
- **User-Supplied Capstone Title:**
  `PREDICTIVE MAINTENANCE FOR 6G-ENABLED SMART MANUFACTURING USING MACHINE LEARNING`
- **Repository Research Paper Title (`RESEARCH_PAPER.md`, Line 1):**
  `Predictive Maintenance and Anomaly Detection in 6G-Integrated Smart Manufacturing Systems`
- **Executive Summary Document Title (`EXECUTIVE_SUMMARY.md`, Line 1):**
  `Executive Summary: 6G Smart Manufacturing Predictive Maintenance`
- **Application Dashboard Page Title (`app.py`, Line 11):**
  `6G Smart Manufacturing - Predictive Maintenance`
- **Application Header Banner (`app.py`, Lines 105–106):**
  `⚡ Predictive Maintenance & Anomaly Detection | 6G-Integrated Smart Manufacturing Intelligence Dashboard | Unified Mentor & Thales Group Framework`
- **GitHub Repository Slug:**
  `predictive-maintenance-6g-manufacturing`

### 1.2 Discrepancy Analysis
The core theme across all sources is identical: applying unsupervised machine learning (specifically Isolation Forest) to multidimensional industrial sensor telemetry and simulated 6G network QoS metrics to forecast machine failures. The repository alternately uses the terms **"6G-Integrated"** and **"6G-Enabled"**, and adds **"and Anomaly Detection"** to reflect the technical mechanism. For the university internship report, both titles represent the same implementation, with the formal project title being **"PREDICTIVE MAINTENANCE FOR 6G-ENABLED SMART MANUFACTURING USING MACHINE LEARNING"**.

---

## 2. PROBLEM IDENTIFICATION AND INDUSTRIAL MOTIVATION

### 2.1 The Industrial Challenge
In modern manufacturing (encompassing Computer Numerical Control (CNC) milling centers, automated robotic welding cells, injection molding systems, and high-speed rotary packaging equipment), unexpected machine breakdown represents one of the largest drains on industrial profitability.

Global manufacturing estimates indicate that unscheduled machine downtime costs heavy industries upwards of **$50 Billion annually** (`EXECUTIVE_SUMMARY.md`, Line 11). For a high-throughput manufacturing line, an idle production cell can result in losses between $20,000 and $250,000 per hour due to:
1. Direct production output halts and unfulfilled customer quotas.
2. Idle labor wages and emergency overtime pay for specialized repair technicians.
3. Secondary collateral mechanical damage caused by catastrophic part failure (e.g., a seized bearing damaging the spindle shaft and motor drive).
4. Raw material wastage (scrapped in-process parts during sudden halts).

### 2.2 Maintenance Paradigms: Comparative Analysis

| Maintenance Strategy | Operational Mechanism | Trigger Event | Advantages | Critical Limitations |
| :--- | :--- | :--- | :--- | :--- |
| **Reactive Maintenance** (*Run-to-Fail*) | Equipment is operated until a physical breakdown or functional failure occurs. | Catastrophic failure, motor seizure, smoke, part fracture. | Zero upfront cost; no monitoring sensors required. | Maximum catastrophic damage; unpredictable production halts; highest repair expense; safety hazards. |
| **Preventive Maintenance** (*Calendar / Time-Based*) | Equipment components are inspected, lubricated, or replaced at fixed calendar or operating-hour intervals. | Elapsed time (e.g., every 30 days) or cycle count (e.g., 5,000 hours). | Reduced catastrophic breakdowns compared to reactive strategy. | Premature replacement of healthy components (up to 30% waste); does not prevent random premature degradation; high labor overhead. |
| **Predictive Maintenance** (*Condition-Based / AI-Driven*) | Continuously monitors operational telemetry (temperature, vibration, power draw, communication latency) using ML algorithms to forecast failures before they occur. | Multivariate statistical anomalies and degradation patterns detected by AI. | Eliminates unscheduled downtime; maximizes asset remaining life; parts replaced strictly when wear signatures emerge; dynamic scheduling. | Requires sensor infrastructure, data pipelines, telemetry networks, and validated machine learning models. |

### 2.3 Why Static Thresholds Fail
Traditional supervisory control and data acquisition (SCADA) systems use fixed upper and lower specification limits (e.g., *Trigger alarm if Temperature > 80°C*). In modern flexible manufacturing, these rules trigger rampant false positives and dangerous false negatives:
- Under **High-Load Mode**, a temperature of 82°C with 22 kW power draw is within acceptable operating thermal equilibrium.
- Under **Eco-Mode** or idle state, a temperature of 74°C with 11 kW power draw is a severe indication of mechanical friction, inadequate lubrication, or electrical shorting, yet static rules fail to detect it.
- **Multivariate Context:** Machine degradation manifests across subtle combinations: a 2 Hz elevation in vibration coupled with a 1.5 kW increase in power draw and a 3°C rolling thermal drift indicates bearing spalling long before any single parameter breaches an emergency threshold.

### 2.4 The Role of 6G Telemetry and Low-Latency Communication
Next-generation **6G wireless networks** provide three transformative capabilities for Industrial Internet of Things (IIoT):
1. **Sub-millisecond Latency (< 1 ms):** Ensures that high-frequency vibration and acoustic emission data can be streamed to edge compute units without transmission queuing delays.
2. **Massive Machine-Type Communications (mMTC):** Supports sensor densities exceeding $10^6 \text{ to } 10^7 \text{ devices/km}^2$, enabling pervasive telemetry on every motor, bearing, valve, and robotic joint.
3. **Integrated Sensing and Communication (ISAC) & Edge AI:** Allows ML inference to execute at the local 6G edge node, enabling closed-loop feedback triggers (e.g., automatically instructing the Programmable Logic Controller (PLC) to down-throttle machine speed before thermal runaway occurs).

---

## 3. OBJECTIVES AND SCOPE

### 3.1 Primary Objective
To design, implement, and validate an end-to-end Machine Learning-powered Predictive Maintenance platform capable of detecting multivariate operational anomalies and degradation signatures across smart manufacturing machinery using high-frequency sensor telemetry and simulated 6G network performance indicators.

### 3.2 Secondary Objectives
1. **Synthetic Data Synthesis:** Model realistic physics-based industrial machine behavior, operational mode modulations (`Normal`, `High-Load`, `Eco-Mode`), progressive wear degradation, random shock anomalies, and 6G network latency profiles (`generate_dataset.py`).
2. **Dynamic Rolling Feature Engineering:** Implement per-machine rolling baselines (moving averages, deviation metrics, vibration-to-power instability ratios, and error escalation indices) to eliminate reliance on static thresholds (`ml_pipeline.py`).
3. **Unsupervised Anomaly Modeling:** Implement and parameterize an **Isolation Forest** ensemble model to compute continuous anomaly scores without requiring pre-labeled failure datasets.
4. **Actionable Risk Classification:** Transform raw decision function outputs into an intuitive 0–100 Anomaly Score mapped to discrete risk tiers (`Low`, `Medium`, `High`) with maintenance recommendations.
5. **Interactive Executive Control Tower:** Develop a production-grade Streamlit web dashboard featuring glassmorphic dark-theme aesthetics, Plotly analytical visualizations, machine-level filtering, threshold tuning, and priority alert queues (`app.py`).

### 3.3 Scope: Implemented vs. Theoretical / Out-of-Scope

```
+---------------------------------------------------+---------------------------------------------------+
|            IMPLEMENTED & VERIFIED                 |         THEORETICAL CONCEPTS / OUT-OF-SCOPE       |
+---------------------------------------------------+---------------------------------------------------+
| - Deterministic synthetic telemetry generator     | - Physical connection to live 6G radio hardware   |
|   (10 machines, 30 days, 10,800 records).         |   or base stations (simulated mathematically).    |
| - 14 raw telemetry features + 9 engineered cols.  | - Physical industrial vibration/thermal sensors.  |
| - Isolation Forest (150 trees, 8% contamination). | - Direct hardware PLC write-back over Modbus/OPC. |
| - MinMax scaled 0-100 Anomaly Scoring.            | - Supervised RUL (Remaining Useful Life) regression|
| - 3-tier maintenance risk mapping.                | - Streaming message queues (Kafka, MQTT broker).  |
| - Streamlit glassmorphic interactive web UI.      | - User authentication, RBAC, or persistent SQL DB.|
| - 4 Plotly interactive charts + alert table.      | - Real-time continuous online learning/retraining.|
+---------------------------------------------------+---------------------------------------------------+
```

---

## 4. DATASET INVESTIGATION AND SCHEMA ANALYSIS

### 4.1 Provenance and Generation Methodology
The dataset is synthetically generated via `generate_dataset.py` using NumPy and Pandas with a deterministic random seed (`np.random.seed(42)`). The script simulates 10 industrial machines (`M_001` through `M_010`) operating over 30 days (June 1, 2026 to June 30, 2026) sampled every 40 minutes (36 records per day per machine).

$$\text{Total Telemetry Rows} = 10 \text{ machines} \times 30 \text{ days} \times 36 \text{ records/day} = 10,800 \text{ rows}$$

### 4.2 Raw Telemetry Schema (`dataset.csv`)

| Column Name | Data Type | Null Count | Numerical Range [Min, Max] | Mean ± Std | Description |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `Date` | `object` (string) | 0 | `2026-06-01` to `2026-06-30` | — | Calendar date of telemetry record |
| `Time` | `object` (string) | 0 | `00:00:00` to `23:20:00` | — | Timestamp captured at 40-minute intervals |
| `Machine_ID` | `object` (string) | 0 | `M_001` to `M_010` | — | Unique machine asset tag (1,080 rows per machine) |
| `Operation_Mode` | `object` (string) | 0 | `Normal`, `High-Load`, `Eco-Mode` | — | Mode distribution: Normal (64.6%), High-Load (25.3%), Eco (10.1%) |
| `Temperature_C` | `float64` | 0 | [48.94, 90.04] °C | 64.47 ± 6.76 | Operating temperature of core assembly |
| `Vibration_Hz` | `float64` | 0 | [36.40, 65.86] Hz | 47.16 ± 5.37 | Mechanical vibration frequency |
| `Power_Consumption_kW` | `float64` | 0 | [9.36, 25.05] kW | 16.74 ± 2.56 | Total electrical power draw |
| `Network_Latency_ms` | `float64` | 0 | [0.02, 10.74] ms | 1.64 ± 1.15 | Simulated 6G URLLC communication latency |
| `Packet_Loss_%` | `float64` | 0 | [0.00, 0.529] % | 0.067 ± 0.084 | Simulated telemetry packet drop rate |
| `Quality_Control_Defect_Rate_%` | `float64` | 0 | [0.05, 1.67] % | 0.57 ± 0.22 | Downstream quality inspection defect rate |
| `Production_Speed_units_per_hr` | `int64` | 0 | [757, 1333] units/hr | 1042.57 ± 128.33 | Factory output rate |
| `Predictive_Maintenance_Score` | `float64` | 0 | [38.26, 100.00] | 92.37 ± 8.28 | Synthetic health ground-truth (100 = Perfect) |
| `Error_Rate_%` | `float64` | 0 | [0.10, 3.74] % | 0.68 ± 0.40 | Machine controller micro-error frequency |
| `Efficiency_Status` | `object` (string) | 0 | `High`, `Medium`, `Low` | — | Categorical status label: High (95.3%), Medium (4.1%), Low (0.6%) |

### 4.3 Data Hygiene and Statistical Integrity
- **Missing / Null Values:** Exactly 0 missing entries across all 14 columns.
- **Duplicate Records:** Exactly 0 duplicate rows.
- **Outlier Dynamics:** Outliers are intentional and physically coupled: when a 5% random shock occurs (`anomaly_spike` $\in [1.3, 1.8]$) or progressive degradation sets in, temperature, vibration, power draw, error rate, and network latency spike in synchrony.

---

## 5. FEATURE ENGINEERING AND DATA TRANSFORMATION

In `ml_pipeline.py` (lines 11–23), raw telemetry is transformed into dynamic contextual indicators to capture temporal drift and mechanical instability.

```
       dataset.csv
           │
           ▼
┌────────────────────────────────────────────────────────┐
│ 1. Time-Series Sorting by [Machine_ID, Date, Time]     │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 2. Per-Machine Grouped Rolling Moving Averages         │
│    - temp_rolling_avg = Rolling Mean (Window = 10)     │
│    - vib_rolling_avg  = Rolling Mean (Window = 10)     │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 3. Deviation & Ratio Feature Computations              │
│    - temp_deviation   = Temperature_C - Rolling Avg    │
│    - vib_deviation    = Vibration_Hz - Rolling Avg     │
│    - vib_power_ratio  = Vibration / (Power + 1e-5)     │
│    - error_escalation = (Error_Rate / Maint_Score)*100 │
│    - score_decay      = 100 - Maint_Score              │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
                 processed_dataset.csv
                 (10,800 rows x 23 columns)
```

### 5.1 Mathematical Definitions of Engineered Features

1. **Rolling Temperature Deviation (`temp_deviation`):**
   $$\bar{T}_{\text{rolling}}(t) = \frac{1}{K} \sum_{i=0}^{K-1} T(t-i), \quad K = 10$$
   $$\Delta T(t) = T(t) - \bar{T}_{\text{rolling}}(t)$$
   *Physical Meaning:* Measures sudden thermal shocks independent of whether the machine is in High-Load or Eco-Mode.

2. **Rolling Vibration Deviation (`vib_deviation`):**
   $$\bar{V}_{\text{rolling}}(t) = \frac{1}{K} \sum_{i=0}^{K-1} V(t-i), \quad K = 10$$
   $$\Delta V(t) = V(t) - \bar{V}_{\text{rolling}}(t)$$
   *Physical Meaning:* Captures instantaneous mechanical vibration transients that indicate bearing micro-fractures.

3. **Vibration-to-Power Instability Ratio (`vib_power_ratio`):**
   $$R_{\text{vib\_pwr}} = \frac{V(t)}{P(t) + 10^{-5}}$$
   *Physical Meaning:* In healthy machinery, vibration scales proportionally with power draw. A surge in vibration while power draw remains flat or decreases indicates mechanical imbalance or structural looseness.

4. **Error Escalation Index (`error_escalation`):**
   $$E_{\text{escalate}} = \frac{\text{Error\_Rate}(t)}{\text{Predictive\_Maintenance\_Score}(t) + 10^{-5}} \times 100$$
   *Physical Meaning:* Amplifies warning signals when controller errors occur simultaneously with deteriorating machine health.

5. **Score Decay (`score_decay`):**
   $$S_{\text{decay}} = 100.0 - \text{Predictive\_Maintenance\_Score}(t)$$

---

## 6. MACHINE LEARNING MODEL: ISOLATION FOREST

### 6.1 Algorithm Selection Justification
In industrial predictive maintenance, operational failure data is inherently scarce, highly imbalanced (< 2% anomalous), and expensive to obtain. Supervised algorithms (e.g., Random Forest Classifier, XGBoost) fail because negative failure examples are rarely available in sufficient volume or variety to train reliable classifiers.

**Isolation Forest** (Liu, Ting, and Zhou, 2008) is uniquely suited for industrial anomaly detection:
1. **Unsupervised Formulation:** Does not require labeled failure examples.
2. **Sub-linear Time Complexity:** Efficient $O(n \log n)$ construction and $O(t \log \psi)$ evaluation, enabling execution on resource-constrained 6G edge compute units.
3. **Partitioning Principle:** Instead of modeling normal cluster densities (like GMM or DBSCAN) or calculating distance matrices (like k-NN or LOF), Isolation Forest explicitly isolates anomalies by constructing random binary trees.

### 6.2 Mathematical Foundation
An Isolation Forest constructs an ensemble of $t$ Isolation Trees ($iTrees$) over subsamples $X' \subset X$ of size $\psi$.
At each node, an isolation tree randomly selects a feature $q$ and a split value $p$ uniformly at random between the minimum and maximum values of $q$ in the current node:

$$p \sim \text{Uniform}(\min(X_{\cdot, q}), \max(X_{\cdot, q}))$$

Because anomalies possess attribute values distinctly different from normal operating points, they require significantly fewer random splits to isolate into leaf nodes, resulting in a substantially shorter tree path length $h(x)$.

The average path length $c(n)$ of an unsuccessful search in a Binary Search Tree (BST) represents the normalization constant for sample size $n$:

$$c(n) = 2 \ln(n - 1) + 0.5772156649 \text{ (Euler-Mascheroni constant)} - \frac{2(n - 1)}{n}$$

The anomaly score $s(x, n)$ for an instance $x$ across an ensemble of $t$ isolation trees is defined as:

$$s(x, n) = 2^{-\frac{\mathbb{E}(h(x))}{c(n)}}$$

Where:
- $\mathbb{E}(h(x)) = \frac{1}{t} \sum_{i=1}^t h_i(x)$ is the expected path length across all $t$ trees.
- If $\mathbb{E}(h(x)) \to 0$, $s \to 1$ (High probability of being an anomaly).
- If $\mathbb{E}(h(x)) \to n - 1$, $s \to 0$ (Normal operational instance).
- If $\mathbb{E}(h(x)) \to c(n)$, $s \to 0.5$ (Instance exhibits no distinct anomaly signature).

### 6.3 Implementation and Hyperparameters
In `ml_pipeline.py` (lines 25–41), the model is configured with:

| Hyperparameter | Configured Value | Default in Scikit-Learn | Rationale / Justification |
| :--- | :---: | :---: | :--- |
| `n_estimators` | `150` | 100 | Enhanced ensemble stability and lower variance across random tree partitions. |
| `contamination` | `0.08` | "auto" | Expected proportion of anomalies in industrial telemetry (~8%). |
| `random_state` | `42` | None | Ensures bit-exact determinism and reproducibility of anomaly scores. |
| `max_samples` | `'auto'` ($\min(256, n)$) | 'auto' | Prevents masking and swamping effects in large datasets. |
| `bootstrap` | `False` | False | Sampling without replacement preserves tree partition independence. |

### 6.4 Selected Feature Vector (11 Features)
The feature matrix $X \in \mathbb{R}^{10800 \times 11}$ supplied to `iso_forest.fit(X)` consists of:
1. `Temperature_C`
2. `Vibration_Hz`
3. `Power_Consumption_kW`
4. `Network_Latency_ms`
5. `Packet_Loss_%`
6. `Quality_Control_Defect_Rate_%`
7. `Error_Rate_%`
8. `temp_deviation`
9. `vib_deviation`
10. `vib_power_ratio`
11. `error_escalation`

### 6.5 Anomaly Score Normalization and Risk Tier Mapping
Scikit-learn's `iso_forest.decision_function(X)` outputs the mean anomaly score where lower values indicate greater abnormality.
In `ml_pipeline.py` (lines 37–51):
1. The sign is inverted: `raw_scores = -iso_forest.decision_function(X)` so that higher values denote higher risk.
2. `MinMaxScaler(feature_range=(0, 100))` normalizes raw scores to a continuous scale:
   $$\text{Anomaly\_Score} = \text{round}\left( \frac{\text{raw} - \min(\text{raw})}{\max(\text{raw}) - \min(\text{raw})} \times 100, 2 \right)$$
3. Discrete Risk Classification:
   $$\text{Maintenance\_Risk\_Level} = \begin{cases} \text{High}, & \text{Anomaly\_Score} \ge 65 \\ \text{Medium}, & 35 \le \text{Anomaly\_Score} < 65 \\ \text{Low}, & \text{Anomaly\_Score} < 35 \end{cases}$$

### 6.6 Empirical Risk Distribution across the Telemetry Population

| Maintenance Risk Tier | Score Range | Record Count | Percentage of Dataset | Operational Status | Recommended Action in Dashboard |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **Low Risk** | [0.0, 34.99] | 9,023 | **83.55%** | Healthy Operation | Routine operation; no action required. |
| **Medium Risk** | [35.0, 64.99] | 1,631 | **15.10%** | Incipient Degradation | Schedule inspection & vibration recalibration within 12–24 hours. |
| **High Risk** | [65.0, 100.0] | 146 | **1.35%** | Critical Failure Risk | Immediate bearing lubrication, thermal check & speed reduction (< 6 hrs). |

---

## 7. HARDWARE DEGRADATION AND MACHINE BREAKDOWN DYNAMICS

Investigation of `processed_dataset.csv` reveals distinct machine-level failure profiles. Machine degradation was initialized at random intervals (`health_decay_start` $\in [15, 25]$) with variable decay rates:

```
            High-Risk Telemetry Count by Machine Asset
  M_009  [==================================================] 78 events (53.4%)
  M_007  [==============>                                   ] 23 events (15.8%)
  M_002  [======>                                           ] 10 events (6.8%)
  M_003  [=====>                                            ]  9 events (6.2%)
  M_008  [=====>                                            ]  9 events (6.2%)
  M_005  [=====>                                            ]  8 events (5.5%)
  M_004  [===>                                              ]  4 events (2.7%)
  M_006  [==>                                               ]  3 events (2.1%)
  M_001  [=>                                                ]  1 event  (0.7%)
  M_010  [=>                                                ]  1 event  (0.7%)
```

### 7.1 Cross-Tier Metric Comparison

| Telemetry Parameter | Low Risk (N=9,023) | Medium Risk (N=1,631) | High Risk (N=146) | Contrast (High vs Low) |
| :--- | :---: | :---: | :---: | :---: |
| **Average Temperature** | 63.96 °C | 66.50 °C | **72.89 °C** | **+8.93 °C (+14.0%)** |
| **Peak Temperature** | 82.49 °C | 83.04 °C | **90.04 °C** | **+7.55 °C** |
| **Average Vibration** | 46.81 Hz | 48.73 Hz | **51.17 Hz** | **+4.36 Hz (+9.3%)** |
| **Peak Vibration** | 61.28 Hz | 64.67 Hz | **65.86 Hz** | **+4.58 Hz** |
| **Average Power Draw** | 16.57 kW | 17.56 kW | **18.54 kW** | **+1.97 kW (+11.9%)** |
| **6G Latency** | 1.48 ms | 2.36 ms | **3.23 ms** | **+1.75 ms (+118.2%)** |
| **6G Packet Loss** | 0.051 % | 0.144 % | **0.232 %** | **+0.181 % (+354.9%)** |
| **Maintenance Health Score**| 94.10 / 100 | 84.97 / 100 | **68.59 / 100** | **-25.51 points (-27.1%)** |

This empirical contrast proves that the Isolation Forest model correctly isolates machines exhibiting combined thermal runaway, mechanical friction, and communication degradation.

---

## 8. SOFTWARE ARCHITECTURE AND APPLICATION DESIGN

### 8.1 Architectural Paradigm
The project adopts a **3-Tier Modular Pipeline Architecture**:
1. **Tier 1 — Data Synthesis & Ingestion:** `generate_dataset.py` creates structured physical and network telemetry.
2. **Tier 2 — Analytics & Machine Learning Engine:** `ml_pipeline.py` executes feature engineering, Isolation Forest anomaly scoring, MinMax normalization, and joblib model serialization.
3. **Tier 3 — Interactive Executive Dashboard:** `app.py` delivers a browser-based Streamlit interface with responsive Plotly visualizations and priority alert scheduling.

### 8.2 Component Interconnection
```
+-----------------------------------------------------------------------------------------+
|                                TIER 1: DATA INGESTION                                  |
|   generate_dataset.py ──> Generates 10,800 records ──> dataset.csv (958 KB)             |
+-----------------------------------------------------------------------------------------+
                                             │
                                             ▼
+-----------------------------------------------------------------------------------------+
|                               TIER 2: ML PIPELINE ENGINE                                |
|   ml_pipeline.py                                                                        |
|     ├── Rolling window feature extraction (Window = 10)                                 |
|     ├── IsolationForest(n_estimators=150, contamination=0.08)                           |
|     ├── MinMaxScaler(0, 100) Anomaly Score scaling                                      |
|     └── Exports:                                                                        |
|           ├── processed_dataset.csv (2.29 MB)                                           |
|           └── model_artifacts.joblib (1.82 MB: {model, scaler, feature_cols})           |
+-----------------------------------------------------------------------------------------+
                                             │
                                             ▼
+-----------------------------------------------------------------------------------------+
|                         TIER 3: PRESENTATION & CONTROL TOWER                            |
|   app.py (Streamlit Web Dashboard)                                                      |
|     ├── Glassmorphic Dark UI Theme (Custom CSS injection)                               |
|     ├── Sidebar Filters: Machine Selector, Operation Mode Multi-select, Risk Slider    |
|     ├── Top KPI Cards: Avg Score, Spikes, Early Warning (18.5 hrs), Prevention Index    |
|     └── 4 Dedicated Analytical Tabs:                                                    |
|           ├── Tab 1: Risk Distribution Pie Chart & Machine Breakdown Bar Chart          |
|           ├── Tab 2: Anomaly Timeline Line Chart, Temp vs Vib & Latency vs Loss Scatter|
|           ├── Tab 3: Interactive Maintenance Alert Queue & Action Recommendations       |
|           └── Tab 4: Temporal Heatmap of Machine Risk Intensity over Days               |
+-----------------------------------------------------------------------------------------+
```

---

## 9. TECHNOLOGY STACK AND DEPENDENCY SPECIFICATIONS

| Technology | Verified Version | Purpose in Repository | Implementation Location | Architecture Tier |
| :--- | :---: | :--- | :--- | :--- |
| **Python** | `3.14.7` | Core programming language environment | System runtime | All |
| **Pandas** | `3.1.1` | Data manipulation, rolling window statistics, CSV I/O | `generate_dataset.py`, `ml_pipeline.py`, `app.py` | Tiers 1, 2, 3 |
| **NumPy** | `2.5.5` | Array operations, random number synthesis, normal/gamma distributions | `generate_dataset.py`, `ml_pipeline.py`, `app.py` | Tiers 1, 2, 3 |
| **Scikit-learn** | `1.9.1` | Isolation Forest unsupervised algorithm, MinMaxScaler | `ml_pipeline.py` | Tier 2 |
| **Joblib** | `1.6.0` | Serialization and persistence of trained model and scaler objects | `ml_pipeline.py`, `model_artifacts.joblib` | Tier 2 |
| **Streamlit** | `1.67.0` | Web application framework, reactive state management, caching | `app.py` | Tier 3 |
| **Plotly** | `7.1.0` | Interactive GPU-accelerated charting (`px.pie`, `px.bar`, `px.line`, `px.scatter`, `px.density_heatmap`) | `app.py` | Tier 3 |
| **Git / GitHub**| `2.x` | Version control, branch tracking, code repository | `.git/`, GitHub remote | DevOps |

---

## 10. SYSTEM EXECUTION AND REPRODUCIBILITY VERIFICATION

### 10.1 Execution Environment Setup
```powershell
# 1. Clone repository
git clone https://github.com/abdul85826/predictive-maintenance-6g-manufacturing.git
cd predictive-maintenance-6g-manufacturing

# 2. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate synthetic dataset
python generate_dataset.py

# 5. Execute ML feature engineering and Isolation Forest training
python ml_pipeline.py

# 6. Launch interactive Streamlit control tower
streamlit run app.py
```

### 10.2 Reproducibility Audit Findings
- **Data Determinism:** Re-running `generate_dataset.py` with `np.random.seed(42)` produces exactly identical rows, identical column order, and an identical MD5 checksum (`eab57a157174890e6b361a8de556d20d`). Absolute difference across all 10,800 rows: **0.0**.
- **Model Determinism:** Re-running `ml_pipeline.py` with `random_state=42` produces identical Isolation Forest tree partitions, identical raw decision scores, identical MinMax scaling, and identical `Maintenance_Risk_Level` assignments. Absolute difference: **0.0**.
- **UI Compilation:** `app.py` compiles cleanly under Streamlit 1.67.0 without syntax or import errors.

---

## 11. TECHNICAL LIMITATIONS AND CODE QUALITY AUDIT

1. **Unsupervised Evaluation Gap:** Because the model is unsupervised, standard supervised classification metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC) are not computed during pipeline execution.
2. **Simulated Lead Time and Prevention KPIs:** In `ml_pipeline.py` (lines 62–63), `avg_lead_time_hrs` is generated using a random uniform draw (`np.random.uniform(14.5, 28.0)`), and `downtime_prevention_idx` is calculated using an arbitrary heuristic formula rather than empirical run-to-failure tracking. In `app.py` (line 137), `18.5 hrs` is hardcoded in the KPI card.
3. **Data Leakage in Temporal Modeling:** Rolling averages (`rolling(window=10)`) and Isolation Forest training are executed across the entire 30-day time series simultaneously without strict train/test temporal splitting. In a live production setting, inference must be performed using an expanding historical window.
4. **Standalone Streamlit Architecture:** `app.py` directly reads `processed_dataset.csv` from disk via `@st.cache_data`. It does not expose a REST API (e.g., FastAPI) or message broker (Kafka/MQTT) for streaming telemetry ingestion.
5. **Simulated 6G Network Layer:** The 6G network latency and packet loss values are generated mathematically from Gamma and Normal distributions; they are not measured from physical 6G hardware testbeds.

---

## 12. ACADEMIC REFERENCES

1. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). *Isolation Forest*. Eighth IEEE International Conference on Data Mining (ICDM '08), Pisa, Italy, pp. 413–422. DOI: `10.1109/ICDM.2008.17`.
2. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, pp. 2825–2830.
3. Saad, W., Bennis, M., & Chen, M. (2020). *A Vision of 6G Wireless Systems: Applications, Trends, Technologies, and Open Research Problems*. IEEE Network, 34(3), pp. 134–142. DOI: `10.1109/MNET.001.1900287`.
4. Mobley, R. K. (2002). *An Introduction to Predictive Maintenance*. 2nd Edition, Butterworth-Heinemann, Elsevier Science, ISBN: 978-0750675314.
5. Zhang, C., Patras, P., & Haddadi, H. (2019). *Deep Learning in Mobile and Wireless Networking: A Survey*. IEEE Communications Surveys & Tutorials, 21(3), pp. 2224–2287. DOI: `10.1109/COMST.2019.2904897`.
