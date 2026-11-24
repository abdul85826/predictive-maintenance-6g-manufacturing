# ACADEMIC INTERNSHIP REPORT CONTENT
## PREDICTIVE MAINTENANCE FOR 6G-ENABLED SMART MANUFACTURING USING MACHINE LEARNING

**Student Name:** Syed Abdul Rehman
**University Registration Number:** 230071601233
**Degree:** Bachelor of Technology in Computer Science and Engineering
**Department:** Department of Computer Science and Engineering
**Institution:** B.S. Abdur Rahman Crescent Institute of Science and Technology, Vandalur, Chennai – 600048, Tamil Nadu, India
**Internship Organization:** Unified Mentor Pvt. Ltd. (Project files also cite: *Unified Mentor / Thales Group Initiative*)
**Designation:** Machine Learning Intern
**Internship Duration:** 15 June 2026 – 15 July 2026 (One Month)
**Offer Letter Date:** 14 June 2026
**Certificate Issuance Date:** 15 July 2026

---

## PRELIMINARY PAGES

### 1. COVER PAGE
```
                          AN INTERNSHIP REPORT ON
       PREDICTIVE MAINTENANCE FOR 6G-ENABLED SMART MANUFACTURING
                        USING MACHINE LEARNING

                              Submitted by
                           SYED ABDUL REHMAN
                       (Reg. No: 230071601233)

                In partial fulfillment for the award of the degree
                                  of
                        BACHELOR OF TECHNOLOGY
                                  in
                   COMPUTER SCIENCE AND ENGINEERING

              DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING
       B.S. ABDUR RAHMAN CRESCENT INSTITUTE OF SCIENCE AND TECHNOLOGY
                 VANDALUR, CHENNAI – 600048, TAMIL NADU, INDIA
                               JULY 2026
```

### 2. BONAFIDE CERTIFICATE
```
       B.S. ABDUR RAHMAN CRESCENT INSTITUTE OF SCIENCE AND TECHNOLOGY
                 VANDALUR, CHENNAI – 600048, TAMIL NADU, INDIA

                           BONAFIDE CERTIFICATE

Certified that this internship report entitled "PREDICTIVE MAINTENANCE FOR 6G-ENABLED
SMART MANUFACTURING USING MACHINE LEARNING" is the bonafide work of SYED ABDUL REHMAN
(Reg. No: 230071601233) who carried out the internship work under our supervision at
Unified Mentor Pvt. Ltd. from 15th June 2026 to 15th July 2026.

INTERNAL SUPERVISOR / FACULTY GUIDE                 HEAD OF THE DEPARTMENT
[Faculty Name & Designation]                        Department of Computer Science & Engg.
Department of Computer Science & Engg.              B.S. Abdur Rahman Crescent Institute
B.S. Abdur Rahman Crescent Institute                of Science and Technology
of Science and Technology                           Chennai – 600048
Chennai – 600048
```

### 3. INTERNSHIP CERTIFICATE
*(Insert scanned copy of official Internship Completion Certificate issued by Unified Mentor Pvt. Ltd. on 15 July 2026 certifying the successful completion of the one-month Machine Learning Internship from 15 June 2026 to 15 July 2026).*

### 4. VIVA VOCE EXAMINATION CERTIFICATE
```
       B.S. ABDUR RAHMAN CRESCENT INSTITUTE OF SCIENCE AND TECHNOLOGY
                 VANDALUR, CHENNAI – 600048, TAMIL NADU, INDIA

                           VIVA-VOCE EXAMINATION

This is to certify that SYED ABDUL REHMAN (Reg. No: 230071601233) has been duly
examined in the Viva-Voce examination for his internship report on ........................

INTERNAL EXAMINER                                   EXTERNAL EXAMINER
```

### 5. ACKNOWLEDGEMENT
I express my deepest gratitude to the Almighty for bestowing upon me the strength and wisdom to complete this internship successfully.

I would like to place on record my sincere gratitude to the management and respected authorities of **B.S. Abdur Rahman Crescent Institute of Science and Technology, Vandalur, Chennai**, for providing the necessary infrastructure, encouragement, and academic atmosphere.

I express my heartfelt thanks to our respected **Head of the Department, Department of Computer Science and Engineering**, and my **Faculty Internship Coordinator**, for their continuous guidance, valuable suggestions, and constant support throughout the duration of my study and internship period.

I am immensely grateful to **Unified Mentor Pvt. Ltd.** for offering me the opportunity to work as a Machine Learning Intern from 15 June 2026 to 15 July 2026. I extend my sincere appreciation to my industry mentors and technical supervisors at Unified Mentor for their guidance, technical reviews, and constructive feedback during the architecture, modeling, and evaluation phases of the project.

Finally, I convey my heartfelt thanks to my parents, family members, and friends whose blessings, patience, and encouragement have always been a source of motivation.

---

### 6. ABSTRACT
In high-precision Industry 4.0 and emerging Industry 5.0 smart factories, unplanned machinery downtime represents a catastrophic source of economic loss, causing an estimated $50 Billion in annual industrial losses globally. Conventional reactive ("run-to-fail") maintenance incurs severe mechanical damage and production halts, while calendar-based preventive maintenance replaces healthy components prematurely. Moreover, traditional supervisory systems relying on static thresholds fail in flexible manufacturing environments where operational baselines fluctuate dynamically across modes (e.g., High-Load vs. Eco-Mode).

This internship project presents the design, implementation, and empirical verification of a comprehensive **Predictive Maintenance Platform for 6G-Enabled Smart Manufacturing**. The system utilizes multidimensional industrial telemetry (operating temperature, vibration frequency, electrical power consumption, quality defect rates, and operational error rates) combined with simulated 6G Ultra-Reliable Low-Latency Communication (URLLC) network QoS indicators (sub-millisecond latency and packet loss).

A high-fidelity synthetic data generator simulates 10 industrial machines over a 30-day operational lifecycle (10,800 telemetry records). A dynamic feature engineering pipeline extracts grouped rolling moving averages, thermal/vibration deviations, and vibration-to-power instability ratios to establish adaptive per-machine baselines. An unsupervised **Isolation Forest** ensemble algorithm ($n_{\text{estimators}} = 150$, $\text{contamination} = 0.08$) partitions multivariate feature space to detect anomalous degradation signatures without requiring historical failure labels. Raw decision scores are normalized via MinMax scaling into a continuous 0–100 Anomaly Score, mapped to discrete risk tiers (`Low`, `Medium`, `High`), and rendered via an interactive, glassmorphic Streamlit executive control dashboard featuring responsive Plotly analytics.

Experimental evaluation across the 10,800 telemetry records demonstrates that the system isolates 146 critical high-risk events (1.35% of total operations), with the highest failure intensity identified on asset `M_009` (78 high-risk spikes) and `M_007` (23 spikes). Machines in the high-risk tier exhibited average temperature increases of +14.0% (reaching 90.04°C) and vibration increases of +9.3% (reaching 65.86 Hz) accompanied by communication degradation. The solution demonstrates the viability of edge-deployed unsupervised AI models for condition-based industrial maintenance.

---

## CHAPTER 1 — COMPANY OVERVIEW

### 1.1 Company Overview
**Unified Mentor Pvt. Ltd.** is an innovative ed-tech and industry-readiness training organization based in India. The company focuses on bridging the gap between collegiate academia and modern industrial requirements by providing specialized internship programs, experiential project tracks, and technical mentorship in emerging computing domains.

### 1.2 Company Services and Activities
Unified Mentor provides structured, industry-aligned training across several technical verticals:
1. **Machine Learning & Artificial Intelligence:** End-to-end data science lifecycle, exploratory data analysis, predictive modeling, deep learning architectures, and production deployment.
2. **Data Science & Business Analytics:** Data cleaning, statistical inference, feature engineering, and executive dashboarding.
3. **Web & Full-Stack Development:** Modern responsive frontend frameworks, backend REST API development, and database architecture.
4. **Cloud Computing & DevOps:** Containerization, CI/CD pipeline implementation, and cloud-hosted application orchestration.

### 1.3 Company Contact and Online Presence
- **Official Website:** `https://www.unifiedmentor.com` (or current verified corporate domain).
- **Headquarters / Operations:** India.
- **Industry Sector:** EdTech, Professional Training, and Software Skills Development.

### 1.4 Internship Organization Details & Collaboration Context
During the internship period from **15 June 2026 to 15 July 2026**, the student served as a **Machine Learning Intern**. The project documentation in the repository also notes association with a **Thales Group Initiative / Framework**, reflecting project problem statements and industrial specifications aligned with high-reliability European aerospace, defense, and smart manufacturing standards.

---

## CHAPTER 2 — ROLE AND RESPONSIBILITIES

### 2.1 Role
- **Designation:** Machine Learning Intern
- **Department / Unit:** AI & Advanced Analytics Project Unit
- **Tenure:** 15 June 2026 – 15 July 2026 (One Month)

### 2.2 Responsibilities
1. **Literature Survey and Problem Formulation:** Investigate the limitations of conventional maintenance paradigms (reactive and preventive) and formulate the architectural requirements for 6G-enabled predictive maintenance.
2. **Data Modeling & Telemetry Synthesis:** Author a deterministic, physics-grounded synthetic telemetry generator in Python (`generate_dataset.py`) simulating 10 industrial machines, three distinct operational modes, progressive wear degradation, random shock anomalies, and 6G URLLC network communication metrics.
3. **Feature Engineering Pipeline:** Formulate and implement dynamic rolling window transformations, vibration-to-power instability ratios, and error escalation indices in `ml_pipeline.py` to decouple anomaly detection from static operating thresholds.
4. **Unsupervised Model Development:** Parameterize and train an Isolation Forest ensemble model using Scikit-learn, optimize hyperparameter configurations (tree depth, estimators, contamination), and implement MinMax score scaling.
5. **Dashboard Architecture & Visualization:** Build an interactive, glassmorphic Streamlit web control tower (`app.py`) integrating Plotly Express visualizations, dynamic machine filtering, risk distribution analysis, and priority maintenance alert queues.
6. **Technical Documentation & Verification:** Document the pipeline architecture, prepare executive summaries and research paper drafts, and verify pipeline reproducibility.

### 2.3 Skills Acquired
- **Applied Machine Learning:** Unsupervised anomaly detection, Isolation Forest theory, binary search tree path partitioning, decision function calibration, and hyperparameter tuning.
- **Industrial Data Science:** Time-series feature engineering, rolling window moving averages, multivariate anomaly correlation, and sensor telemetry synthesis.
- **Software Engineering & Python:** Object-oriented and functional Python, Pandas dataframe transformations, NumPy numerical vectorization, Joblib serialization, and virtual environment dependency management.
- **Interactive Web Development:** Streamlit component design, custom CSS injection (glassmorphic dark UI), and Plotly interactive data visualizations.
- **DevOps & Scientific Discipline:** Git version control, reproducibility auditing, and formal academic technical writing.

---

## CHAPTER 3 — PROJECT OVERVIEW

### 3.1 Introduction and Problem Statement
In automated industrial manufacturing facilities, unexpected equipment breakdown leads to crippling financial loss, safety hazards, and customer delivery penalties. Conventional maintenance approaches are fundamentally flawed:
- **Run-to-fail (Reactive)** leads to catastrophic asset damage and massive production halts.
- **Calendar-based (Preventive)** results in premature replacement of functional parts and substantial operational expenditure.
- **Static SCADA Thresholds** fail to accommodate multi-regime operations (High-Load, Normal, Eco-Mode), generating high false-alarm rates.

The problem addressed in this project is: *How can industrial smart factories leverage low-latency 6G sensor telemetry and unsupervised machine learning to detect incipient mechanical degradation and multivariate anomalies hours before physical breakdown occurs, without relying on static thresholds or scarce historical failure labels?*

### 3.2 Objectives
- **Primary:** To build and validate an unsupervised predictive maintenance platform that detects machine degradation using multivariate sensor telemetry and 6G communication indicators.
- **Secondary:**
  1. Synthesize 10,800 records of multi-machine industrial telemetry with coupled mechanical and network degradation.
  2. Implement rolling baseline feature engineering to adapt dynamically to operating regimes.
  3. Deploy an Isolation Forest model with 150 estimators to generate normalized 0–100 Anomaly Scores.
  4. Design a glassmorphic Streamlit dashboard providing real-time fleet risk visibility and automated maintenance recommendations.

### 3.3 Scope
- **In Scope:** Synthetic telemetry generation for 10 machines, 11-feature ML pipeline, Isolation Forest training, MinMax normalization, 3-tier risk mapping, and interactive Streamlit UI.
- **Out of Scope (Future Work):** Direct physical 6G hardware deployment, physical CNC machine sensor wiring, write-back PLC control over Modbus/OPC-UA, and supervised Remaining Useful Life (RUL) regression.

### 3.4 Key Features
- **Deterministic 6G Telemetry Simulator:** Reproducible synthesis of temperature, vibration, power, speed, latency, and packet loss.
- **Dynamic Contextual Baselines:** 10-period rolling moving averages that prevent false positives during high-load regimes.
- **Unsupervised Anomaly Isolation:** High-dimensional partitioning via 150 isolation trees requiring zero labeled failure instances.
- **Continuous 0–100 Anomaly Scoring:** MinMax-calibrated health indices providing granular machine degradation tracking.
- **Glassmorphic Executive Dashboard:** Interactive web UI with fleet overview, time-series anomaly drift, bivariate sensor scatters, and priority alert queues.

### 3.5 Technology Stack

```
+-----------------------------------------------------------------------------------------+
| PROGRAMMING LANGUAGE: Python 3.14.7                                                     |
+-----------------------------------------------------------------------------------------+
| DATA PROCESSING & ARRAYS: Pandas 3.1.1, NumPy 2.5.5                                     |
+-----------------------------------------------------------------------------------------+
| MACHINE LEARNING & ARTIFACTS: Scikit-Learn 1.9.1 (IsolationForest, MinMaxScaler),       |
|                                Joblib 1.6.0                                             |
+-----------------------------------------------------------------------------------------+
| PRESENTATION & VISUALIZATION: Streamlit 1.67.0, Plotly 7.1.0                            |
+-----------------------------------------------------------------------------------------+
| VERSION CONTROL: Git & GitHub Remote Repository                                         |
+-----------------------------------------------------------------------------------------+
```

### 3.6 System Architecture Summary
The system operates as a modular, three-tier architecture:
1. **Data Ingestion Tier:** `generate_dataset.py` creates `dataset.csv`.
2. **Analytics & ML Tier:** `ml_pipeline.py` processes features, trains the model, and outputs `processed_dataset.csv` and `model_artifacts.joblib`.
3. **Presentation Tier:** `app.py` loads processed data and model artifacts, rendering an interactive executive control dashboard.

---

## CHAPTER 4 — THEORETICAL ANALYSIS

### 4.1 Predictive Maintenance Fundamentals
Predictive Maintenance (PdM) evaluates in-service equipment condition to determine when maintenance should be performed. PdM follows the **P-F Curve** (Potential Failure to Functional Failure), which illustrates that equipment exhibits detectable physical anomalies (ultrasonic noise, vibration shifts, thermal rise) long before functional breakdown occurs.

### 4.2 Industrial Internet of Things (IIoT) & Sensor Telemetry
Smart manufacturing assets are instrumented with multi-modal sensors:
- **Piezoelectric Accelerometers:** Capture high-frequency structural vibration (Hz) indicating bearing pitting and rotor misalignment.
- **Infrared Thermocouples / RTDs:** Monitor core operating temperatures (°C) reflecting friction and coolant failures.
- **Current Transducers & Power Meters:** Track electrical power draw (kW), identifying motor strain and mechanical overload.

### 4.3 6G Wireless Telemetry in Smart Manufacturing
While 5G networks introduced enhanced mobile broadband (eMBB) and preliminary URLLC, **6G networks** represent a generational paradigm shift:
- **Sub-millisecond Latency (< 0.1 – 1 ms):** Guarantees zero-lag transmission of continuous high-frequency vibration streams.
- **Massive Connectivity:** Up to $10^7 \text{ devices/km}^2$, enabling pervasive sensing across all machine subcomponents.
- **Edge Computing Integration:** Direct integration of AI inference engines at local 6G cell towers and edge nodes.

### 4.4 Machine Learning for Anomaly Detection
Supervised ML methods require extensive labeled positive and negative failure data, which is virtually nonexistent in well-maintained factories. Anomaly detection is therefore fundamentally an **unsupervised learning problem**, where models learn the geometric boundary of normal operational behavior and flag instances that deviate from this manifold.

### 4.5 Isolation Forest Mathematical Architecture
The Isolation Forest algorithm isolates anomalies rather than profiling normal data points.

Given a dataset $X = \{x_1, \dots, x_n\}$ of $n$ instances in a $d$-dimensional space, an **Isolation Tree (iTree)** is a proper binary tree where each node either has zero children (leaf) or exactly two children.

1. **Random Recursive Partitioning:** At each node, a feature $q$ is chosen randomly, and a split point $p$ is chosen uniformly between $\min(X_{\cdot, q})$ and $\max(X_{\cdot, q})$. The sample is split into two subsets: $X_{\text{left}} = \{x \in X \mid x_q < p\}$ and $X_{\text{right}} = \{x \in X \mid x_q \ge p\}$.
2. **Path Length $h(x)$:** The number of edges traversed from the root node to a terminating leaf node.
3. **Average Path Length $c(n)$:** The average path length of an unsuccessful search in a Binary Search Tree:
   $$c(n) = 2 \ln(n - 1) + 0.5772156649 - \frac{2(n - 1)}{n}$$
4. **Ensemble Anomaly Score $s(x, n)$:**
   $$s(x, n) = 2^{-\frac{\mathbb{E}(h(x))}{c(n)}}$$
   - When $\mathbb{E}(h(x)) \to 0 \implies s \to 1$ (Definite Anomaly).
   - When $\mathbb{E}(h(x)) \to n - 1 \implies s \to 0$ (Normal Instance).

---

## CHAPTER 5 — DESIGN AND IMPLEMENTATION

### 5.1 Design Considerations
- **Non-Invasive Monitoring:** Must analyze telemetry passively without interrupting machine production cycles.
- **Regime Independence:** Must eliminate false alarms during legitimate high-load operations.
- **Lightweight Inference:** Low computational complexity suitable for edge deployment.

### 5.2 Synthetic Data Generation Implementation (`generate_dataset.py`)
```python
# Code excerpt from generate_dataset.py (lines 10-26, 47-65)
np.random.seed(42)
machines = [f"M_{i:03d}" for i in range(1, num_machines + 1)]
operation_modes = ["Normal", "High-Load", "Eco-Mode"]

for m in machines:
    machine_baselines[m] = {
        "base_temp": np.random.uniform(55, 68),
        "base_vib": np.random.uniform(40, 55),
        "base_power": np.random.uniform(12, 18),
        "health_decay_start": np.random.randint(15, 25),
        "degradation_rate": np.random.uniform(0.05, 0.25)
    }
```
The script implements mode-specific modifiers (+10°C, +4 kW, +250 speed units for High-Load), linear wear degradation past `health_decay_start`, a 4% random shock multiplier, Gamma-distributed 6G latency, and coupled error/defect rates.

### 5.3 Feature Engineering Pipeline (`ml_pipeline.py`)
```python
# Code excerpt from ml_pipeline.py (lines 12-22)
df = df.sort_values(by=["Machine_ID", "Date", "Time"]).reset_index(drop=True)

df["temp_rolling_avg"] = df.groupby("Machine_ID")["Temperature_C"].transform(
    lambda x: x.rolling(window=10, min_periods=1).mean()
)
df["temp_deviation"] = df["Temperature_C"] - df["temp_rolling_avg"]

df["vib_rolling_avg"] = df.groupby("Machine_ID")["Vibration_Hz"].transform(
    lambda x: x.rolling(window=10, min_periods=1).mean()
)
df["vib_deviation"] = df["Vibration_Hz"] - df["vib_rolling_avg"]

df["vib_power_ratio"] = df["Vibration_Hz"] / (df["Power_Consumption_kW"] + 1e-5)
df["error_escalation"] = df["Error_Rate_%"] / (df["Predictive_Maintenance_Score"] + 1e-5) * 100
```

### 5.4 Model Training & Score Calibration (`ml_pipeline.py`)
```python
# Code excerpt from ml_pipeline.py (lines 33-40)
iso_forest = IsolationForest(n_estimators=150, contamination=0.08, random_state=42)
iso_forest.fit(X)

# Raw decision function score inversion (higher = more anomalous)
raw_scores = -iso_forest.decision_function(X)

scaler = MinMaxScaler(feature_range=(0, 100))
df["Anomaly_Score"] = np.round(scaler.fit_transform(raw_scores.reshape(-1, 1)).flatten(), 2)
```

### 5.5 Interactive Dashboard Architecture (`app.py`)
`app.py` configures a glassmorphic dark theme and structures the user journey into 4 tabs:
1. **Overview & Risk Distribution:** Interactive donut chart of risk proportions (`px.pie`) and machine breakdown bar chart (`px.bar`).
2. **Anomaly & Sensor Trends:** Time-series line chart tracking Anomaly Score drift with horizontal alert thresholds (at 35 and 65), alongside multivariate scatter plots.
3. **Maintenance Alert Panel:** Filterable priority queue displaying high-risk telemetry events, estimated lead-time windows, and prescriptive actions.
4. **Historical Risk Escalation:** Density heatmap (`px.density_heatmap`) illustrating the progressive degradation of machines over the 30-day monitoring window.

---

## CHAPTER 6 — RESULTS AND ANALYSIS

### 6.1 Functional Verification Results

| Test ID | Module Tested | Input Data | Expected Result | Verified Result | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| **TC-01** | Data Generation | Seed = 42, 10 machines, 30 days | 10,800 rows, 14 cols, zero nulls | 10,800 rows, 14 cols, zero nulls, exact MD5 | **PASSED** |
| **TC-02** | Time-Series Sort | Raw `dataset.csv` | Ordered by Machine, Date, Time | Correct sequential order | **PASSED** |
| **TC-03** | Rolling Features | Window = 10 on Temp & Vib | Rolling averages and deviations | Computed grouped by Machine_ID | **PASSED** |
| **TC-04** | Model Fit | 11 feature columns | IsolationForest fit without error | 150 trees trained, seed=42 | **PASSED** |
| **TC-05** | Anomaly Scaling | Negative decision function | Normalized scores in [0.0, 100.0] | Scores in [0.0, 100.0], Mean = 22.67 | **PASSED** |
| **TC-06** | Risk Mapping | Thresholds at 35 and 65 | Low, Medium, High categories | 9023 Low, 1631 Medium, 146 High | **PASSED** |
| **TC-07** | Artifact Persistence | Model, scaler, feature cols | Serialized `model_artifacts.joblib` | Saved file (1.82 MB), loaded cleanly | **PASSED** |
| **TC-08** | Streamlit UI | `processed_dataset.csv` | Full dashboard compilation | Rendered all 4 tabs, zero errors | **PASSED** |

### 6.2 Empirical Anomaly and Risk Breakdown

```
========================================================================================
FLEET RISK CLASSIFICATION SUMMARY (TOTAL POPULATION: 10,800 RECORDS)
========================================================================================
Tier         Score Range      Records     Percentage    Key Operational Profile
----------------------------------------------------------------------------------------
Low Risk     0.0 - 34.99       9,023        83.55%      Temp: 63.96°C, Vib: 46.81Hz, Latency: 1.48ms
Medium Risk 35.0 - 64.99       1,631        15.10%      Temp: 66.50°C, Vib: 48.73Hz, Latency: 2.36ms
High Risk   65.0 - 100.0         146         1.35%      Temp: 72.89°C, Vib: 51.17Hz, Latency: 3.23ms
========================================================================================
```

### 6.3 Machine-Specific Degradation Findings
Machine `M_009` suffered the most acute degradation, accumulating **78 high-risk spikes** (53.4% of all fleet-wide critical events) with an average temperature of 78.82°C and peak temperatures exceeding 90°C. Machine `M_007` registered **23 high-risk events** driven by excessive vibration (averaging 59.93 Hz). Conversely, machines `M_001` and `M_010` remained highly stable, recording only 1 high-risk spike each over the entire month.

### 6.4 Challenges Faced
1. **Unsupervised Metric Ambiguity:** Absence of labeled failure truth required relying on domain physics and feature correlation rather than standard confusion matrices.
2. **Dynamic Baseline Formulation:** Standard global moving averages failed across mode switches; resolving this required grouping by `Machine_ID` and computing rolling statistics per individual asset.
3. **Library Version Compatibility:** Managing unpickling warnings between Scikit-learn 1.9.0 and 1.9.1 required careful environment configuration.

---

## CHAPTER 7 — CONCLUSION
The one-month Machine Learning internship at **Unified Mentor Pvt. Ltd.** provided an invaluable opportunity to investigate and solve real-world industrial challenges in smart manufacturing. The developed **Predictive Maintenance Platform for 6G-Enabled Smart Manufacturing** successfully integrates synthetic telemetry generation, dynamic rolling feature engineering, and unsupervised Isolation Forest anomaly detection into a responsive, glassmorphic Streamlit executive control tower.

By modeling multivariate interactions across thermal, mechanical, electrical, and 6G communication telemetry, the system successfully overcomes the fundamental limitations of static thresholding, providing early detection of incipient machine degradation. The project provided rich practical experience in applied data science, algorithm implementation, web visualization, and engineering documentation.

---

## CHAPTER 8 — REFERENCES
1. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). *Isolation Forest*. In 2008 Eighth IEEE International Conference on Data Mining (pp. 413-422). IEEE. DOI: `10.1109/ICDM.2008.17`.
2. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825-2830.
3. Saad, W., Bennis, M., & Chen, M. (2020). *A Vision of 6G Wireless Systems: Applications, Trends, Technologies, and Open Research Problems*. IEEE Network, 34(3), 134-142. DOI: `10.1109/MNET.001.1900287`.
4. Mobley, R. K. (2002). *An Introduction to Predictive Maintenance*. Butterworth-Heinemann. ISBN: 978-0750675314.
5. Zhang, C., Patras, P., & Haddadi, H. (2019). *Deep Learning in Mobile and Wireless Networking: A Survey*. IEEE Communications Surveys & Tutorials, 21(3), 2224-2287. DOI: `10.1109/COMST.2019.2904897`.
6. Streamlit Documentation. (2026). *Streamlit: An open-source app framework for Machine Learning and Data Science*. `https://docs.streamlit.io/`.
7. Plotly Technologies Inc. (2026). *Collaborative Data Science*. `https://plotly.com/python/`.

---

## APPENDIX

### A1. REPOSITORY STRUCTURE
```
predictive-maintenance-6g-manufacturing/
│
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── dataset.csv                  # 10,800 rows x 14 columns
├── processed_dataset.csv        # 10,800 rows x 23 columns
├── model_artifacts.joblib       # Serialized Isolation Forest & Scaler
│
├── generate_dataset.py          # Synthetic telemetry data generation
├── ml_pipeline.py               # Feature engineering & ML model training
├── app.py                       # Streamlit web dashboard
│
├── EXECUTIVE_SUMMARY.md         # Industry executive briefing
└── RESEARCH_PAPER.md            # Academic research paper draft
```

### A2. EXECUTION INSTRUCTIONS
```powershell
# Activate Environment
.\venv\Scripts\Activate.ps1

# Run Pipeline
python generate_dataset.py
python ml_pipeline.py

# Launch UI
streamlit run app.py
```
