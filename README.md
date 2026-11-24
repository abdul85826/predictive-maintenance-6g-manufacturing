<div align="center">
  <img src="assets/branding/project_logo.svg" alt="Predictive Intelligence Logo" width="460" />

  # Predictive Maintenance for 6G-Enabled Smart Manufacturing Using Machine Learning

  **An End-to-End Unsupervised Condition Monitoring & Anomaly Detection Platform**

  [![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
  [![Framework](https://img.shields.io/badge/Algorithm-Isolation%20Forest-forestgreen.svg)](https://scikit-learn.org/)
  [![UI](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
  [![Tests](https://img.shields.io/badge/Tests-9%20Passing-brightgreen.svg)](#testing--verification)

</div>

---

## 📌 Project Overview

In high-precision Industry 4.0 and emerging Industry 5.0 smart manufacturing environments, unplanned machinery downtime causes an estimated **$50 Billion in annual industrial losses globally**. Conventional reactive maintenance repairs equipment after catastrophic breakdown, while calendar-based preventive maintenance replaces functional components prematurely. Furthermore, traditional static threshold alerts (e.g., *Alert if Temp > 80°C*) fail in multi-regime factories where parameters fluctuate legitimately between High-Load and Eco-Mode regimes.

This repository implements an end-to-end **Predictive Maintenance and Anomaly Detection Platform** powered by:
1. **Multi-Modal Industrial Telemetry:** Core operating temperature, structural vibration, electrical power draw, operational speed, quality defect rate, and machine error rates.
2. **Simulated 6G URLLC Telemetry:** Sub-millisecond communication latency and telemetry packet loss.
3. **Dynamic Grouped Rolling Feature Engineering:** 10-period moving averages and deviation metrics per machine that eliminate false alarms during high-load regimes.
4. **Unsupervised Machine Learning (Isolation Forest):** 150 Isolation Trees isolating multi-sensor anomalies without requiring labeled historical failure datasets.
5. **Natural Industrial Executive Control Tower:** Responsive Streamlit dashboard designed according to a clean, ergonomic industrial palette (`#F5F6F2`, `#176B62`, `#39845A`, `#C44F4F`).

---

## 🎓 Academic & Internship Information

- **Author / Student:** Syed Abdul Rehman
- **University Register Number:** 230071601233
- **Degree:** Bachelor of Technology (B.Tech) in Computer Science and Engineering
- **Institution:** B.S. Abdur Rahman Crescent Institute of Science and Technology, Vandalur, Chennai, India
- **Internship Organization:** Unified Mentor Pvt. Ltd. (*Project problem context: Unified Mentor / Thales Group Initiative*)
- **Position:** Machine Learning Intern
- **Duration:** 15 June 2026 – 15 July 2026 (One Month)

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          TIER 1: DATA INGESTION                                 │
│  generate_dataset.py ──> Deterministic Synthesis (Seed 42) ──> dataset.csv      │
│  (10 Machines: M_001 to M_010 • 30 Days • 10,800 Telemetry Records)             │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        TIER 2: ML ANALYTICS ENGINE                              │
│  ml_pipeline.py                                                                 │
│  ├── Dynamic Rolling Moving Averages (Window = 10 grouped by Machine_ID)        │
│  ├── Deviation & Physical Ratios (vib_power_ratio, error_escalation)            │
│  ├── IsolationForest(n_estimators=150, contamination=0.08, random_state=42)     │
│  ├── MinMaxScaler(0, 100) Anomaly Score Normalization                           │
│  └── Serializes: processed_dataset.csv (2.29 MB) & model_artifacts.joblib       │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                   TIER 3: NATURAL INDUSTRIAL CONTROL TOWER                      │
│  app.py (Streamlit Web Application)                                             │
│  ├── Original Local SVG Vector Branding (assets/branding/)                      │
│  ├── Fleet Filter Engine: Machine Selector, Date Range, Modes, Risk Cutoff      │
│  ├── Top KPI Ribbon: Active Records, Avg Score (22.7), Critical Spikes (146)    │
│  ├── Plotly Charts: Risk Donut, Machine Breakdown, Anomaly Timeline, Scatters   │
│  ├── Priority Maintenance Alert Queue with Prescriptive Actions & CSV Export    │
│  └── Fleet-Wide 30-Day Temporal Degradation Heatmap                             │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Mathematical Formulation

### 1. Isolation Forest Anomaly Detection
The Isolation Forest isolates anomalies by randomly partitioning feature space. Because anomalous points require fewer partitions to isolate in binary trees, their path length $h(x)$ is significantly shorter than normal operational clusters.

The anomaly score $s(x, n)$ for an instance $x$ across an ensemble of $t$ trees is defined as:

$$s(x, n) = 2^{-\frac{\mathbb{E}(h(x))}{c(n)}}$$

Where $c(n)$ is the average path length of unsuccessful searches in a Binary Search Tree:

$$c(n) = 2 \ln(n - 1) + 0.5772156649 - \frac{2(n - 1)}{n}$$

### 2. MinMax Anomaly Score Normalization
Raw decision function outputs are inverted and scaled to an intuitive continuous $[0, 100]$ index:

$$\text{Anomaly\_Score} = \text{round}\left( \frac{\text{raw} - \min(\text{raw})}{\max(\text{raw}) - \min(\text{raw})} \times 100, 2 \right)$$

### 3. Calibrated Risk Tiers
- **Low Risk (Score < 35.0):** Normal operating condition. No intervention required (83.55% of operations).
- **Medium Risk (35.0 $\le$ Score < 65.0):** Incipient wear or transient shock. Schedule vibration recalibration and lubrication within 12–24 hours (15.10% of operations).
- **High Risk (Score $\ge$ 65.0):** Critical mechanical degradation or thermal runaway. Immediate inspection and speed down-throttling required within 4–12 hours (1.35% / 146 instances).

---

## 📊 Empirical Findings & Fleet Diagnostics

| Risk Tier | Anomaly Score Range | Telemetry Instances | Percentage | Mean Temp (°C) | Mean Vib (Hz) | 6G Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Low Risk** | 0.00 – 34.99 | 9,023 | **83.55%** | 63.96 °C | 46.81 Hz | 1.48 ms |
| **Medium Risk** | 35.00 – 64.99 | 1,631 | **15.10%** | 66.50 °C | 48.73 Hz | 2.36 ms |
| **High Risk** | 65.00 – 100.00 | 146 | **1.35%** | **72.89 °C** (Max: 90.04) | **51.17 Hz** (Max: 65.86) | **3.23 ms** (Max: 10.74) |

### Machine-Level Degradation Breakdown
- **Machine `M_009`:** Suffered the most acute mechanical failure, generating **78 critical high-risk spikes** (53.4% of all fleet-wide critical events) with temperatures reaching 90.04°C.
- **Machine `M_007`:** Second most degraded asset with **23 high-risk spikes** (15.8%) characterized by structural vibration spikes reaching 65.86 Hz.
- **Machines `M_001` & `M_010`:** Maintained optimal baseline health, registering only 1 high-risk transient event each across the entire 30-day lifecycle.

---

## 🚀 Quickstart & Installation

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14
- Git

### 2. Environment Setup
```powershell
# Clone the repository
git clone https://github.com/abdul85826/predictive-maintenance-6g-manufacturing.git
cd predictive-maintenance-6g-manufacturing

# Activate Virtual Environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Install Dependencies
pip install -r requirements.txt
```

### 3. Generate Telemetry & Run ML Pipeline
```powershell
# 1. Synthesize 10,800 telemetry records (10 machines, 30 days)
python generate_dataset.py

# 2. Execute feature engineering & train Isolation Forest model
python ml_pipeline.py
```

### 4. Launch the Interactive Dashboard
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501` to view the Predictive Intelligence console.

---

## 🧪 Testing & Verification

Execute the automated test suite verifying environment dependencies, SVG branding assets, dataset schemas, zero missing/duplicate records, model persistence, and sample inference:

```powershell
python -m unittest tests/test_system.py
```

**Test Output:**
```
.........
----------------------------------------------------------------------
Ran 9 tests in 11.150s

OK
```

---

## 📁 Repository Directory Structure

```
predictive-maintenance-6g-manufacturing/
│
├── .gitignore                      # Git exclusion rules
├── LICENSE                         # MIT License
├── README.md                       # Comprehensive technical presentation
├── requirements.txt                # Dependency specifications
│
├── dataset.csv                     # Raw telemetry (10,800 rows x 14 cols)
├── processed_dataset.csv           # Feature-engineered telemetry with Anomaly Scores (10,800 x 23 cols)
├── model_artifacts.joblib          # Serialized Isolation Forest, Scaler & Feature Columns
│
├── generate_dataset.py             # Deterministic synthetic data generator (seed 42)
├── ml_pipeline.py                  # Rolling feature extraction & Isolation Forest training
├── app.py                          # Streamlit Natural Industrial control tower
│
├── assets/
│   └── branding/
│       ├── project_logo.svg        # Horizontal vector brand logo (light surface)
│       ├── project_logo_dark.svg   # Horizontal vector brand logo (dark surface)
│       ├── project_symbol.svg      # Compact precision gear & signal symbol
│       └── favicon.svg             # Application browser favicon
│
├── tests/
│   └── test_system.py              # Automated test harness (9 unit & integration tests)
│
├── PROJECT_MASTER_DOCUMENTATION.md # Exhaustive technical reference & university thesis preparation
├── TECHNICAL_AUDIT.md              # Senior engineer audit & discrepancy register
├── INTERNSHIP_REPORT_CONTENT.md    # 35-page structured university report draft
├── ARCHITECTURE_AND_DIAGRAMS.md    # 8 complete Mermaid architecture diagrams & flowcharts
├── SCREENSHOT_AND_EVIDENCE_INDEX.md# Master index of figures, tables, and visual evidence
└── STUDENT_VERIFICATION_QUESTIONS.md# Institutional checklist for student confirmation
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
Copyright (c) 2026 Syed Abdul Rehman.
