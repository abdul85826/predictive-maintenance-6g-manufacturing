# TECHNICAL AUDIT REPORT: 6G SMART MANUFACTURING PREDICTIVE MAINTENANCE

**Project Name:** Predictive Maintenance for 6G-Enabled Smart Manufacturing
**Repository:** `https://github.com/abdul85826/predictive-maintenance-6g-manufacturing`
**Auditor:** Senior Software Architect, ML Engineer & Technical Auditor
**Date of Audit:** November 2026
**Audited Commit:** `0768e63` (Merge commit on branch `main`)
**Student Developer:** Syed Abdul Rehman (Crescent Register No: `230071601233`)
**Organization:** Unified Mentor Pvt. Ltd. (In association with Thales Group Initiative mention)

---

## 1. EXECUTIVE AUDIT SUMMARY

This technical audit provides an objective, rigorous, line-by-line examination of the repository's codebase, data pipelines, machine learning models, user interfaces, documentation, and academic claims.

The project demonstrates a **fully functional, reproducible, and well-structured offline prototype** implementing unsupervised anomaly detection for industrial telemetry. The data synthesis module generates realistic multivariate sensor drift across 10 machines over a 30-day operational horizon. Feature engineering effectively models temporal rolling baselines, and Scikit-learn's Isolation Forest algorithm successfully detects coupled thermal, mechanical, and network anomalies. The Streamlit dashboard offers an impressive dark-themed, glassmorphic visual interface with rich Plotly analytics.

However, the audit identified several **critical discrepancies between the claims made in the documentation (`EXECUTIVE_SUMMARY.md`, `RESEARCH_PAPER.md`) and the actual Python source code (`ml_pipeline.py`, `app.py`)**, including simulated lead-time KPIs, hardcoded metrics, lack of labeled evaluation benchmarks, and the absence of live physical 6G hardware.

---

## 2. FEATURE IMPLEMENTATION STATUS MATRIX

| Feature / Capability | Claimed Status | Verified Status | Implementation File & Line | Audit Verdict & Technical Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Synthetic Telemetry Generation** | Functional | **Fully Implemented** | `generate_dataset.py`: Lines 5–104 | Generates 10,800 rows × 14 columns across 10 machines with seed 42. Deterministic and reproducible. |
| **Operational Modes Modeling** | Functional | **Fully Implemented** | `generate_dataset.py`: Lines 39–45 | Models `Normal` (65%), `High-Load` (25%), and `Eco-Mode` (10%) with distinct temperature, power, and speed offsets. |
| **Gradual Wear Degradation** | Functional | **Fully Implemented** | `generate_dataset.py`: Lines 47–51 | Implements linear wear starting between days 15–25 with variable rates. |
| **Simulated 6G Telemetry** | Functional | **Fully Implemented (Simulated)** | `generate_dataset.py`: Lines 63–65 | Latency generated via Gamma distribution; packet loss via Normal distribution. Pure mathematical simulation; no physical radio hardware. |
| **Time-Series Sorting** | Functional | **Fully Implemented** | `ml_pipeline.py`: Line 12 | Sorts by `['Machine_ID', 'Date', 'Time']` to ensure sequential rolling window calculations. |
| **Rolling Window Baselines** | Functional | **Fully Implemented** | `ml_pipeline.py`: Lines 14–18 | 10-period rolling averages and deviations computed grouped by `Machine_ID`. |
| **Physical Ratio Features** | Functional | **Fully Implemented** | `ml_pipeline.py`: Lines 20–22 | Implements `vib_power_ratio`, `error_escalation`, and `score_decay`. |
| **Isolation Forest Anomaly Model** | Functional | **Fully Implemented** | `ml_pipeline.py`: Lines 33–34 | `IsolationForest(n_estimators=150, contamination=0.08, random_state=42)` fitted on 11 features. |
| **Anomaly Score Normalization** | Functional | **Fully Implemented** | `ml_pipeline.py`: Lines 37–40 | Inverts negative decision function and applies `MinMaxScaler(0, 100)`. |
| **3-Tier Risk Classification** | Functional | **Fully Implemented** | `ml_pipeline.py`: Lines 43–51 | Categorizes into `High` (>=65), `Medium` (>=35), and `Low` (<35). |
| **Model Artifact Serialization** | Functional | **Fully Implemented** | `ml_pipeline.py`: Line 55 | Saves model, scaler, and feature list into `model_artifacts.joblib`. |
| **Streamlit Interactive UI** | Functional | **Fully Implemented** | `app.py`: Lines 1–267 | Complete multi-tab glassmorphic dashboard with Plotly charts and reactive filters. |
| **Machine & Mode Filtering** | Functional | **Fully Implemented** | `app.py`: Lines 88–103 | Interactive sidebar controls dynamically update metrics and visualizations. |
| **Dynamic Alert Queue** | Functional | **Fully Implemented** | `app.py`: Lines 229–247 | Filtered inspection table with automated action recommendations and lead time windows. |
| **Early Warning Lead Time (18.5 hrs)** | Empirical Claim | **Simulated / Hardcoded** | `ml_pipeline.py`: Line 62, `app.py`: Line 137 | **Discrepancy:** In `ml_pipeline.py`, lead time is generated via `np.random.uniform(14.5, 28.0)`. In `app.py`, `"18.5 hrs"` is static text in the metric card. Not empirically measured. |
| **Downtime Prevention Index (96.5%)** | Empirical Claim | **Heuristic Formula** | `ml_pipeline.py`: Line 63, `app.py`: Line 115 | Calculated using formula `(1.0 - (high_risk / total)) * 96.5`. Scaled by constant; not an empirical operational survival metric. |
| **Closed-Loop PLC Control** | Claimed | **Concept Only (Unimplemented)** | `RESEARCH_PAPER.md`: Line 106 | No Modbus, OPC-UA, or PLC hardware interface exists in the codebase. |
| **Streaming / Real-Time Broker** | Claimed | **Unimplemented** | `EXECUTIVE_SUMMARY.md`: Line 44 | No Kafka, MQTT, RabbitMQ, or WebSocket server implemented; reads static CSV from disk. |
| **Supervised Evaluation Metrics** | Implied | **Not Applicable / Unimplemented**| None | No Confusion Matrix, Precision, Recall, F1, or ROC-AUC computed because learning is unsupervised. |

---

## 3. CODE QUALITY AND ARCHITECTURAL OBSERVATIONS

### 3.1 Strengths
1. **Clean Modular Structure:** Clear separation between synthetic data synthesis (`generate_dataset.py`), analytics/modeling (`ml_pipeline.py`), and visualization (`app.py`).
2. **Determinism and Reproducibility:** Explicit seeding (`np.random.seed(42)` and `random_state=42`) guarantees exact numerical reproducibility across independent environments.
3. **Data Sanitization:** Zero null values, zero duplicate rows, and safe division guards (`+ 1e-5`) in ratio feature calculations.
4. **UI Design Excellence:** Sophisticated CSS styling (glassmorphism, subtle gradients, high-contrast dark theme) and responsive Plotly integrations.
5. **Fallback Handling in UI:** `app.py` gracefully checks for `processed_dataset.csv` before falling back to `dataset.csv` or displaying informative warning banners.

### 3.2 Code Quality Issues and Technical Debt

#### Observation 1: Unused Engineered Feature in Model Training
- **File:** `ml_pipeline.py`, Lines 22 and 25–29
- **Issue:** In Line 22, the script engineers `df["score_decay"] = 100.0 - df["Predictive_Maintenance_Score"]`. However, `score_decay` is completely omitted from `feature_cols` (Lines 25–29) and therefore never passed into `iso_forest.fit(X)`.
- **Severity:** Low. `score_decay` is derived directly from the synthetic ground-truth score; omitting it actually prevents potential target leakage into the unsupervised model. However, calculating an unused feature constitutes dead code.

#### Observation 2: Duplicate Import in `app.py`
- **File:** `app.py`, Lines 5–6
- **Issue:**
  ```python
  5: import plotly.graph_objects as io
  6: import plotly.graph_objects as go
  ```
  `plotly.graph_objects` is imported twice under two different aliases (`io` and `go`), neither of which is subsequently referenced (the file strictly utilizes `plotly.express as px`).
- **Severity:** Trivial / Code Polish.

#### Observation 3: Deprecated Streamlit Parameter
- **File:** `app.py`, Lines 176, 191, 205, 217, 227, 244, 264
- **Issue:** Uses `use_container_width=True` on `st.plotly_chart` and `st.dataframe`. In Streamlit versions >= 1.60, this logs deprecation warnings advising migration to `width='stretch'`.
- **Severity:** Low. Does not break execution in current runtime.

#### Observation 4: Scikit-Learn Unpickle Version Inconsistency
- **File:** `model_artifacts.joblib`
- **Issue:** Runtime logs warn that `IsolationForest` and `MinMaxScaler` were pickled with Scikit-learn version `1.9.0`, whereas the environment runs version `1.9.1`.
- **Severity:** Low. Unpickling succeeds without regression, but models should ideally be serialized in the matching runtime environment.

---

## 4. DISCREPANCIES BETWEEN DOCUMENTATION AND SOURCE CODE

### Discrepancy 1: High-Risk Instance Count & Percentage
- **Documentation Claim (`RESEARCH_PAPER.md`, Line 96):**
  `High-Risk Telemetry Spikes: 4.2%`
- **Empirical Reality in `processed_dataset.csv`:**
  `High` risk instances (Anomaly Score >= 65): **146 records out of 10,800**, which is exactly **1.35%**.
- **Root Cause:** The documentation rounded or assumed a different contamination threshold during initial drafting.

### Discrepancy 2: Population Mean Anomaly Score
- **Documentation Claim (`RESEARCH_PAPER.md`, Line 95):**
  `Average Anomaly Score: 24.8 / 100`
- **Empirical Reality in `processed_dataset.csv`:**
  Exact mean anomaly score across all 10,800 records: **22.67 / 100**.

### Discrepancy 3: Early Warning Lead Time (18.5 Hours)
- **Documentation Claim (`RESEARCH_PAPER.md`, Line 13, 97; `EXECUTIVE_SUMMARY.md`, Line 17):**
  Presents "18.5 Hours Early Warning Lead Time" as an empirical validation result.
- **Empirical Reality in Code:**
  - In `ml_pipeline.py` (Line 62): `avg_lead_time_hrs = round(np.random.uniform(14.5, 28.0), 1)`
  - In `app.py` (Line 137): Hardcoded string `<div class="metric-value">18.5 hrs</div>`.
- **Audit Verdict:** The lead time is a research target metric simulated via uniform random variables, not an empirically observed failure progression tracked across physical machine run-to-failure cycles.

### Discrepancy 4: Downtime Prevention Index (96.5%)
- **Documentation Claim (`RESEARCH_PAPER.md`, Line 98):**
  `Downtime Prevention Index: 96.5%`
- **Empirical Reality in Code:**
  - In `ml_pipeline.py` (Line 63): `downtime_prevention_idx = round((1.0 - (high_risk_count / total_records)) * 96.5, 2)` (evaluates to **95.19%**).
  - In `app.py` (Line 115): `downtime_index = round((1.0 - (high_risk_records / max(1, total_records))) * 97.4, 1)` (evaluates to **96.1%** for unfiltered data).
- **Audit Verdict:** This metric is derived from an arbitrary heuristic formula scaling the proportion of non-high-risk records by a constant multiplier (96.5 or 97.4).

### Discrepancy 5: 6G Physical Connectivity vs. Simulation
- **Documentation Claim (`EXECUTIVE_SUMMARY.md`, `RESEARCH_PAPER.md`):**
  Suggests active deployment in a "6G URLLC integrated factory network".
- **Empirical Reality in Code:**
  No networking sockets, RF hardware interfaces, or 5G/6G core network APIs are invoked. 6G latency and packet loss are synthesized purely as random statistical distributions.

---

## 5. BUGS, SECURITY, AND PRODUCTION RISKS

### 5.1 Security Evaluation
1. **Credentials & Secrets:** Zero hardcoded API keys, tokens, or database passwords detected. Repository is clean of credential leaks.
2. **Input Validation:** The Streamlit app does not accept arbitrary file uploads or user-supplied raw text strings that could be subject to code injection.
3. **Execution Safety:** Model deserialization via `joblib.load()` carries inherent remote code execution (RCE) risk if untrusted artifact files are loaded. In production, weights should be stored in secure formats (e.g., ONNX, SafeTensors) or validated cryptographically via SHA-256 signatures before loading.

### 5.2 Scalability & Performance Bottlenecks
1. **In-Memory Dataset Loading:** `app.py` loads the entire dataset into Pandas memory via `@st.cache_data`. While optimal for 10,800 records (2.3 MB), this will crash memory buffers if scaled to millions of streaming IoT records.
2. **Batch Rolling Windows:** `ml_pipeline.py` computes rolling statistics over the full historical batch. For streaming data, an online streaming accumulator (such as Welford's algorithm or an sliding circular buffer in Redis) must be implemented.

---

## 6. AUDIT CONCLUSION & RECOMMENDATIONS FOR INTERNSHIP REPORT

1. **Acknowledge the Project as an Advanced Proof-of-Concept (PoC):** In the academic internship report, describe the system honestly as a **high-fidelity simulation and machine learning diagnostic prototype** designed for 6G smart manufacturing scenarios.
2. **Transparently Define Metric Derivations:** Explicitly explain in the report that the "18.5 Hours Lead Time" and "96.5% Downtime Prevention Index" represent **simulated operational targets and mathematical model bounds**, rather than field-measured physical breakdown telemetry.
3. **Highlight Genuine Technical Accomplishments:** Emphasize the multivariate Isolation Forest mathematics, the dynamic rolling feature engineering that eliminates static threshold blindspots, and the production-grade glassmorphic Streamlit dashboard.
