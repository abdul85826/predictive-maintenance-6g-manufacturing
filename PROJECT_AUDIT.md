# PROJECT AUDIT REPORT

**Project:** Predictive Maintenance for 6G-Enabled Smart Manufacturing Using Machine Learning
**Repository:** `https://github.com/abdul85826/predictive-maintenance-6g-manufacturing`
**Auditor:** Senior Software Architect, ML Engineer, Product Designer & QA Lead
**Audit Date:** November 2026
**Environment:** Windows (PowerShell), Python 3.14.7, Pandas 3.1.1, Scikit-learn 1.9.1, Streamlit 1.67.0, Plotly 7.1.0

---

## 1. REPOSITORY INVENTORY & FILE STRUCTURE

```
predictive-maintenance-6g-manufacturing/
│
├── .gitignore                      # Git exclusion rules (Python, venv, cache)
├── LICENSE                         # MIT License (Copyright 2026 Syed Abdul Rehman)
├── README.md                       # Bare repository title (41 bytes)
├── requirements.txt                # 6 direct project dependencies
│
├── dataset.csv                     # Raw synthetic telemetry (10,800 rows x 14 cols, 958 KB)
├── processed_dataset.csv           # Processed telemetry with ML scores (10,800 x 23 cols, 2.29 MB)
├── model_artifacts.joblib          # Serialized model, scaler, and feature list (1.82 MB)
│
├── generate_dataset.py             # Deterministic synthetic data generator (107 lines)
├── ml_pipeline.py                  # Feature engineering & Isolation Forest pipeline (74 lines)
├── app.py                          # Streamlit interactive dashboard (267 lines)
│
├── EXECUTIVE_SUMMARY.md            # Strategic executive briefing (52 lines)
└── RESEARCH_PAPER.md               # Draft research paper (112 lines)
```

---

## 2. CODEBASE & FUNCTIONAL ANALYSIS

### 2.1 Data Synthesis (`generate_dataset.py`)
- **Parameters:** 10 machines (`M_001` - `M_010`), 30 days (`2026-06-01` to `2026-06-30`), 40-minute intervals (36 records/day/machine).
- **Total Rows:** $10 \times 30 \times 36 = 10,800$ rows.
- **Reproducibility:** Seed fixed via `np.random.seed(42)`. Verified bit-exact (MD5: `eab57a157174890e6b361a8de556d20d`).
- **Physics Modeling:** Realistic base offsets per machine, multi-mode offsets (`Normal`, `High-Load`, `Eco-Mode`), linear wear past day 15–25, random 4% shock multiplier, Gamma-distributed network latency, Normal-distributed packet loss, coupled error and defect rates.
- **Integrity:** Zero null values, zero duplicates.

### 2.2 Feature Engineering & Model Training (`ml_pipeline.py`)
- **Ordering:** Correctly sorts by `['Machine_ID', 'Date', 'Time']`.
- **Dynamic Baselines:** Computes 10-window rolling mean of temperature and vibration grouped by `Machine_ID`. Computes deviations `temp_deviation` and `vib_deviation`.
- **Physical Ratios:** Computes `vib_power_ratio` ($\frac{\text{Vibration}}{\text{Power} + 10^{-5}}$) and `error_escalation` ($\frac{\text{Error\_Rate}}{\text{Maint\_Score} + 10^{-5}} \times 100$).
- **Dead Feature:** Computes `score_decay = 100.0 - Predictive_Maintenance_Score` at line 22, but does not include it in `feature_cols` (line 25).
- **Model:** `IsolationForest(n_estimators=150, contamination=0.08, random_state=42)` trained on 11 features.
- **Normalization:** Inverts negative decision scores and scales to $[0.0, 100.0]$ with `MinMaxScaler`.
- **Tiers:** High ($\ge 65$), Medium ($\ge 35$ and $< 65$), Low ($< 35$).
- **Discrepancy:** In lines 62–63, `avg_lead_time_hrs` is drawn randomly via `np.random.uniform(14.5, 28.0)` and `downtime_prevention_idx` is calculated using an ad-hoc multiplier formula.

### 2.3 Web Application (`app.py`)
- **Architecture:** Streamlit application loading `processed_dataset.csv` or `dataset.csv` using `@st.cache_data`.
- **Theme:** Dark glassmorphism with high contrast neon accents (`#38bdf8`, `#ef4444`, `#10b981`, `#f59e0b`).
- **Sidebar:** External URL image icon (`https://img.icons8.com/...`) which fails offline; machine selector, mode filter, risk slider. Lacks date range filter and filter reset button.
- **KPI Metrics:** Metric card displays hardcoded `"18.5 hrs"` lead time and heuristic downtime prevention index.
- **Plotly Visualizations:** Donut risk pie chart, high-risk machine bar chart, time-series anomaly drift line chart, temperature vs. vibration scatter, 6G latency vs. packet loss scatter, and temporal risk heatmap.
- **Alert Queue:** Interactive dataframe with prescriptive maintenance actions.
- **Syntax & Execution:** Successfully imports and compiles without crashes, but logs Streamlit deprecation warnings (`use_container_width=True`) and has duplicate imports (`plotly.graph_objects as io` and `as go`).

---

## 3. IMPLEMENTATION STATUS MATRIX

| Capability | Category | Reality in Code |
| :--- | :---: | :--- |
| **Synthetic Telemetry Generation** | Fully Implemented | 10,800 rows generated deterministically with seed 42. |
| **Time-Series Baseline Extraction** | Fully Implemented | 10-window rolling mean per machine. |
| **Isolation Forest Anomaly Modeling** | Fully Implemented | 150 trees, contamination 0.08, 11 features. |
| **Anomaly Score Normalization** | Fully Implemented | MinMaxScaler scaled 0 to 100. |
| **Multi-Tier Risk Classification** | Fully Implemented | Low (83.55%), Medium (15.10%), High (1.35%). |
| **Interactive Streamlit Dashboard** | Fully Implemented | 4 tabs with Plotly charts and alert queue. |
| **Offline Brand Identity / Logo** | Missing | Uses external web link icon; no local vector branding. |
| **Date Range Filter** | Missing | Filter only supports Machine, Mode, and Threshold. |
| **Filter Reset Functionality** | Missing | Users must manually revert individual controls. |
| **Lead-Time Estimation** | Simulated / Heuristic | Uniform random number generation (14.5 - 28.0 hrs). |
| **Downtime Prevention Index** | Heuristic Formula | Scaled percentage calculation, not survival analysis. |
| **Live 6G Network Connectivity** | Theoretical / Simulated | Mathematical Gamma/Normal distribution; no RF hardware. |
| **Closed-Loop PLC Control** | Proposed / Out-of-Scope | Textual recommendation only; no Modbus/OPC-UA layer. |
| **Automated Testing Suite** | Missing | No `tests/` directory or automated validation scripts. |

---

## 4. AUDIT SUMMARY CONCLUSION

The foundation is mathematically sound, deterministic, and highly functional. To make this project a world-class academic engineering exemplar:
1. Replace the external icon with an authentic, original SVG brand identity.
2. Refactor the visual design system to a clean, natural industrial aesthetic.
3. Clean Python code defects (imports, dead code, deprecation warnings).
4. Provide transparent, truthful metric calculations and explanations.
5. Enhance filtering (add date range and reset button).
6. Build an automated test suite verifying data schemas, ML pipeline, and dashboard stability.
