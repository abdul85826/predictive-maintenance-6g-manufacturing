# BUG AND ERROR REGISTER

**Project:** Predictive Maintenance for 6G-Enabled Smart Manufacturing Using Machine Learning
**Repository:** `https://github.com/abdul85826/predictive-maintenance-6g-manufacturing`
**Date:** November 2026
**Status:** Open / Actionable

---

## 1. COMPREHENSIVE DEFECT REGISTER

| Defect ID | Severity | Category | File Location | Description | Root Cause | Proposed Remediation |
| :---: | :---: | :---: | :--- | :--- | :--- | :--- |
| **BUG-01** | Low | Code Quality | `app.py`: Lines 5–6 | Duplicate and unused imports of `plotly.graph_objects` under aliases `io` and `go`. | Redundant imports left during drafting; code strictly utilizes `plotly.express as px`. | Remove redundant imports or use `go` cleanly where required. |
| **BUG-02** | Low | Deprecation Warning | `app.py`: Lines 176, 191, 205, 217, 227, 244, 264 | Deprecation warnings logged in Streamlit >= 1.60 for `use_container_width=True`. | Streamlit updated its layout API to standardize on `width='stretch'`. | Replace `use_container_width=True` with `width="stretch"` or wrap with backward-compatible helper. |
| **BUG-03** | Medium | Reliability / Offline | `app.py`: Line 83 | External image dependency `https://img.icons8.com/isometric-folders/100/processor.png` in sidebar. | Relies on third-party internet CDN; fails in air-gapped, offline, or restricted university labs. | Replace with local, self-contained SVG project brand asset (`assets/branding/project_symbol.svg`). |
| **BUG-04** | Low | Dead Code | `ml_pipeline.py`: Line 22 | `score_decay` is engineered and saved into `processed_dataset.csv`, but omitted from `feature_cols`. | Calculated during feature drafting but excluded from `X` to avoid target leakage; left uncleaned. | Document feature as diagnostic auxiliary column or remove dead assignment. |
| **BUG-05** | High | Scientific Integrity | `ml_pipeline.py`: Line 62, `app.py`: Line 137 | Early warning lead time is drawn from random uniform distribution (`np.random.uniform(14.5, 28.0)`) and hardcoded as `"18.5 hrs"`. | Heuristic placeholder used during prototyping instead of real degradation curve calculation. | Replace hardcoded metric with transparent degradation-based lead-time estimation based on rolling drift slope and clearly disclose methodology. |
| **BUG-06** | High | Scientific Integrity | `ml_pipeline.py`: Line 63, `app.py`: Line 115 | Downtime Prevention Index is calculated with an arbitrary heuristic multiplier (`* 96.5` or `* 97.4`). | Ad-hoc formula invented to approximate a business KPI rather than empirical survival metric. | Recalibrate as an honest **Asset Reliability Index** or **Operational Availability Index** calculated directly from empirical non-anomalous operational hours. |
| **BUG-07** | Medium | Discrepancy | `RESEARCH_PAPER.md`: Line 96 | Claims High-Risk Telemetry Spikes = 4.2%, whereas actual data contains exactly 146 records (1.35%). | Documentation was written with a theoretical estimate or earlier iteration with different contamination. | Correct documentation to state exact empirical count: 146 records (1.35%). |
| **BUG-08** | Medium | Discrepancy | `RESEARCH_PAPER.md`: Line 95 | Claims Average Anomaly Score = 24.8 / 100, whereas actual empirical mean is 22.67 / 100. | Discrepancy from unrounded draft estimates. | Align documentation with exact empirical mean (22.7 / 100). |
| **BUG-09** | Medium | UX / Functionality | `app.py`: Sidebar | Missing Date Range filter. Dashboard only allows Machine ID and Operation Mode selection. | Feature was planned but omitted from the initial Streamlit implementation. | Implement a dynamic `st.sidebar.date_input` control filtering data within `2026-06-01` to `2026-06-30`. |
| **BUG-10** | Low | UX / Usability | `app.py`: Sidebar | No Filter Reset button; users must manually reset each dropdown and slider. | Absence of session-state reset control. | Add a "Reset Filters" button in the sidebar that restores default fleet-wide view. |
| **BUG-11** | Medium | QA / Testing | Repository Root | Complete absence of automated tests or test runner scripts. | Fast prototyping prioritized initial demo over test harness. | Create `tests/test_pipeline.py` with PyTest / Python `unittest` verifying data schema, ML pipeline, and model loading. |
| **BUG-12** | Low | Environment / Pickle | `model_artifacts.joblib` | InconsistentVersionWarning: estimator pickled with Scikit-learn 1.9.0 loaded in 1.9.1. | Model was serialized in a development environment with Scikit-learn 1.9.0. | Re-serialize model artifacts using the active runtime environment to ensure clean unpickling. |
| **BUG-13** | Low | Brand Identity | `app.py`: Line 12 | Uses generic emoji `"⚡"` as page icon and favicon. | Lack of dedicated project icon asset. | Link to `assets/branding/favicon.svg` in page config and header. |
