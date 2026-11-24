<!-- markdownlint-disable MD033 MD013 MD025 MD024 -->
# ARCHITECTURE AND TECHNICAL DIAGRAMS SPECIFICATION
## PREDICTIVE MAINTENANCE FOR 6G-ENABLED SMART MANUFACTURING

This document provides formal architectural models, flowcharts, and Data Flow Diagrams (DFDs) representing the end-to-end design of the predictive maintenance platform. Every diagram is accompanied by an architectural breakdown mapped directly to source files and line numbers.

---

## 1. HIGH-LEVEL SYSTEM ARCHITECTURE

```mermaid
graph TD
    subgraph "Smart Factory Floor & IIoT Layer"
        M1["Machine Cell M_001"]
        M2["Machine Cell M_002"]
        M3["Machine Cell ..."]
        M10["Machine Cell M_010"]
        Sensors["Multi-Modal Sensors<br/>(Thermal, Vibration, Power, Speed)"]
        M1 & M2 & M3 & M10 --> Sensors
    end

    subgraph "6G Telemetry Simulation Layer"
        URLLC["6G URLLC Radio Interface<br/>(Sub-ms Latency, Packet Loss)"]
        Sensors --> URLLC
        Gen["Data Synthesis Engine<br/>(generate_dataset.py)"]
        URLLC --> Gen
        RawDB[("Raw Telemetry Store<br/>dataset.csv")]
        Gen --> RawDB
    end

    subgraph "ML Analytics & Feature Engine"
        Pipeline["ML Pipeline Service<br/>(ml_pipeline.py)"]
        RawDB --> Pipeline
        FE["Rolling Window Feature Engineering<br/>(Window = 10 per Machine)"]
        IsoModel["Isolation Forest Anomaly Model<br/>(150 Trees, Contamination 0.08)"]
        Scaler["MinMax Score Normalizer<br/>(0 - 100 Scale)"]
        Pipeline --> FE
        FE --> IsoModel
        IsoModel --> Scaler
        ProcDB[("Processed Store<br/>processed_dataset.csv")]
        ModelArt[("Model Artifacts<br/>model_artifacts.joblib")]
        Scaler --> ProcDB
        IsoModel --> ModelArt
        Scaler --> ModelArt
    end

    subgraph "Presentation & Executive Control Layer"
        Streamlit["Executive Control Tower<br/>(app.py - Streamlit)"]
        ProcDB --> Streamlit
        ModelArt -.-> Streamlit
        Dashboard["Interactive Dashboard<br/>(Overview, Trends, Alerts, Heatmaps)"]
        Streamlit --> Dashboard
        Engineer["Plant Maintenance Engineer / Supervisor"]
        Dashboard <--> Engineer
    end
```

### Technical Explanation: High-Level Architecture
- **Factory Floor Layer:** Simulates 10 industrial machines (`M_001` to `M_010`) operating across `Normal`, `High-Load`, and `Eco-Mode` regimes (`generate_dataset.py`, lines 11–12).
- **6G Telemetry Layer:** Combines physical sensors with 6G network indicators (Gamma-distributed latency and Normal-distributed packet loss) saved to `dataset.csv` (`generate_dataset.py`, lines 63–66).
- **Analytics Engine:** Ingests `dataset.csv`, computes grouped rolling moving averages and physical ratios, fits 150 isolation trees, normalizes scores from 0 to 100, and saves `processed_dataset.csv` and `model_artifacts.joblib` (`ml_pipeline.py`, lines 7–55).
- **Presentation Layer:** Reactive web UI in `app.py` delivering real-time filtering, risk aggregation, and prescriptive inspection scheduling.

---

## 2. COMPONENT ARCHITECTURE DIAGRAM

```mermaid
classDiagram
    class DataGenerator {
        +int num_machines
        +int days
        +int records_per_day
        +dict machine_baselines
        +generate_smart_manufacturing_dataset()
        +export_to_csv()
    }

    class FeatureEngineeringPipeline {
        +DataFrame raw_df
        +int rolling_window
        +sort_timeseries()
        +compute_rolling_averages()
        +compute_deviations()
        +compute_instability_ratios()
    }

    class IsolationForestModel {
        +int n_estimators
        +float contamination
        +int random_state
        +list feature_cols
        +fit(X)
        +decision_function(X)
    }

    class ScoreCalibrator {
        +MinMaxScaler scaler
        +invert_decision_scores()
        +normalize_to_100()
        +assign_risk_level(score)
    }

    class StreamlitDashboard {
        +DataFrame dataset
        +dict active_filters
        +render_kpi_cards()
        +render_risk_pie_chart()
        +render_anomaly_trend_chart()
        +render_bivariate_scatter()
        +render_alert_priority_queue()
        +render_temporal_heatmap()
    }

    DataGenerator ..> FeatureEngineeringPipeline : dataset.csv
    FeatureEngineeringPipeline --> IsolationForestModel : Feature Matrix X (11 cols)
    IsolationForestModel --> ScoreCalibrator : Raw Decision Scores
    ScoreCalibrator ..> StreamlitDashboard : processed_dataset.csv
    IsolationForestModel ..> StreamlitDashboard : model_artifacts.joblib
```

### Technical Explanation: Component Architecture
- **DataGenerator (`generate_dataset.py`):** Configures machine baseline dictionaries with initial operating temperatures, vibration, power, wear inception days, and degradation velocities.
- **FeatureEngineeringPipeline (`ml_pipeline.py`):** Executes time-series sorting and windowed group transformations using Pandas `.transform(lambda x: x.rolling(10, min_periods=1).mean())`.
- **IsolationForestModel (`ml_pipeline.py`):** Encapsulates Scikit-learn's `IsolationForest` parameterized with 150 estimators and 8% contamination.
- **ScoreCalibrator (`ml_pipeline.py`):** Inverts the negative decision function and maps scores into $[0.0, 100.0]$, classifying them into discrete `High`, `Medium`, and `Low` risk buckets.
- **StreamlitDashboard (`app.py`):** Ingests processed data, provides reactive caching via `@st.cache_data`, and generates Plotly graphical objects.

---

## 3. DATA FLOW DIAGRAMS (DFD)

### 3.1 DFD Level 0 (Context Diagram)

```mermaid
graph LR
    FactorySensors["Smart Manufacturing Machines<br/>& 6G Telemetry Sensors"]
    System(("6G Smart Manufacturing<br/>Predictive Maintenance<br/>Platform"))
    Engineer["Plant Maintenance Engineer<br/>& Plant Operations Lead"]

    FactorySensors -->|"Raw Multi-Sensor Telemetry<br/>& Network QoS Signals"| System
    System -->|"Maintenance Risk Alerts,<br/>Prescriptive Actions & RUL Estimates"| Engineer
    Engineer -->|"Machine Filtering,<br/>Threshold Tuning & Mode Selections"| System
```

### 3.2 DFD Level 1 (Functional Decomposition)

```mermaid
graph TD
    S1["Sensor Stream / Generator"] -->|"Physical & Network Metrics"| P1["1.0 Ingestion & Time-Series Sorting"]
    P1 -->|"Sorted Dataframe"| D1[("D1: Raw Telemetry Store")]

    D1 -->|"Raw Records"| P2["2.0 Dynamic Baseline & Feature Extraction"]
    P2 -->|"11-Dimensional Feature Vector X"| P3["3.0 Unsupervised Anomaly Isolation"]

    P3 -->|"Raw Decision Scores"| P4["4.0 Score Calibration & Risk Mapping"]
    P4 -->|"Processed Records with Anomaly Scores"| D2[("D2: Processed Dataset Store")]
    P3 -->|"Model & Scaler Weights"| D3[("D3: Serialized Artifact Store")]

    D2 -->|"Aggregated Telemetry"| P5["5.0 Executive Dashboard Presentation"]
    D3 -.->|"Model Hyperparameters"| P5

    User["Plant Engineer"] <-->|"Filter Queries & Alert Triggers"| P5
```

---

## 4. MACHINE LEARNING PIPELINE DIAGRAM

```mermaid
flowchart TD
    subgraph "Feature Extraction (11 Features)"
        F1["Temperature_C"]
        F2["Vibration_Hz"]
        F3["Power_Consumption_kW"]
        F4["Network_Latency_ms"]
        F5["Packet_Loss_%"]
        F6["Quality_Control_Defect_Rate_%"]
        F7["Error_Rate_%"]
        F8["temp_deviation = Temp - Rolling10(Temp)"]
        F9["vib_deviation = Vib - Rolling10(Vib)"]
        F10["vib_power_ratio = Vib / (Power + 1e-5)"]
        F11["error_escalation = (Error_Rate / Maint_Score) * 100"]
    end

    X["Feature Matrix X<br/>Shape: (10800, 11)"]
    F1 & F2 & F3 & F4 & F5 & F6 & F7 & F8 & F9 & F10 & F11 --> X

    subgraph "Isolation Forest Ensemble"
        IF["IsolationForest(n_estimators=150, contamination=0.08, random_state=42)"]
        T1["iTree 1<br/>Random Splits"]
        T2["iTree 2<br/>Random Splits"]
        T150["iTree 150<br/>Random Splits"]
        X --> IF
        IF --> T1 & T2 & T150
    end

    subgraph "Path Length & Scoring"
        PL["Average Path Length E(h(x))"]
        T1 & T2 & T150 --> PL
        RawScore["Decision Score: s(x, n) = 2^(-E(h(x)) / c(n))"]
        PL --> RawScore
        Invert["Invert Sign: raw_scores = -decision_function(X)"]
        RawScore --> Invert
    end

    subgraph "Risk Normalization"
        MMS["MinMaxScaler(feature_range=(0, 100))"]
        Invert --> MMS
        AS["Anomaly_Score in [0.00, 100.00]"]
        MMS --> AS
        Rule{"Score Threshold Classification"}
        AS --> Rule
        Rule -->|Score >= 65| High["High Risk (146 records / 1.35%)<br/>Action: Immediate Inspection (< 6 hrs)"]
        Rule -->|35 <= Score < 65| Med["Medium Risk (1631 records / 15.10%)<br/>Action: Schedule Recalibration (12-24 hrs)"]
        Rule -->|Score < 35| Low["Low Risk (9023 records / 83.55%)<br/>Action: Normal Operation"]
    end
```

---

## 5. APPLICATION EXECUTION FLOWCHART (`app.py`)

```mermaid
flowchart TD
    Start([Launch: streamlit run app.py]) --> PageCfg[Set Page Config: Title, Icon, Wide Layout]
    PageCfg --> InjectCSS[Inject Glassmorphic Dark Theme CSS]
    InjectCSS --> LoadData{Check for Datafile}

    LoadData -->|processed_dataset.csv exists| ReadProc[Load processed_dataset.csv via @st.cache_data]
    LoadData -->|Only dataset.csv exists| ReadRaw[Load dataset.csv with Warning Banner]
    LoadData -->|No file found| ShowErr[Display Error Banner & Halt]

    ReadProc --> Sidebar[Render Sidebar Controls]
    ReadRaw --> Sidebar

    Sidebar --> FilterM[Select Machine ID: All or M_001..M_010]
    Sidebar --> FilterMode[Multiselect: Normal, High-Load, Eco-Mode]
    Sidebar --> FilterThresh[Set Anomaly Alert Threshold: Slider 0-100, Default 35]

    FilterThresh --> ApplyFilter[Filter Dataframe according to User Selections]

    ApplyFilter --> RenderKPIs[Render Top KPI Metric Cards:<br/>Avg Score, High Risk Count, Lead Time, Prevention Index]

    RenderKPIs --> TabSelection{User Selects Dashboard Tab}

    TabSelection -->|Tab 1| T1[Overview & Risk Distribution:<br/>- Risk Donut Pie Chart<br/>- Machine Breakdown Bar Chart]
    TabSelection -->|Tab 2| T2[Anomaly & Sensor Trends:<br/>- Time-Series Anomaly Drift with Alert Lines<br/>- Temp vs Vib Scatter<br/>- Latency vs Packet Loss Scatter]
    TabSelection -->|Tab 3| T3[Maintenance Alert Panel:<br/>- Priority Filtered Inspection Table<br/>- Prescriptive Action Recommendations<br/>- Lead Time Windows]
    TabSelection -->|Tab 4| T4[Historical Risk Escalation:<br/>- Density Heatmap of Machine Risk Intensity over Days]

    T1 & T2 & T3 & T4 --> End([Await User Interaction / Reactive Re-run])
```

---

## 6. MODEL TRAINING FLOWCHART (`ml_pipeline.py`)

```mermaid
flowchart TD
    Start([Execute: python ml_pipeline.py]) --> ReadCSV[Load dataset.csv via pd.read_csv]
    ReadCSV --> SortDF[Sort Dataframe by Machine_ID, Date, Time]

    SortDF --> GroupM[Group Dataframe by Machine_ID]
    GroupM --> RollT[Compute rolling 10-window mean of Temperature_C]
    GroupM --> RollV[Compute rolling 10-window mean of Vibration_Hz]

    RollT --> DiffT[Compute temp_deviation = Temp - temp_rolling_avg]
    RollV --> DiffV[Compute vib_deviation = Vib - vib_rolling_avg]

    DiffT & DiffV --> RatioVP[Compute vib_power_ratio = Vib / Power + 1e-5]
    RatioVP --> ErrEsc[Compute error_escalation = Error_Rate / Maint_Score * 100]
    ErrEsc --> ScoreDecay[Compute score_decay = 100 - Maint_Score]

    ScoreDecay --> DefineX[Assemble Feature Matrix X from 11 Selected Columns]
    DefineX --> FillNA[Impute any missing values with 0 via .fillna 0]

    FillNA --> InitModel[Initialize IsolationForest: n_estimators=150, contamination=0.08, random_state=42]
    InitModel --> FitModel[Fit Isolation Forest Model on X]

    FitModel --> PredictRaw[Compute raw decision scores via -iso_forest.decision_function X]
    PredictRaw --> InitScaler[Initialize MinMaxScaler: range = 0 to 100]
    InitScaler --> FitScaler[Transform raw scores to Anomaly_Score]

    FitScaler --> ApplyTiers[Assign Maintenance_Risk_Level: High >= 65, Medium >= 35, Low < 35]

    ApplyTiers --> SaveProc[Export processed_dataset.csv to disk]
    ApplyTiers --> SaveJoblib[Export model_artifacts.joblib with Model, Scaler, and Features]

    SaveProc & SaveJoblib --> PrintKPIs[Print Summary Metrics to Terminal Output]
    PrintKPIs --> End([Pipeline Complete])
```

---

## 7. MODEL INFERENCE FLOWCHART (REAL-TIME STREAMING EXTENSION)

```mermaid
flowchart TD
    StreamIn([Incoming 6G Telemetry Frame]) --> ParseJSON[Parse Telemetry JSON:<br/>Temp, Vib, Power, Latency, Packet Loss, Speed, Mode]
    ParseJSON --> Buffer[Query Circular Window Buffer for Machine_ID<br/>Length = 10 Historical Frames]

    Buffer --> CompRoll[Compute Dynamic Rolling Means & Deviations]
    CompRoll --> AssembleX[Assemble 11-Dimensional Feature Vector x]

    AssembleX --> LoadArtifacts[Load Model & Scaler from model_artifacts.joblib]
    LoadArtifacts --> ComputePath[Compute Average Path Length across 150 iTrees]

    ComputePath --> CalcDecision[Calculate Decision Function: raw_score = -decision_function x]
    CalcDecision --> ScaleScore[Apply Saved MinMaxScaler: Anomaly_Score in 0-100]

    ScaleScore --> EvaluateRisk{Risk Tier Evaluation}

    EvaluateRisk -->|Score >= 65| TriggerCrit[CRITICAL ALERT TRIGGER:<br/>1. Log Priority Incident<br/>2. Send Automated 6G Down-Throttle to Machine PLC<br/>3. Dispatch Immediate Maintenance Work Order]
    EvaluateRisk -->|35 <= Score < 65| TriggerWarn[WARNING ALERT:<br/>1. Flag for Routine Inspection<br/>2. Increase Telemetry Sampling Frequency]
    EvaluateRisk -->|Score < 35| LogNormal[NORMAL STATE:<br/>Store Telemetry in Historical Time-Series DB]

    TriggerCrit & TriggerWarn & LogNormal --> AwaitNext([Await Next Telemetry Packet])
```
