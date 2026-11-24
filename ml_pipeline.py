"""
Predictive Maintenance for 6G-Enabled Smart Manufacturing
Machine Learning Pipeline & Anomaly Detection Service

Module: ml_pipeline.py
Author: Syed Abdul Rehman
Organization: Unified Mentor Pvt. Ltd.
Framework: Isolation Forest Unsupervised Anomaly Detection
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

# Base project directory for robust relative path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset.csv")
PROCESSED_PATH = os.path.join(BASE_DIR, "processed_dataset.csv")
MODEL_ARTIFACTS_PATH = os.path.join(BASE_DIR, "model_artifacts.joblib")


def run_ml_pipeline():
    """
    Executes the end-to-end data processing, feature engineering,
    unsupervised Isolation Forest modeling, and risk scoring pipeline.
    """
    print(f"[ML Pipeline] Loading telemetry dataset from: {DATASET_PATH}")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Input dataset not found at: {DATASET_PATH}. Please run generate_dataset.py first.")

    df = pd.read_csv(DATASET_PATH)

    # 1. Baseline Behavior & Rolling Feature Engineering per Machine_ID
    # Sorting ensures sequential rolling window computation across time
    df = df.sort_values(by=["Machine_ID", "Date", "Time"]).reset_index(drop=True)

    # Dynamic rolling baselines (Window = 10 records / ~6.6 hours of operation)
    df["temp_rolling_avg"] = df.groupby("Machine_ID")["Temperature_C"].transform(
        lambda x: x.rolling(window=10, min_periods=1).mean()
    )
    df["temp_deviation"] = df["Temperature_C"] - df["temp_rolling_avg"]

    df["vib_rolling_avg"] = df.groupby("Machine_ID")["Vibration_Hz"].transform(
        lambda x: x.rolling(window=10, min_periods=1).mean()
    )
    df["vib_deviation"] = df["Vibration_Hz"] - df["vib_rolling_avg"]

    # Physical instability ratios with division-by-zero guards
    df["vib_power_ratio"] = df["Vibration_Hz"] / (df["Power_Consumption_kW"] + 1e-5)
    df["error_escalation"] = df["Error_Rate_%"] / (df["Predictive_Maintenance_Score"] + 1e-5) * 100

    # Auxiliary diagnostic column (retained for backward schema compatibility)
    df["score_decay"] = 100.0 - df["Predictive_Maintenance_Score"]

    # 2. Selected Feature Matrix for Unsupervised Anomaly Modeling (11 Features)
    feature_cols = [
        "Temperature_C",
        "Vibration_Hz",
        "Power_Consumption_kW",
        "Network_Latency_ms",
        "Packet_Loss_%",
        "Quality_Control_Defect_Rate_%",
        "Error_Rate_%",
        "temp_deviation",
        "vib_deviation",
        "vib_power_ratio",
        "error_escalation",
    ]

    X = df[feature_cols].fillna(0)

    # 3. Isolation Forest Ensemble Anomaly Detection
    # 150 estimators for tree stability; 8% contamination boundary; seed=42 for reproducibility
    iso_forest = IsolationForest(
        n_estimators=150,
        contamination=0.08,
        random_state=42,
        max_samples="auto",
        bootstrap=False,
    )
    iso_forest.fit(X)

    # Raw decision function score (inverting sign so higher values denote higher anomaly risk)
    raw_scores = -iso_forest.decision_function(X)

    # MinMax Normalization to continuous 0.00 - 100.00 scale
    scaler = MinMaxScaler(feature_range=(0, 100))
    df["Anomaly_Score"] = np.round(scaler.fit_transform(raw_scores.reshape(-1, 1)).flatten(), 2)

    # 4. Multi-Tier Risk Classification
    def assign_risk(score: float) -> str:
        if score >= 65.0:
            return "High"
        elif score >= 35.0:
            return "Medium"
        else:
            return "Low"

    df["Maintenance_Risk_Level"] = df["Anomaly_Score"].apply(assign_risk)

    # 5. Export Processed Dataset & Serialized Model Artifacts
    df.to_csv(PROCESSED_PATH, index=False)
    joblib.dump(
        {"model": iso_forest, "scaler": scaler, "feature_cols": feature_cols},
        MODEL_ARTIFACTS_PATH,
    )

    # 6. Empirical Pipeline KPI Verification
    total_records = len(df)
    high_risk_records = len(df[df["Maintenance_Risk_Level"] == "High"])
    med_risk_records = len(df[df["Maintenance_Risk_Level"] == "Medium"])
    low_risk_records = len(df[df["Maintenance_Risk_Level"] == "Low"])
    high_risk_pct = round((high_risk_records / total_records) * 100, 2)
    operational_availability = round((1.0 - (high_risk_records / total_records)) * 100, 2)
    mean_anomaly_score = round(df["Anomaly_Score"].mean(), 2)

    high_risk_machines = sorted(df[df["Maintenance_Risk_Level"] == "High"]["Machine_ID"].unique().tolist())

    print("\n========================================================")
    print("      ML PIPELINE EXECUTION COMPLETED SUCCESSFULLY      ")
    print("========================================================")
    print(f"Total Telemetry Records Processed: {total_records}")
    print(f"Population Mean Anomaly Score   : {mean_anomaly_score} / 100")
    print(f"Low Risk (Healthy) Records      : {low_risk_records} ({round(low_risk_records/total_records*100, 2)}%)")
    print(f"Medium Risk (Early Warning)     : {med_risk_records} ({round(med_risk_records/total_records*100, 2)}%)")
    print(f"High Risk (Critical Alert)      : {high_risk_records} ({high_risk_pct}%)")
    print(f"Fleet Operational Availability  : {operational_availability}%")
    print(f"Degraded Machine Assets Flagged : {high_risk_machines}")
    print(f"Processed Dataset Saved to      : {PROCESSED_PATH}")
    print(f"Model Artifacts Saved to        : {MODEL_ARTIFACTS_PATH}")
    print("========================================================\n")


if __name__ == "__main__":
    run_ml_pipeline()
