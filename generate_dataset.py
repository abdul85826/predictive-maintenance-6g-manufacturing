import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_smart_manufacturing_dataset(num_machines=10, days=30, records_per_day_per_machine=36):
    """
    Generates synthetic 6G smart manufacturing telemetry data with realistic sensor profiles,
    6G network metrics, and failure degradation patterns.
    """
    np.random.seed(42)
    machines = [f"M_{i:03d}" for i in range(1, num_machines + 1)]
    operation_modes = ["Normal", "High-Load", "Eco-Mode"]
    
    start_date = datetime(2026, 6, 1, 0, 0)
    data = []
    
    # Machine baseline parameters
    machine_baselines = {}
    for m in machines:
        machine_baselines[m] = {
            "base_temp": np.random.uniform(55, 68),
            "base_vib": np.random.uniform(40, 55),
            "base_power": np.random.uniform(12, 18),
            "health_decay_start": np.random.randint(15, 25), # day when wear starts
            "degradation_rate": np.random.uniform(0.05, 0.25)
        }
        
    total_intervals = days * records_per_day_per_machine
    minute_interval = int((24 * 60) / records_per_day_per_machine)
    
    for i in range(total_intervals):
        current_dt = start_date + timedelta(minutes=i * minute_interval)
        current_day = (current_dt - start_date).days
        
        for m in machines:
            base = machine_baselines[m]
            
            # Select operating mode
            mode_prob = [0.65, 0.25, 0.10]
            op_mode = np.random.choice(operation_modes, p=mode_prob)
            
            mode_temp_modifier = 10.0 if op_mode == "High-Load" else (-3.0 if op_mode == "Eco-Mode" else 0.0)
            mode_power_modifier = 4.0 if op_mode == "High-Load" else (-2.5 if op_mode == "Eco-Mode" else 0.0)
            mode_speed_modifier = 250 if op_mode == "High-Load" else (-150 if op_mode == "Eco-Mode" else 0.0)
            
            # Health degradation factor
            if current_day >= base["health_decay_start"]:
                days_decaying = current_day - base["health_decay_start"]
                degradation = days_decaying * base["degradation_rate"]
            else:
                degradation = 0.0
                
            # Random anomalies (5% chance of spike)
            anomaly_spike = 1.0
            if np.random.rand() < 0.04:
                anomaly_spike = np.random.uniform(1.3, 1.8)
                
            # Physics calculations
            temp_c = round(base["base_temp"] + mode_temp_modifier + (degradation * 3.5) * anomaly_spike + np.random.normal(0, 1.2), 2)
            vib_hz = round(base["base_vib"] + (degradation * 4.2) * anomaly_spike + np.random.normal(0, 1.5), 2)
            power_kw = round(base["base_power"] + mode_power_modifier + (degradation * 1.1) * anomaly_spike + np.random.normal(0, 0.6), 2)
            
            # 6G Network Telemetry (Ultra-low latency base ~ 1-3ms)
            network_latency_ms = round(np.random.gamma(shape=2.0, scale=0.8) + (1.5 if anomaly_spike > 1.2 else 0), 2)
            packet_loss_pct = round(max(0.0, np.random.normal(0.05, 0.03) + (0.4 if network_latency_ms > 4 else 0)), 3)
            
            # Production Metrics
            base_speed = 1000 + mode_speed_modifier - (degradation * 20)
            speed_units_per_hr = int(max(200, base_speed + np.random.normal(0, 25)))
            
            error_rate_pct = round(max(0.1, 0.5 + (degradation * 0.8) * anomaly_spike + np.random.normal(0, 0.2)), 2)
            defect_rate_pct = round(max(0.05, 0.3 + (error_rate_pct * 0.4) + np.random.normal(0, 0.15)), 2)
            
            # Predictive Maintenance Score (100 = Perfect health, decaying to <40 during high degradation)
            maint_score = round(max(10.0, min(100.0, 98.0 - (degradation * 12.0) - (temp_c - base["base_temp"]) * 0.8 - (vib_hz - base["base_vib"]) * 0.6 + np.random.normal(0, 1.5))), 2)
            
            # Efficiency Status
            if maint_score >= 75 and error_rate_pct < 2.5:
                efficiency_status = "High"
            elif maint_score >= 45 and error_rate_pct < 5.5:
                efficiency_status = "Medium"
            else:
                efficiency_status = "Low"
                
            data.append({
                "Date": current_dt.strftime("%Y-%m-%d"),
                "Time": current_dt.strftime("%H:%M:%S"),
                "Machine_ID": m,
                "Operation_Mode": op_mode,
                "Temperature_C": temp_c,
                "Vibration_Hz": vib_hz,
                "Power_Consumption_kW": power_kw,
                "Network_Latency_ms": network_latency_ms,
                "Packet_Loss_%": packet_loss_pct,
                "Quality_Control_Defect_Rate_%": defect_rate_pct,
                "Production_Speed_units_per_hr": speed_units_per_hr,
                "Predictive_Maintenance_Score": maint_score,
                "Error_Rate_%": error_rate_pct,
                "Efficiency_Status": efficiency_status
            })
            
    df = pd.DataFrame(data)
    df.to_csv("dataset.csv", index=False)
    print(f"Dataset successfully generated with {len(df)} rows and saved to dataset.csv.")

if __name__ == "__main__":
    generate_smart_manufacturing_dataset()
