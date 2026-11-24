# UI / UX AUDIT AND INDUSTRIAL DESIGN SYSTEM SPECIFICATION

**Project:** Predictive Maintenance for 6G-Enabled Smart Manufacturing Using Machine Learning
**Current Interface:** Streamlit 1.67.0 with Dark Glassmorphism CSS injection
**Target Interface:** Natural Industrial Analytics Console
**Date:** November 2026

---

## 1. CURRENT UI DEFECTS AND CRITIQUE

The current interface in `app.py` uses a dark glassmorphic theme inspired by generic consumer web dashboards. While visually striking, an engineering audit reveals significant usability and stylistic deficiencies:

1. **Dark "Gamer / Neon" Aesthetics:** Deep purple-black background (`#0b0f19` to `#161b22`) with high-saturation neon blues (`#38bdf8`) and purples looks like an esports platform rather than an authentic industrial control room console (e.g., Honeywell, Siemens, ABB, Rockwell Automation).
2. **External Web Asset Dependency:** Sidebar displays an icon from an external CDN (`https://img.icons8.com/...`). If internet connectivity drops or is blocked by university proxies, the sidebar renders an unsightly broken image box.
3. **Tabbed Fragmentation:** The dashboard is divided into 4 horizontal tabs. Users must constantly click back and forth between tabs to correlate risk distribution (Tab 1), time-series anomaly spikes (Tab 2), and the alert queue (Tab 3).
4. **Missing Temporal Controls:** Industrial engineers must inspect specific time windows (e.g., last 24 hours, weekend shifts, high-load cycles). The current interface only allows filtering by Machine ID and Mode, with no Date Range selector.
5. **Heuristic Hardcoded KPIs:** The top KPI bar presents `"18.5 hrs Early Lead Time"` as a static string and calculates `"Downtime Prevention Index"` using an ad-hoc formula multiplying by 97.4, giving an illusion of real-time operational telemetry.
6. **Mobile & Viewport Cramping:** Certain Plotly charts (e.g., the 2-column scatter plots in Tab 2) suffer from severe horizontal cramping on screens narrower than 1200px.

---

## 2. THE NATURAL INDUSTRIAL DESIGN SYSTEM

The redesigned interface will adopt a **Premium Natural Industrial Design System** engineered for precision, cognitive clarity, and long operational sessions without eye fatigue.

### 2.1 Color Palette Specifications

```
+-------------------------------------------------------------------------------+
| COMPONENT                  HEX CODE     SWATCH / DESCRIPTION                  |
+-------------------------------------------------------------------------------+
| Primary Background         #F5F6F2      Warm off-white, calm and non-glare     |
| Main Card Surface          #FFFFFF      Crisp white with subtle elevation     |
| Primary Text               #202A2E      Deep industrial charcoal (high contrast)|
| Secondary / Muted Text     #66747A      Muted slate grey for captions & units |
| Primary Brand              #176B62      Deep forest teal (trust & precision)  |
| Secondary Brand            #477D83      Muted blue-green for charts & accents |
| Card & Table Border        #E2E7E4      Soft neutral grey for clear boundaries|
| Success / Low Risk         #39845A      Natural moss green (calm, safe)       |
| Warning / Medium Risk      #C58A32      Muted amber (clear early warning)     |
| Critical / High Risk       #C44F4F      Controlled terracotta red (alerting)  |
+-------------------------------------------------------------------------------+
```

### 2.2 Typography & Visual Hierarchy
- **Primary Typeface:** Clean humanist sans-serif (`Inter`, `-apple-system`, `Segoe UI`, `Roboto`, `Helvetica Neue`, sans-serif).
- **KPI Metrics:** Large 32px / 2.2rem bold typography in `#176B62` or `#202A2E`, accompanied by 12px uppercase slate labels and explicit physical units (e.g., `records`, `/ 100`, `%`).
- **Cards & Surfaces:** Rounded corners restricted to a subtle 8px border-radius; 1px solid border in `#E2E7E4`; soft ambient drop shadow (`0 1px 3px rgba(0,0,0,0.05)`).
- **Status Badges:** Pill-shaped inline badges with 4px vertical padding and 10px horizontal padding, utilizing low-saturation background tints and high-contrast text.

### 2.3 Single-Page Hierarchical Flow
Instead of hiding critical analytics behind tabs, the redesigned console presents a logical, continuous industrial monitoring flow:
1. **Header & Context Bar:** Project logo, product title ("Predictive Intelligence"), academic context, and telemetry status indicator (10 machines, 10,800 records).
2. **Executive KPI Ribbon:** 4 clean metric cards (Total Records, Monitored Machines, High-Risk Events, Average Anomaly Score) with honest, mathematically verified numbers.
3. **Primary Diagnostic Row (Side-by-Side):**
   - Left: Fleet Maintenance Risk Distribution (Clean Donut Chart in `#39845A`, `#C58A32`, `#C44F4F`).
   - Right: Machine-Wise High-Risk Telemetry Spikes (Clean horizontal/vertical bar chart in `#C44F4F` highlighting `M_009` and `M_007`).
4. **Time-Series Telemetry & Anomaly Drift:** Full-width interactive line chart showing daily anomaly scores with calibrated warning (35) and critical (65) threshold lines.
5. **Multivariate Sensor Correlation Matrix:** Dual-column scatter charts (Thermal vs. Vibration Instability and 6G Latency vs. Packet Loss) using industrial color scales.
6. **Prioritized Maintenance Inspection Queue:** Full-width actionable table showing active alerts, lead-time windows, and prescriptive maintenance actions.
7. **Temporal Degradation Heatmap:** Fleet-wide heatmap displaying progressive machine wear over the 30-day monitoring window.
8. **Methodology & Transparent Data Disclosure:** Explicit disclosure box detailing synthetic telemetry parameters, mathematical Isolation Forest principles, and 6G simulation boundaries.

---

## 3. RESPONSIVE DESIGN REQUIREMENTS
- **Desktop (>= 1280px):** Multi-column metric ribbons, side-by-side diagnostic charts, full-width alert table.
- **Laptop / Tablet (768px - 1279px):** 2x2 metric card grid, charts dynamically resize to container width (`width="stretch"`).
- **Mobile (< 768px):** Single-column stacked layout, horizontally scrollable data tables, responsive touch-friendly sidebar toggles.
