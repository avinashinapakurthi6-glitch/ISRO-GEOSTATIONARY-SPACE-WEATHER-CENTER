# 🛰️ ISRO Geostationary Space Weather Center

> **Energetic Particle Radiation Environment Forecasting System for Geostationary Orbit**

An operational space weather forecasting toolkit designed to predict energetic electron fluxes (>2 MeV) at geostationary orbit **30–45 minutes, 6 hours, and 12 hours** in advance — protecting ISRO's geostationary communication and meteorological satellites (e.g. GSATs) from deep dielectric charging and electrostatic discharge anomalies.

🌐 **Live Demo:** [isro-geostationary-space-weather-center.onrender.com](https://isro-geostationary-space-weather-center.onrender.com/)

---

## 🖥️ Frontend Dashboard (Next.js)

A fully interactive **Next.js 16** web dashboard replicating all functionality of the Streamlit operational dashboard — built with React, TypeScript, and Recharts for rich, interactive visualizations.

### Features
| Tab | Description |
|-----|-------------|
| 🛰️ **Real-Time Forecasts** | 5 KPI cards (Flux, Vsw, Bz, Pdyn, Hazard), Multi-Horizon forecast chart with quantile bands, TFT VSN Feature Importance bar chart |
| 📊 **Solar Wind Telemetry** | 24-hour ingestion history for Vsw, IMF Bz, Proton Density (N), Dynamic Pressure (Pdyn) |
| 🛡️ **Active Shielding & Trajectory** | Lorentz deflection physics simulation, Auto-Pilot / Manual shielding controls, Proactive Safety Index, Trajectory phase shift |
| 🔄 **ISRO GRASP Calibration** | GOES vs GRASP scatter calibration, Diurnal phase shift overlay, Live GSAT Impact Peak Estimator |
| 📈 **Performance & Validation** | Model evaluation metrics table (RMSE, MAE, R², Skill Score), Quantile Reliability Curve, Storm Window predictions |
| 📜 **Operational Bulletin** | Auto-generated forecast advisory bulletin with download (.txt) |

### Sidebar Controls
- **Simulation Presets** — Quiet / Moderate Storm / Severe CME
- **Interactive Sliders** — Solar Wind Speed (Vsw), IMF Bz, Proton Density (N)
- **Model Selector** — TFT, XGBoost, LSTM, Persistence (primary + overlay comparison)
- **Live Telemetry Refresh** button

### Running the Frontend
```bash
cd frontend
npm install
npm run dev
# → http://localhost:3000
```

### Frontend Tech Stack
- **Framework:** Next.js 16 (Turbopack) + React 19 + TypeScript
- **Charts:** Recharts (LineChart, BarChart, AreaChart, ScatterChart)
- **Styling:** Vanilla CSS — dark space theme (`#060913`), glassmorphism cards, Outfit + Space Grotesk fonts
- **Fonts:** Google Fonts — Outfit (body), Space Grotesk (headings/mono)

---

## 🌌 System Architecture & Physics

The forecasting system combines physics-based features with deep learning time-series models:

1. **Burton-McPherron-O'Brien Dst Proxy:** Southward IMF component ($Bz\_south$) drives magnetic reconnection, injecting energy into the ring current. The system solves the differential equation:
   $$\frac{d(\text{Dst})}{dt} = Q(t) - \frac{\text{Dst}}{\tau}$$
   where $Q(t) \propto V_{\text{sw}} \times Bz\_south$ and $\tau = 7.7\text{ hours}$ (decay rate).

2. **Solar Wind Dynamic Pressure:** Computed as $P_{\text{dyn}} = 2 \times 10^{-6} \times N \times V_{\text{sw}}^2$ (nPa), representing magnetopause compression. Extremely high pressure causes magnetopause shadowing, leading to prompt electron flux dropouts.

3. **Temporal Fusion Transformer (TFT):** The primary forecasting model natively supports multi-horizon forecasting, includes multi-head self-attention to capture long-term sequence dependencies, and outputs probabilistic quantile predictions (10th, 50th, and 90th percentiles).

4. **Longitudinal Cross-Calibration:** Solves the longitudinal gradient between NOAA GOES (American longitudes) and ISRO GRASP (Indian longitude ~74°E) via linear regression in log-space:
   $$\log_{10}(\text{GRASP}) = 0.5131 \times \log_{10}(\text{GOES}) + 0.5669$$
   Shifting the diurnal peaks by **+10 hours** local time.

---

## 📂 Project Directory Structure

```
ISRO-HACK/
├── frontend/                    # ← Next.js Interactive Dashboard
│   ├── app/
│   │   ├── components/
│   │   │   └── Dashboard.tsx    # Main dashboard (6 tabs, all charts)
│   │   ├── utils/
│   │   │   └── simulation.ts    # Space weather simulation engine
│   │   ├── globals.css          # Dark theme design system
│   │   ├── layout.tsx
│   │   └── page.tsx
│   ├── public/
│   │   └── isro-logo.png        # Official ISRO logo
│   ├── package.json
│   └── next.config.ts
│
├── data/
│   ├── raw/                     # CDF files (GOES, Wind, GRASP)
│   └── processed/               # Cached Parquet files and scaler
├── models/                      # Trained PyTorch/XGBoost models
├── plots/                       # EDA and evaluation figures
├── requirements.txt             # Python dependencies
├── generate_mock_data.py        # High-fidelity space weather CDF generator
├── data_downloader.py           # NASA CDAWeb CDF data downloader
├── data_reader.py               # CDF reader and interpolation aligner
├── feature_engineer.py          # Feature engineering calculations
├── eda_visualizer.py            # EDA visualization plots
├── models.py                    # TFT, Seq2Seq LSTM, and XGBoost modules
├── train_xgb.py                 # XGBoost training script
├── train_lstm.py                # LSTM training script
├── train_tft.py                 # TFT training script
├── evaluate.py                  # Test set evaluation and reliability diagrams
├── grasp_validation.py          # ISRO payload cross-calibration
├── dashboard.py                 # Streamlit operational dashboard
└── render.yaml                  # Render.com deployment config
```

---

## 🚀 Setup & Execution Guide

### Python Backend

#### 1. Initialize Environment & Install Dependencies
```bash
python --version   # Requires 3.12+
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

#### 2. Generate Synthetic Training Data
```bash
python generate_mock_data.py
# To download real NASA data:
python data_downloader.py
```

#### 3. Read and Align Datasets
```bash
python data_reader.py
```

#### 4. Engineer Physics-Based Features
```bash
python feature_engineer.py
```

#### 5. Generate Exploratory Plots
```bash
python eda_visualizer.py
# Output → plots/ directory
```

#### 6. Train Forecasting Models
```bash
python train_xgb.py    # XGBoost
python train_lstm.py   # Encoder-Decoder LSTM
python train_tft.py    # Temporal Fusion Transformer (Primary)
```

#### 7. Evaluate & Validate
```bash
python evaluate.py        # RMSE, MAE, R², Skill Score, Reliability Diagrams
python grasp_validation.py  # GOES–GRASP cross-calibration
```

#### 8. Run Streamlit Dashboard
```bash
streamlit run dashboard.py
```

### Next.js Frontend

```bash
cd frontend
npm install
npm run dev      # Development → http://localhost:3000
npm run build    # Production build
npm start        # Production server
```

---

## 📈 Operational Hazard Definitions

The forecast curves output three hazard zones to guide satellite operations:

| Zone | Flux Level | Status | Risk |
|------|-----------|--------|------|
| 🟢 **GREEN** | < 10³ e-/cm²/s/sr | Quiet | No electrostatic discharge hazards |
| 🟡 **YELLOW** | 10³ – 10⁴ e-/cm²/s/sr | Warning | Moderate risk of surface/dielectric charging |
| 🔴 **RED** | > 10⁴ e-/cm²/s/sr | Critical Alert | High risk of ESD, dielectric breakdown, spacecraft telemetry anomalies |

---

## 🔬 Model Performance Summary

| Model | Horizon | RMSE | MAE | Skill Score |
|-------|---------|------|-----|-------------|
| TFT | +30m | 0.142 | 0.108 | 0.61 |
| TFT | +6h | 0.271 | 0.204 | 0.54 |
| TFT | +12h | 0.342 | 0.261 | 0.48 |
| LSTM | +30m | 0.159 | 0.121 | 0.57 |
| XGBoost | +30m | 0.178 | 0.134 | 0.52 |
| Persistence | +30m | 0.218 | 0.171 | 0.00 (baseline) |

---

## 🛡️ Active Shielding Physics

The shielding simulation models Lorentz deflection of relativistic electrons (>2 MeV):

$$\eta = 1 - e^{-\gamma \sqrt{P}}$$

Where:
- $\eta$ = shielding efficiency
- $\gamma = 2.0$ (coil effectiveness factor)
- $P$ = power allocation (0–100%)
- Max dipole moment: **500 A·m²**

**Quadratic Dipole Scaling (Physical):**
$$\text{MDE} = \min\left(100,\ 25 \times (\log_{10}F - 3)^2\right)$$

---

*Built for the ISRO Space Weather Hackathon — Satellite Safeguard Center for Geostationary Orbit*
