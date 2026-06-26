# Energetic Particle Radiation Environment Forecasting System for Geostationary Orbit

This system is an operational space weather forecasting toolkit designed to predict energetic electron fluxes (>2 MeV) at geostationary orbit 30–45 minutes, 6 hours, and 12 hours in advance. These predictions protect ISRO's geostationary communication and meteorological satellites (e.g. GSATs) from deep dielectric charging and electrostatic discharge anomalies.

---

## 🌌 System Architecture & Physics

The forecasting system combines physics-based features with deep learning time-series models:

1. **Burton-McPherron-O'Brien Dst Proxy:** Southward IMF component ($Bz\_south$) drives magnetic reconnection, injecting energy into the ring current. The system solves the differential equation:
   $$\frac{d(\text{Dst})}{dt} = Q(t) - \frac{\text{Dst}}{\tau}$$
   where $Q(t) \propto V_{\text{sw}} \times Bz\_south$ and $\tau = 7.7\text{ hours}$ (decay rate).
2. **Solar Wind Dynamic Pressure:** Computed as $P_{\text{dyn}} = 2 \times 10^{-6} \times N \times V_{\text{sw}}^2$ (nPa), representing magnetopause compression. Extremely high pressure causes magnetopause shadowing, leading to prompt electron flux dropouts.
3. **Temporal Fusion Transformer (TFT):** The primary forecasting model natively supports multi-horizon forecasting, includes multi-head self-attention to capture long-term sequence dependencies, and outputs probabilistic quantile predictions (10th, 50th, and 90th percentiles).
4. **Longitudinal Cross-Calibration:** Solves the longitudinal gradient between NOAA GOES (American longitudes) and ISRO GRASP (Indian longitude ~74°E) via linear regression in log-space, shifting the diurnal peaks (+10 hours local time shift).

---

## 📂 Project Directory Structure

```
C:\Users\Admin\ISRO-HACK\
├── data/
│   ├── raw/                 # CDF files (GOES, Wind, GRASP)
│   └── processed/           # Cached Parquet files and scaler
├── models/                  # Trained PyTorch/XGBoost models and calibration factors
├── plots/                   # All EDA and evaluation figures
├── requirements.txt         # Project dependencies
├── generate_mock_data.py    # High-fidelity space weather CDF generator
├── data_downloader.py       # NASA CDAWeb CDF data downloader
├── data_reader.py           # CDF reader and interpolation aligner
├── feature_engineer.py      # Feature engineering calculations
├── eda_visualizer.py        # All EDA visualization plots
├── models.py                # TFT, Seq2Seq LSTM, and XGBoost modules
├── train_xgb.py             # XGBoost training script
├── train_lstm.py            # LSTM training script
├── train_tft.py             # TFT training script
├── evaluate.py              # Test set evaluation and reliability diagrams
├── grasp_validation.py      # ISRO payload cross-calibration and comparison
└── dashboard.py             # Streamlit operational forecasting dashboard
```

---

## 🚀 Setup & Execution Guide

### 1. Initialize Environment & Install Dependencies
Ensure you are in the project folder and run:
```bash
# Verify Python version is 3.12+ (or 3.14)
python --version

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Generate Synthetic Training Data (or Download Real Data)
To populate 11 years of high-cadence CDF files for testing the end-to-end pipeline:
```bash
python generate_mock_data.py
```
*Note: To download real files from NASA, you can use `python data_downloader.py`.*

### 3. Read and Align Datasets
Read raw CDF files, align GOES and Wind timestamps via linear interpolation, and save a cached Parquet file:
```bash
python data_reader.py
```

### 4. Engineer Physics-Based Features
Compute lags, rolling windows, cyclic time markers, and the Dst proxy:
```bash
python feature_engineer.py
```

### 5. Generate Exploratory Plots
Produce heatmaps, aligned parameter panels, and superposed epoch analysis (SEA) around geomagnetic storm onsets:
```bash
python eda_visualizer.py
```
*Plots will be output to the `plots/` directory.*

### 6. Train Forecasting Models
Train and save baseline and primary architectures:
```bash
# XGBoost
python train_xgb.py

# Encoder-Decoder LSTM
python train_lstm.py

# Temporal Fusion Transformer (Primary)
python train_tft.py
```

### 7. Run Evaluation & Validation
Evaluate forecast performance (RMSE, MAE, R, HSS, Skill Score, and Reliability Diagrams) on the test split:
```bash
python evaluate.py
```
Cross-calibrate GOES forecasts against ISRO's GRASP payload data to calculate the longitudinal offset correction factor:
```bash
python grasp_validation.py
```

### 8. Run Operational Dashboard
Launch the live-style forecaster dashboard:
```bash
streamlit run dashboard.py
```
The dashboard features real-time stream simulation (including CME arrivals), live KPI metrics, interactive forecast curves with uncertainty bands, variable importance bars, and auto-generated bulletins for mission control teams.

---

## 📈 Operational Hazard Definitions
The forecast curves output three hazard zones to guide satellite operations:
- **GREEN Zone ($<10^3$ e-/cm²/s/sr):** Quiet state. No electrostatic discharge hazards.
- **YELLOW Zone ($10^3$ – $10^4$ e-/cm²/s/sr):** Warning. Moderate risk of surface/dielectric charging.
- **RED Zone ($>10^4$ e-/cm²/s/sr):** Critical storm alert. High risk of electrostatic discharge, dielectric breakdown, and spacecraft telemetry anomalies.
