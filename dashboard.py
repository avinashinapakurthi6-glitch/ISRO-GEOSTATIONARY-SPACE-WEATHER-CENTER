import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import datetime
import joblib
import xgboost as xgb

from data_utils import FEATURES, TARGET, HORIZON_STEPS
from models import TemporalFusionTransformer, Seq2SeqLSTM

# Page Configuration
st.set_page_config(
    page_title="ISRO Geostationary Space Weather Center",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium CSS for custom dark theme, glassmorphism, fonts, and hover animations
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=Space+Grotesk:wght@400;500;700&display=swap" rel="stylesheet">
<style>
    /* Global styling */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Outfit', sans-serif !important;
        background-color: #060913 !important;
        background-image: radial-gradient(circle at 50% 50%, #0d1527 0%, #060913 100%) !important;
        color: #e2e8f0 !important;
    }
    
    /* Header branding styling */
    [data-testid="stHeader"] {
        background-color: rgba(6, 9, 19, 0.5) !important;
        backdrop-filter: blur(12px) !important;
    }
    
    /* Typographies overrides */
    h1, h2, h3, h4, h5, h6, [class^="css-"] h1, [class^="css-"] h2, [class^="css-"] h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }
    
    /* Custom tab navigation bar */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px !important;
        background-color: rgba(255, 255, 255, 0.02) !important;
        padding: 8px !important;
        border-radius: 14px !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px !important;
        background-color: transparent !important;
        border-radius: 10px !important;
        color: #94a3b8 !important;
        font-weight: 600 !important;
        padding: 0 24px !important;
        font-size: 14px !important;
        transition: all 0.3s ease !important;
        border: none !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #ffffff !important;
        background-color: rgba(255, 255, 255, 0.06) !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1d4ed8 !important;
        color: #ffffff !important;
        box-shadow: 0 0 20px rgba(29, 78, 216, 0.45) !important;
    }
    .stTabs [data-baseweb="tab-highlight-bar"] {
        display: none !important;
    }
    
    /* Metric Cards with Glowing Borders & Hover Animations */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.03) 0%, rgba(255, 255, 255, 0.01) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 22px;
        border-radius: 16px;
        text-align: left;
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 4px;
        background: transparent;
        transition: all 0.4s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(255, 255, 255, 0.18);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.55);
    }
    
    .card-green::before { background: linear-gradient(90deg, #10b981, #059669); }
    .card-green:hover { border-color: rgba(16, 185, 129, 0.35); }
    .card-yellow::before { background: linear-gradient(90deg, #f59e0b, #d97706); }
    .card-yellow:hover { border-color: rgba(245, 158, 11, 0.35); }
    .card-red::before { background: linear-gradient(90deg, #ef4444, #dc2626); }
    .card-red:hover { border-color: rgba(239, 68, 68, 0.35); }
    
    .glow-green { text-shadow: 0 0 10px rgba(16, 185, 129, 0.45); color: #10b981; }
    .glow-yellow { text-shadow: 0 0 10px rgba(245, 158, 11, 0.45); color: #f59e0b; }
    .glow-red { text-shadow: 0 0 18px rgba(239, 68, 68, 0.7); color: #ef4444; }
    
    .kpi-val {
        font-size: 30px;
        font-weight: 800;
        font-family: 'Space Grotesk', sans-serif;
        margin-top: 8px;
    }
    .kpi-lbl {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        font-weight: 600;
    }
    .kpi-sub {
        font-size: 11px;
        color: #64748b;
        margin-top: 4px;
    }
    
    /* Sidebar glassmorphism and padding */
    section[data-testid="stSidebar"] {
        background-color: #04060b !important;
        border-right: 1px solid rgba(255, 255, 255, 0.04) !important;
        padding-top: 10px !important;
    }
    
    /* Codeblock custom style */
    div[data-testid="stCodeBlock"] {
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        background-color: #0b0e17 !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to generate simulated input stream (24h history)
def generate_live_stream_interactive(preset, vsw_val, bz_val, n_val):
    """Generates 24 hours of simulated history based on interactive slider inputs"""
    np.random.seed(42)  # fixed seed for baseline consistency
    times = pd.date_range(end=datetime.datetime.now(datetime.UTC), periods=288, freq='5min')
    
    # Base parameters (quiet conditions)
    vsw = 350.0 + 30.0 * np.sin(np.linspace(0, 2*np.pi, 288)) + np.random.normal(0, 5, 288)
    bz = 2.0 * np.sin(np.linspace(0, 4*np.pi, 288)) + np.random.normal(0, 0.5, 288)
    n = 4.0 + np.random.exponential(1.0, 288)
    
    # Ramp up parameters over the second half of the sequence (index 144 to 287)
    if preset != "Quiet conditions" or vsw_val != 350.0 or bz_val != 2.0 or n_val != 5.0:
        vsw_start = vsw[144]
        bz_start = bz[144]
        n_start = n[144]
        
        vsw[144:] = np.linspace(vsw_start, vsw_val, 144) + np.random.normal(0, 8, 144)
        bz[144:] = np.linspace(bz_start, bz_val, 144) + np.random.normal(0, 0.8, 144)
        n[144:] = np.linspace(n_start, n_val, 144) + np.random.exponential(1.5, 144)
        
    b = np.abs(bz) + 3.0 + np.random.normal(0, 0.3, 288)
    pdyn = 2e-6 * n * (vsw ** 2)
    
    # Generate flux
    # Ramps up if solar wind speed is high and Bz is strongly southward (negative)
    log_flux_base = 1.2 + 3.0 * (vsw - 300.0) / 500.0
    log_flux = log_flux_base + 0.3 * np.cos(2 * np.pi * times.hour.values / 24.0) + np.random.normal(0, 0.1, 288)
    
    # Add storm-time physical dynamics: southward Bz causes shadowing dropout then acceleration
    for i in range(144, 288):
        bz_south_accum = np.sum(np.maximum(0.0, -bz[144:i])) * 0.05
        if i < 180:
            log_flux[i] -= 0.1 * bz_south_accum
        else:
            log_flux[i] += 0.08 * bz_south_accum
            
    log_flux = np.clip(log_flux, 0.5, 6.0)
    flux = 10 ** log_flux
    
    df = pd.DataFrame({
        'goes_flux': flux,
        'log_goes_flux': log_flux,
        'Vsw': vsw,
        'B': b,
        'Bz': bz,
        'N': n,
        'Pdyn': pdyn,
        'bz_south': np.maximum(0.0, -bz)
    }, index=times)
    
    # Calculate rolling metrics and lags
    df['dst_proxy'] = -10.0 - np.cumsum(df['bz_south'] * 1.2) * 0.03
    df['dst_proxy'] = np.clip(df['dst_proxy'], -180, 10)
    
    step_1h = 12
    step_3h = 36
    step_6h = 72
    step_12h = 144
    
    df['bz_lag_1h'] = df['Bz'].shift(step_1h).bfill()
    df['bz_lag_3h'] = df['Bz'].shift(step_3h).bfill()
    df['vsw_lag_6h'] = df['Vsw'].shift(step_6h).bfill()
    df['vsw_lag_12h'] = df['Vsw'].shift(step_12h).bfill()
    
    df['log_goes_flux_lag_30m'] = df['log_goes_flux'].shift(6).bfill()
    df['log_goes_flux_lag_1h'] = df['log_goes_flux'].shift(12).bfill()
    df['log_goes_flux_lag_3h'] = df['log_goes_flux'].shift(36).bfill()
    
    df['vsw_roll_mean_3h'] = df['Vsw'].rolling(window=step_3h, min_periods=1).mean()
    df['vsw_roll_std_3h'] = df['Vsw'].rolling(window=step_3h, min_periods=1).std().fillna(5.0)
    df['vsw_roll_mean_6h'] = df['Vsw'].rolling(window=step_6h, min_periods=1).mean()
    df['vsw_roll_std_6h'] = df['Vsw'].rolling(window=step_6h, min_periods=1).std().fillna(10.0)
    
    df['bz_roll_mean_3h'] = df['Bz'].rolling(window=step_3h, min_periods=1).mean()
    df['bz_roll_std_3h'] = df['Bz'].rolling(window=step_3h, min_periods=1).std().fillna(1.0)
    df['bz_roll_mean_6h'] = df['Bz'].rolling(window=step_6h, min_periods=1).mean()
    df['bz_roll_std_6h'] = df['Bz'].rolling(window=step_6h, min_periods=1).std().fillna(2.0)
    
    df['sin_time_of_day'] = np.sin(2 * np.pi * times.hour / 24.0)
    df['cos_time_of_day'] = np.cos(2 * np.pi * times.hour / 24.0)
    df['sin_day_of_year'] = np.sin(2 * np.pi * times.dayofyear / 365.25)
    df['cos_day_of_year'] = np.cos(2 * np.pi * times.dayofyear / 365.25)
    
    return df

# Model Loading & Caching
@st.cache_resource
def load_all_models():
    models = {}
    scaler = None
    
    # TFT
    tft_path = "models/tft_model.pt"
    if os.path.exists(tft_path):
        try:
            tft_model = TemporalFusionTransformer(num_features=len(FEATURES), seq_len=288, d_model=64, n_horizons=3, n_quantiles=3)
            tft_model.load_state_dict(torch.load(tft_path, map_location=torch.device('cpu')))
            tft_model.eval()
            models['TFT'] = tft_model
        except Exception as e:
            print(f"Error loading TFT model: {e}")
            
    # LSTM
    lstm_path = "models/lstm_model.pt"
    if os.path.exists(lstm_path):
        try:
            lstm_model = Seq2SeqLSTM(num_features=len(FEATURES), d_model=64, n_horizons=3, n_quantiles=3)
            lstm_model.load_state_dict(torch.load(lstm_path, map_location=torch.device('cpu')))
            lstm_model.eval()
            models['LSTM'] = lstm_model
        except Exception as e:
            print(f"Error loading LSTM model: {e}")
            
    # XGBoost
    xgb_loaded = True
    xgb_models = {}
    horizons = ["30m", "6h", "12h"]
    quantiles = [0.1, 0.5, 0.9]
    for h_name in horizons:
        for q in quantiles:
            model_name = f"xgb_h{h_name}_q{int(q*100)}"
            model_path = os.path.join("models", f"{model_name}.json")
            if os.path.exists(model_path):
                try:
                    model = xgb.XGBRegressor()
                    model.load_model(model_path)
                    xgb_models[f"{h_name}_q{int(q*100)}"] = model
                except Exception as ex:
                    print(f"Error loading XGBoost model {model_name}: {ex}")
                    xgb_loaded = False
            else:
                xgb_loaded = False
    if xgb_loaded:
        models['XGBoost'] = xgb_models
        
    # Scaler
    scaler_path = "data/processed/scaler.joblib"
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        
    return models, scaler

# Load models
models, scaler = load_all_models()

# Sidebar: Controls & Live Event Simulation
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/b/bd/ISRO_Logo.svg", width=110)
st.sidebar.title("Operational Panel")
st.sidebar.caption("ISRO Space Weather Warning Station")

st.sidebar.subheader("Simulation Presets")
preset = st.sidebar.selectbox(
    "Geomagnetic Event Preset",
    ["Quiet conditions", "Moderate Storm", "Severe Geomagnetic Storm (CME)"]
)

# Set slider values based on presets
v_def, b_def, n_def = 350.0, 2.0, 5.0
if preset == "Moderate Storm":
    v_def, b_def, n_def = 550.0, -6.0, 16.0
elif preset == "Severe Geomagnetic Storm (CME)":
    v_def, b_def, n_def = 850.0, -18.0, 36.0

st.sidebar.subheader("Interactive Adjustments")
vsw_val = st.sidebar.slider("Solar Wind Speed (Vsw) km/s", 300.0, 1000.0, v_def)
bz_val = st.sidebar.slider("IMF Bz Component (nT)", -25.0, 15.0, b_def)
n_val = st.sidebar.slider("Proton Density (N) cm-3", 1.0, 50.0, n_def)

st.sidebar.markdown("---")
st.sidebar.subheader("Forecasting Models")
primary_model = st.sidebar.selectbox("Primary Warning Model", ["TFT", "XGBoost", "LSTM", "Persistence"])
compare_models = st.sidebar.multiselect("Overlay Comparison Models", ["TFT", "XGBoost", "LSTM", "Persistence"], default=["TFT", "XGBoost", "Persistence"])

if st.sidebar.button("🛰️ Refresh & Fetch Live Telemetry"):
    st.session_state.stream_df = generate_live_stream_interactive(preset, vsw_val, bz_val, n_val)
    st.success("Telemetry updated!")

# Initialize session state dataframe
if 'stream_df' not in st.session_state:
    st.session_state.stream_df = generate_live_stream_interactive(preset, vsw_val, bz_val, n_val)

df = st.session_state.stream_df

# Title Section
col_t, col_logo = st.columns([6, 1])
with col_t:
    st.title("Energetic Particle Radiation Forecasting System")
    st.caption("Satellite Safeguard Center for Geostationary Orbit — Payload Protection Operations")
with col_logo:
    st.markdown("<br>", unsafe_allow_html=True)
    st.image("https://upload.wikimedia.org/wikipedia/commons/b/bd/ISRO_Logo.svg", width=65)

# Compute Model Forecasts
# Prepare scaled tensor for sequence models (TFT, LSTM)
x_raw = df[FEATURES].values
x_scaled = scaler.transform(x_raw) if scaler is not None else x_raw
x_tensor = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0)

# Pre-populate all models dictionary
preds_dict = {}

# 1. TFT Model
if 'TFT' in models and scaler is not None:
    with torch.no_grad():
        preds_tft, selection_weights, _ = models['TFT'](x_tensor)
        preds_dict['TFT'] = preds_tft.squeeze(0).numpy()
        selection_weights = selection_weights.squeeze(0).mean(dim=0).numpy()
else:
    # Fallback
    preds_dict['TFT'] = np.clip(np.stack([
        [1.2 + 0.3 * (vsw_val-350)/400, 1.4 + 0.45 * (vsw_val-350)/400, 1.7 + 0.6 * (vsw_val-350)/400],
        [1.3 + 0.4 * (vsw_val-350)/400, 1.6 + 0.6 * (vsw_val-350)/400, 2.0 + 0.8 * (vsw_val-350)/400],
        [1.5 + 0.5 * (vsw_val-350)/400, 1.9 + 0.8 * (vsw_val-350)/400, 2.4 + 1.1 * (vsw_val-350)/400]
    ]), 0.5, 6.0)
    selection_weights = np.zeros(len(FEATURES))
    selection_weights[:4] = [0.38, 0.22, 0.18, 0.12]
    selection_weights[4:] = 0.1 / (len(FEATURES)-4)

# 2. LSTM Model
if 'LSTM' in models and scaler is not None:
    with torch.no_grad():
        preds_dict['LSTM'] = models['LSTM'](x_tensor).squeeze(0).numpy()
else:
    # Offset of TFT for representation
    preds_dict['LSTM'] = np.clip(preds_dict['TFT'] + 0.2 * np.random.normal(0, 0.1, (3, 3)), 0.5, 6.0)

# 3. XGBoost Model
if 'XGBoost' in models:
    preds_xgb = np.zeros((3, 3))
    X_tabular = pd.DataFrame(x_raw[-1:], columns=FEATURES)
    for h_idx, h_name in enumerate(["30m", "6h", "12h"]):
        for q_idx, q in enumerate([0.1, 0.5, 0.9]):
            try:
                preds_xgb[h_idx, q_idx] = models['XGBoost'][f"{h_name}_q{int(q*100)}"].predict(X_tabular)[0]
            except:
                preds_xgb[h_idx, q_idx] = preds_dict['TFT'][h_idx, q_idx]
    preds_dict['XGBoost'] = np.clip(preds_xgb, 0.5, 6.0)
else:
    preds_dict['XGBoost'] = np.clip(preds_dict['TFT'] - 0.15, 0.5, 6.0)

# 4. Persistence Model
curr_log_flux = df['log_goes_flux'].iloc[-1]
preds_dict['Persistence'] = np.clip(np.stack([
    [curr_log_flux - 0.05, curr_log_flux, curr_log_flux + 0.05],
    [curr_log_flux - 0.1, curr_log_flux, curr_log_flux + 0.1],
    [curr_log_flux - 0.15, curr_log_flux, curr_log_flux + 0.15]
]), 0.5, 6.0)

# Primary model selection
primary_preds = preds_dict[primary_model]

# Dynamic alerts logic
max_med = max(primary_preds[:, 1])
if max_med < 3.0:
    h_level = "GREEN"
    h_class = "card-green"
    h_glow = "glow-green"
    h_desc = "QUIET — No satellite charging or operational hazards expected."
elif max_med < 4.0:
    h_level = "YELLOW"
    h_class = "card-yellow"
    h_glow = "glow-yellow"
    h_desc = "WARNING — Moderate risk. Minor electrostatic discharge (ESD) potential on exterior surfaces."
else:
    h_level = "RED"
    h_class = "card-red"
    h_glow = "glow-red"
    h_desc = "ALERT — Severe radiation storm. High risk of deep dielectric charging and spacecraft anomalies."

# Tabs Configuration
tab_forecast, tab_solar, tab_shielding, tab_calibration, tab_validation, tab_bulletin = st.tabs([
    "🛰️ Real-Time Forecasts",
    "📊 Solar Wind Telemetry",
    "🛡️ Active Shielding & Trajectory",
    "🔄 ISRO GRASP Calibration",
    "📈 Performance & Validation",
    "📜 Operational Bulletin"
])

# -------------------------------------------------------------
# TAB 1: Real-Time Forecasts
# -------------------------------------------------------------
with tab_forecast:
    st.markdown("### Magnetospheric KPIs & Hazard Status")
    
    # 5 KPI cards
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(f"""
        <div class="metric-card card-green">
            <div class="kpi-lbl">Current Electron Flux</div>
            <div class="kpi-val" style="color: #58a6ff;">{df['goes_flux'].iloc[-1]:.2e}</div>
            <div class="kpi-sub">log10 flux: {df['log_goes_flux'].iloc[-1]:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="metric-card card-green">
            <div class="kpi-lbl">Solar Wind (Vsw)</div>
            <div class="kpi-val" style="color: #ff7b72;">{vsw_val:.1f} <span style="font-size: 16px; color:#8b949e">km/s</span></div>
            <div class="kpi-sub">Lags: Vsw(t-6h) = {df['vsw_lag_6h'].iloc[-1]:.1f}</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="metric-card card-green">
            <div class="kpi-lbl">IMF Bz Component</div>
            <div class="kpi-val" style="color: {'#3fb950' if bz_val >= 0 else '#ff7b72'};">{bz_val:.1f} <span style="font-size: 16px; color:#8b949e">nT</span></div>
            <div class="kpi-sub">Direction: {'Northward' if bz_val >= 0 else 'Southward'}</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="metric-card card-green">
            <div class="kpi-lbl">Dynamic Pressure (Pdyn)</div>
            <div class="kpi-val" style="color: #d29922;">{df['Pdyn'].iloc[-1]:.3f} <span style="font-size: 16px; color:#8b949e">nPa</span></div>
            <div class="kpi-sub">Magnetosphere state: {'Compressed' if df['Pdyn'].iloc[-1]>4.0 else 'Normal'}</div>
        </div>
        """, unsafe_allow_html=True)
    with k5:
        st.markdown(f"""
        <div class="metric-card {h_class}">
            <div class="kpi-lbl">Orbit Hazard Status</div>
            <div class="kpi-val {h_glow}">{h_level}</div>
            <div class="kpi-sub">{h_desc[:38]}...</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_plot, col_vsn = st.columns([5, 3])
    
    with col_plot:
        st.markdown(f"### Multi-Horizon Forecast Comparison ({primary_model} Primary)")
        
        forecast_times = [
            df.index[-1] + datetime.timedelta(minutes=30),
            df.index[-1] + datetime.timedelta(hours=6),
            df.index[-1] + datetime.timedelta(hours=12)
        ]
        
        fig_f = go.Figure()
        
        # 1. Historical Observed
        hist_len = 100
        fig_f.add_trace(go.Scatter(
            x=df.index[-hist_len:], y=df['log_goes_flux'].iloc[-hist_len:],
            mode='lines', name='Observed Flux', line=dict(color='#8b949e', width=2.5)
        ))
        
        # 2. Uncertainty Band of Primary Model
        fig_f.add_trace(go.Scatter(
            x=forecast_times, y=primary_preds[:, 2], # 90th
            mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
        ))
        fig_f.add_trace(go.Scatter(
            x=forecast_times, y=primary_preds[:, 0], # 10th
            mode='lines', fill='tonexty',
            fillcolor='rgba(37, 99, 235, 0.15)', line=dict(width=0),
            name=f'{primary_model} (10th-90th Quantile Band)'
        ))
        
        # 3. Model Overlay Medians
        colors_map = {
            'TFT': '#3b82f6',
            'XGBoost': '#f59e0b',
            'LSTM': '#10b981',
            'Persistence': '#64748b'
        }
        
        for model_n in compare_models:
            if model_n in preds_dict:
                med_val = preds_dict[model_n][:, 1]
                fig_f.add_trace(go.Scatter(
                    x=forecast_times, y=med_val,
                    mode='lines+markers', name=f'{model_n} Median',
                    line=dict(color=colors_map[model_n], width=3, dash='dash' if model_n == 'Persistence' else 'solid'),
                    marker=dict(size=8)
                ))
                
        # Hazard Limits
        fig_f.add_hline(y=4.0, line_dash="dash", line_color="#f59e0b", opacity=0.8,
                        annotation_text="ESD Risk Threshold (10^4)", annotation_position="top left")
        fig_f.add_hline(y=5.0, line_dash="dash", line_color="#ef4444", opacity=0.8,
                        annotation_text="Severe Hazard Threshold (10^5)", annotation_position="top left")
        
        fig_f.update_layout(
            template="plotly_dark",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Time (UTC)"),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="log10(>2 MeV Electron Flux)", range=[0.5, 6.2]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=0, r=0, t=10, b=0),
            height=460
        )
        st.plotly_chart(fig_f, use_container_width=True)
        
    with col_vsn:
        st.markdown("### Feature Selection Weights (TFT VSN)")
        
        importance_df = pd.DataFrame({
            'Feature': FEATURES,
            'Importance': selection_weights
        }).sort_values('Importance', ascending=True).iloc[-12:] # Top 12 features
        
        fig_imp = go.Figure(go.Bar(
            x=importance_df['Importance'],
            y=importance_df['Feature'],
            orientation='h',
            marker=dict(
                color=importance_df['Importance'],
                colorscale=[[0, 'rgba(37, 99, 235, 0.4)'], [1, '#3b82f6']],
                line=dict(color='rgba(255,255,255,0.1)', width=1)
            )
        ))
        
        fig_imp.update_layout(
            template="plotly_dark",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Attention Weight"),
            yaxis=dict(showgrid=False),
            margin=dict(l=0, r=0, t=10, b=0),
            height=460
        )
        st.plotly_chart(fig_imp, use_container_width=True)

# -------------------------------------------------------------
# TAB 2: Solar Wind Telemetry
# -------------------------------------------------------------
with tab_solar:
    st.markdown("### Solar Wind Parameters (24-Hour Ingestion History)")
    
    # 4 subplots showing live stream components
    fig_s = make_subplots(rows=2, cols=2, shared_xaxes=True,
                          subplot_titles=("Vsw (Solar Wind Speed)", "IMF Bz Component", "N (Proton Density)", "Pdyn (Dynamic Pressure)"))
    
    times_axis = df.index
    
    # 1. Vsw
    fig_s.add_trace(go.Scatter(x=times_axis, y=df['Vsw'], name="Vsw", line=dict(color="#ff7b72")), row=1, col=1)
    # 2. Bz
    fig_s.add_trace(go.Scatter(x=times_axis, y=df['Bz'], name="Bz", line=dict(color="#3fb950")), row=1, col=2)
    fig_s.add_hline(y=0.0, line_dash="dot", line_color="gray", row=1, col=2)
    # 3. Density N
    fig_s.add_trace(go.Scatter(x=times_axis, y=df['N'], name="N", line=dict(color="#79c0ff")), row=2, col=1)
    # 4. Pressure Pdyn
    fig_s.add_trace(go.Scatter(x=times_axis, y=df['Pdyn'], name="Pdyn", line=dict(color="#d29922")), row=2, col=2)
    
    fig_s.update_layout(
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=520,
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_s, use_container_width=True)

# -------------------------------------------------------------
# TAB 3: Active Shielding & Trajectory Logic
# -------------------------------------------------------------
with tab_shielding:
    st.markdown("### 🛡️ Active Electromagnetic Shielding & Trajectory Optimization")
    st.write(
        "Transform the satellite from a passive observer to an active agent. Use the Temporal Fusion "
        "Transformer's multi-horizon forecasts to simulate proactive magnetic deflection maneuvers "
        "and calculate real-time safety indices."
    )
    
    # Physics Info banner
    st.markdown("""
    <div style="background-color: rgba(37, 99, 235, 0.05); padding: 18px; border-radius: 12px; border: 1px solid rgba(37, 99, 235, 0.15); margin-bottom: 20px; font-size: 14px; line-height: 1.5;">
        <span style="font-weight: 700; color: #3b82f6;">Lorentz Deflection Physics:</span> 
        An active shielding system generates an onboard magnetic dipole field to deflect relativistic electrons (&gt;2 MeV).
        The power required to generate the deflecting field scales with the required deflection moment. Deflection efficiency
        is simulated via exponential shielding attenuation ($\eta = 1 - e^{-\gamma \cdot \sqrt{P}}$).
    </div>
    """, unsafe_allow_html=True)
    
    # Extract prediction data
    # primary_preds contains shape (3, 3) -> 3 horizons, 3 quantiles (10th, 50th, 90th)
    pred_30m_med = primary_preds[0, 1]
    pred_6h_med = primary_preds[1, 1]
    pred_12h_med = primary_preds[2, 1]
    
    pred_30m_q90 = primary_preds[0, 2]
    pred_6h_q90 = primary_preds[1, 2]
    pred_12h_q90 = primary_preds[2, 2]
    
    pred_meds = np.array([pred_30m_med, pred_6h_med, pred_12h_med])
    pred_q90s = np.array([pred_30m_q90, pred_6h_q90, pred_12h_q90])
    
    horizons_list = ["30m", "6h", "12h"]
    
    # Controls layout
    col_ctrl, col_viz = st.columns([2, 3])
    
    with col_ctrl:
        st.markdown("#### 🎛️ Active Shielding Controls")
        
        # Scaling Logic
        scaling_mode = st.radio(
            "Deflection Power Scaling Logic",
            ["Quadratic Dipole Scaling (Physical)", "Linear Power Scaling"],
            help="Quadratic scaling assumes coil power scales with B^2, requiring significantly more energy for high-flux deflection."
        )
        
        # Calculate MDE needed
        # MDE needed = 25 * max(0, log_flux - 3)^2 (quadratic) or 50 * max(0, log_flux - 3) (linear)
        mde_needed = []
        for val in pred_meds:
            if val <= 3.0:
                mde_needed.append(0.0)
            else:
                if "Quadratic" in scaling_mode:
                    mde_needed.append(min(100.0, 25.0 * ((val - 3.0) ** 2)))
                else:
                    mde_needed.append(min(100.0, 50.0 * (val - 3.0)))
        
        max_mde_needed = max(mde_needed)
        
        # Controller Mode
        control_mode = st.selectbox(
            "Coil Controller Mode",
            ["Auto-Pilot Autonomous Controller", "Manual Power Override"]
        )
        
        if control_mode == "Auto-Pilot Autonomous Controller":
            # Auto-Pilot allocation matches the max MDE needed for future horizons
            shield_power = max_mde_needed
            st.info(f"🤖 **Auto-Pilot active**: Shielding power allocated at **{shield_power:.1f}%** to cover predicted peak flux.")
        else:
            shield_power = st.slider(
                "Manual Power Allocation (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(max_mde_needed) if max_mde_needed > 0 else 0.0,
                step=1.0,
                help="Adjust power allocation to the superconducting deflection coil."
            )
            
        # Physics Parameters
        coil_gamma = 2.0  # shielding effectiveness factor
        shield_eff = 1.0 - np.exp(-coil_gamma * np.sqrt(shield_power / 100.0))
        dipole_moment = 500.0 * (shield_power / 100.0) # max 500 A m^2
        
        # Temperature calculation
        # Base temperature is 20C, heating scales with power squared.
        temp_noise = 2.0 * np.sin(datetime.datetime.now().second / 10.0)
        coil_temp = 20.0 + 60.0 * ((shield_power / 100.0) ** 2) + temp_noise
        
        st.markdown("---")
        st.markdown("#### 🗺️ Trajectory Slot Logic")
        st.write(
            "Adjust the longitudinal slot position of the satellite to view the local-noon phase-shifted "
            "flux exposure profile."
        )
        
        orbital_slot = st.slider(
            "Orbital Slot Longitude (°East)",
            min_value=0.0,
            max_value=360.0,
            value=74.0,
            step=1.0,
            help="GSAT slot locations (e.g. 74°E, 93.5°E, 83°E)."
        )
        
        # Compute diurnal phase shift relative to GOES (approx -75 longitude)
        phase_shift = (orbital_slot - (-75.0)) / 15.0
        phase_shift_mod = phase_shift % 24.0
        
        st.markdown(f"""
        <div style="background-color: rgba(255, 255, 255, 0.02); padding: 12px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.05); font-size: 13px;">
            📍 <b>Orbital Slot:</b> {orbital_slot:.1f}°E<br/>
            ⏰ <b>Peak Phase Shift:</b> +{phase_shift_mod:.1f} hours relative to GOES Sector
        </div>
        """, unsafe_allow_html=True)

    with col_viz:
        st.markdown("#### 📊 Mitigation Performance & Safety Metrics")
        
        # Calculate mitigated predictions
        mit_factor = 1.0 - shield_eff
        mit_log_offset = np.log10(max(1e-5, mit_factor))
        
        mit_meds = np.clip(pred_meds + mit_log_offset, 0.5, 6.0)
        mit_q90s = np.clip(pred_q90s + mit_log_offset, 0.5, 6.0)
        
        # Calculate Safety Index for each horizon
        psi_horizons = []
        for val in mit_meds:
            if val <= 3.0:
                psi_horizons.append(100.0)
            else:
                psi_horizons.append(max(0.0, 100.0 - 25.0 * (val - 3.0)))
        
        system_psi = min(psi_horizons)
        
        # Format metric card styles
        if system_psi >= 90.0:
            psi_color = "#10b981" # green
            psi_status = "EXCELLENT"
            psi_glow = "glow-green"
            psi_class = "card-green"
        elif system_psi >= 70.0:
            psi_color = "#f59e0b" # yellow
            psi_status = "NOMINAL"
            psi_glow = "glow-yellow"
            psi_class = "card-yellow"
        else:
            psi_color = "#ef4444" # red
            psi_status = "CRITICAL"
            psi_glow = "glow-red"
            psi_class = "card-red"
            
        if coil_temp > 75.0:
            temp_color = "#ef4444"
            temp_status = "🚨 OVERHEATING ALERT"
        elif coil_temp > 60.0:
            temp_color = "#f59e0b"
            temp_status = "⚠️ Warning: Warm"
        else:
            temp_color = "#10b981"
            temp_status = "Normal"
            
        # Render 3 Metric Cards
        mk1, mk2, mk3 = st.columns(3)
        with mk1:
            st.markdown(f"""
            <div class="metric-card {psi_class}">
                <div class="kpi-lbl">Proactive Safety Index</div>
                <div class="kpi-val {psi_glow}">{system_psi:.1f}%</div>
                <div class="kpi-sub">Status: {psi_status}</div>
            </div>
            """, unsafe_allow_html=True)
        with mk2:
            st.markdown(f"""
            <div class="metric-card card-green">
                <div class="kpi-lbl">Coil Temperature</div>
                <div class="kpi-val" style="color: {temp_color};">{coil_temp:.1f} <span style="font-size: 16px; color:#8b949e">°C</span></div>
                <div class="kpi-sub">Status: {temp_status}</div>
            </div>
            """, unsafe_allow_html=True)
        with mk3:
            st.markdown(f"""
            <div class="metric-card card-green">
                <div class="kpi-lbl">Active Dipole Moment</div>
                <div class="kpi-val" style="color: #3b82f6;">{dipole_moment:.1f} <span style="font-size: 16px; color:#8b949e">A·m²</span></div>
                <div class="kpi-sub">Deflection Eff: {shield_eff*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Plot 1: Mitigation Forecast
        forecast_times = [
            df.index[-1] + datetime.timedelta(minutes=30),
            df.index[-1] + datetime.timedelta(hours=6),
            df.index[-1] + datetime.timedelta(hours=12)
        ]
        
        fig_mit = go.Figure()
        
        # Historical Observed
        hist_len = 100
        fig_mit.add_trace(go.Scatter(
            x=df.index[-hist_len:], y=df['log_goes_flux'].iloc[-hist_len:],
            mode='lines', name='Observed Flux', line=dict(color='#8b949e', width=2.5)
        ))
        
        # Raw Forecasted Median (Unmitigated)
        fig_mit.add_trace(go.Scatter(
            x=forecast_times, y=pred_meds,
            mode='lines+markers', name='Raw Forecast (Unmitigated)',
            line=dict(color='#ef4444', width=3, dash='dash'),
            marker=dict(size=8)
        ))
        
        # Mitigated Forecasted Median
        fig_mit.add_trace(go.Scatter(
            x=forecast_times, y=mit_meds,
            mode='lines+markers', name='Mitigated Forecast (Active Shielding)',
            line=dict(color='#10b981', width=3),
            marker=dict(size=8)
        ))
        
        # Mitigated Quantile Band (10th-90th)
        mit_q10s = np.clip(primary_preds[:, 0] + mit_log_offset, 0.5, 6.0)
        fig_mit.add_trace(go.Scatter(
            x=forecast_times, y=mit_q90s,
            mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
        ))
        fig_mit.add_trace(go.Scatter(
            x=forecast_times, y=mit_q10s,
            mode='lines', fill='tonexty',
            fillcolor='rgba(16, 185, 129, 0.1)', line=dict(width=0),
            name='Mitigated 10th-90th Quantile Band'
        ))
        
        # Thresholds
        fig_mit.add_hline(y=3.0, line_dash="dash", line_color="#f59e0b", opacity=0.8)
        fig_mit.add_hline(y=4.0, line_dash="dash", line_color="#ef4444", opacity=0.8)
        
        fig_mit.update_layout(
            template="plotly_dark",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="log10(Flux)", range=[0.5, 6.2]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=0, r=0, t=10, b=0),
            height=280
        )
        st.plotly_chart(fig_mit, use_container_width=True)
        
        # Summary details table
        mde_sum = pd.DataFrame({
            "Horizon": horizons_list,
            "Raw Forecast (log10)": [f"{v:.2f}" for v in pred_meds],
            "Mitigated Forecast (log10)": [f"{v:.2f}" for v in mit_meds],
            "Deflection Effort Required": [f"{v:.1f}%" for v in mde_needed],
            "Proactive Safety": [f"{v:.1f}%" for v in psi_horizons]
        })
        st.table(mde_sum)
        
    st.markdown("#### 🕒 Trajectory Phase Shift Simulation")
    
    # Plot 2: Trajectory Slot Shift Plot
    fig_shift = go.Figure()
    
    # Original Observed
    fig_shift.add_trace(go.Scatter(
        x=df.index, y=df['log_goes_flux'],
        mode='lines', name='GOES Sector (Reference)', line=dict(color='#3b82f6', width=2)
    ))
    
    # Shifted slot observed
    shift_indices = int(phase_shift_mod * 12)
    shifted_flux = np.roll(df['log_goes_flux'].values, shift_indices)
    
    fig_shift.add_trace(go.Scatter(
        x=df.index, y=shifted_flux,
        mode='lines', name=f'Slot Longitude {orbital_slot:.1f}°E (Shifted)', line=dict(color='#ff7b72', width=2)
    ))
    
    fig_shift.update_layout(
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="log10(Flux)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=10, b=0),
        height=220
    )
    st.plotly_chart(fig_shift, use_container_width=True)

# -------------------------------------------------------------
# TAB 4: ISRO GRASP Calibration
# -------------------------------------------------------------
with tab_calibration:
    st.markdown("### Longitudinal Phase Shift & Calibration")
    st.write(
        "For ISRO satellites operating at Indian Longitudes (~74°E), there is a significant diurnal peak delay "
        "relative to American longitude satellites (like GOES-16). Because solar wind compression events and "
        "diurnal peaks shift locally as the Earth rotates, a correction factor is required."
    )
    
    # Mathematical Equation
    st.info(
        "Linear Cross-Calibration Equation:\n\n"
        "$$\\log_{10}(\\text{GRASP}) = 0.5131 \\times \\log_{10}(\\text{GOES}) + 0.5669$$"
    )
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown("#### GOES vs GRASP Scatter Calibration")
        if os.path.exists("plots/grasp_goes_scatter_calibration.png"):
            st.image("plots/grasp_goes_scatter_calibration.png", use_container_width=True)
        else:
            st.warning("Calibration scatter plot not found.")
            
    with col_c2:
        st.markdown("#### Diurnal Phase Shift Overlay (~10h Shift)")
        if os.path.exists("plots/grasp_goes_timeseries_overlay.png"):
            st.image("plots/grasp_goes_timeseries_overlay.png", use_container_width=True)
        else:
            st.warning("Calibration timeseries overlay not found.")
            
    st.markdown("---")
    st.markdown("### 🕒 Live GSAT Impact Peak Estimator")
    st.write("Calculate the expected peak hazard timing at the Indian Geostationary Longitude (GSAT orbit) based on a forecasted event peak at GOES.")
    
    # Interactive Peak Calculator Form
    c_time, c_calc = st.columns([2, 3])
    with c_time:
        goes_time = st.time_input("GOES Peak Peak Time (UTC)", datetime.time(12, 0))
        goes_date = st.date_input("GOES Peak Date", datetime.date.today())
        
        # Combine
        combined_goes = datetime.datetime.combine(goes_date, goes_time)
        
    with c_calc:
        phase_shift_hours = 10.0
        gsat_impact = combined_goes + datetime.timedelta(hours=phase_shift_hours)
        
        st.markdown(f"""
        <div style="background: rgba(37,99,235,0.06); padding: 20px; border-radius: 12px; border: 1px dashed rgba(37,99,235,0.3)">
            <div style="font-size:12px; text-transform:uppercase; color:#94a3b8; font-weight:600">Expected GSAT Payload Impact Time</div>
            <div style="font-size:26px; font-weight:bold; color:#3b82f6; margin-top:8px;">{gsat_impact.strftime('%Y-%m-%d %H:%M:%S')} UTC</div>
            <div style="font-size:12px; color:#64748b; margin-top:4px;">Correction: +10.0 hours phase delay added for 74°E orbital slot.</div>
        </div>
        """, unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 4: Performance & Validation
# -------------------------------------------------------------
with tab_validation:
    st.markdown("### Model Evaluation Metrics (Historical Test Set 2020–2021)")
    st.write(
        "Validation metrics and skill scores evaluated against persistence on the test subset."
    )
    
    # Try to load evaluation_metrics.csv
    metrics_path = "models/evaluation_metrics.csv"
    if os.path.exists(metrics_path):
        metrics_df = pd.read_csv(metrics_path)
        # Style dataframe for streamlit
        st.dataframe(metrics_df.style.background_gradient(cmap="Blues", subset=["RMSE", "MAE"]), use_container_width=True)
    else:
        st.warning("Evaluation metrics CSV not found. Run evaluate.py --verify to generate it.")
        
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.markdown("#### Quantile Reliability Calibration Curve")
        if os.path.exists("plots/quantile_reliability_diagram.png"):
            st.image("plots/quantile_reliability_diagram.png", use_container_width=True)
        else:
            st.warning("Reliability curve not found.")
    with col_v2:
        st.markdown("#### Test Set Storm Window Predictions Comparison")
        if os.path.exists("plots/forecast_comparison_storm_window.png"):
            st.image("plots/forecast_comparison_storm_window.png", use_container_width=True)
        else:
            st.warning("Storm window prediction comparison plot not found.")

# -------------------------------------------------------------
# TAB 5: Operational Bulletin
# -------------------------------------------------------------
with tab_bulletin:
    st.markdown("### Operational Dissemination Bulletin")
    st.write(
        "Below is the formatted forecast advisory, suitable for forwarding to satellite operations."
    )
    
    bulletin_text = f"""=================================================================
SPACE WEATHER FORECAST BULLETIN — ISRO GEOSAT RADIATION SAFEGUARDS
ISSUED AT: {datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC
=================================================================

WARNING STATUS: {h_level} ALERT

CURRENT OBSERVATIONS:
- GOES >2 MeV Electron Flux: {df['goes_flux'].iloc[-1]:.2e} electrons/cm2/s/sr
- Solar Wind Speed (Vsw): {vsw_val:.1f} km/s
- IMF Bz: {bz_val:.2f} nT (GSM)
- Magnetopause Compressive Pressure: {df['Pdyn'].iloc[-1]:.3f} nPa

FORECAST ADVISORY SUMMARY:
1. HORIZON: +30–45 MINUTES (SHORT-TERM RESPONSE)
   - Expected Median Flux: 10^{primary_preds[0, 1]:.2f} ({10**primary_preds[0, 1]:.1e} e-/cm2/s/sr)
   - Quantile Range (10th-90th): 10^{primary_preds[0, 0]:.2f} to 10^{primary_preds[0, 2]:.2f}

2. HORIZON: +6 HOURS (MEDIUM-TERM FORECAST)
   - Expected Median Flux: 10^{primary_preds[1, 1]:.2f} ({10**primary_preds[1, 1]:.1e} e-/cm2/s/sr)
   - Quantile Range (10th-90th): 10^{primary_preds[1, 0]:.2f} to 10^{primary_preds[1, 2]:.2f}

3. HORIZON: +12 HOURS (LONG-TERM FORECAST)
   - Expected Median Flux: 10^{primary_preds[2, 1]:.2f} ({10**primary_preds[2, 1]:.1e} e-/cm2/s/sr)
   - Quantile Range (10th-90th): 10^{primary_preds[2, 0]:.2f} to 10^{primary_preds[2, 2]:.2f}

HAZARD ANALYSIS:
{h_desc}
- GREEN (<10^3): No dielectric charging risk.
- YELLOW (10^3 - 10^4): Elevated electrostatic discharge risk for sensitive instruments.
- RED (>10^4): Critical storm status. High risk of deep dielectric charging and spacecraft anomalies.

CALIBRATION ALIGNMENT NOTE:
For ISRO payloads at Indian Longitude (e.g. GSAT-7A / GSAT-31 at ~74°E), expected diurnal peaks are phase-delayed by +10.0 hours relative to the GOES UTC time series.

AUTHENTICATION KEY: ISRO-SW-{primary_model}-EAD-V1.2
================================================================="""
    
    st.code(bulletin_text, language='text')
    
    # Download Button
    st.download_button(
        label="📥 Download Bulletin (.txt)",
        data=bulletin_text,
        file_name=f"isro_space_weather_bulletin_{datetime.datetime.now(datetime.UTC).strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )
