import os
import pandas as pd
import numpy as np

def calculate_dst_proxy(vsw, bz, cadence_mins=5):
    """
    Computes a physical Dst index proxy using the Burton-McPherron-O'Brien relation:
    d(Dst)/dt = Q(t) - Dst / tau
    where:
    - Q(t) is the ring current injection rate, driven by the solar wind electric field:
      Q(t) = -d * (Vsw * Bz_south) if Vsw * Bz_south > threshold else 0
    - tau is the ring current decay time (~7.7 hours).
    """
    n_steps = len(vsw)
    dst_proxy = np.zeros(n_steps)
    
    # Constants
    tau_hours = 7.7
    dt_hours = cadence_mins / 60.0
    decay_factor = np.exp(-dt_hours / tau_hours)
    
    # Southward Bz
    bz_south = np.maximum(0.0, -bz)
    # Solar wind electric field proxy: Ey = Vsw * Bz_south (converted to mV/m approx)
    # Vsw is in km/s, Bz is in nT. Ey is Vsw * Bz * 1e-3.
    ey = vsw * bz_south * 1e-3
    
    # Injection coefficient
    d = -4.4  # nT/h per mV/m
    threshold = 0.5 # mV/m
    
    curr_dst = -20.0 # baseline quiet-time Dst
    for i in range(n_steps):
        ey_val = ey[i]
        if ey_val > threshold:
            injection = d * (ey_val - threshold)
        else:
            injection = 0.0
            
        # Dst decay towards quiet-time baseline (-20 nT)
        curr_dst = (curr_dst + 20.0) * decay_factor - 20.0 + injection * dt_hours
        dst_proxy[i] = curr_dst
        
    return dst_proxy

def engineer_features(data_path="data/processed/aligned_data.parquet", out_path="data/processed/features_data.parquet", cadence_mins=5):
    """
    Computes physics-based features, lags, rolling statistics, cyclic features, and log10 target.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Aligned data parquet not found at {data_path}. Run data_reader.py first.")
        
    print(f"Loading aligned data from {data_path}...")
    df = pd.read_parquet(data_path)
    
    # 1. Target log10 transform
    # goes_flux is in cm^-2 s^-1 sr^-1. Log10 makes the distribution symmetric.
    df['log_goes_flux'] = np.log10(np.clip(df['goes_flux'], 1.0, 1e7))
    
    # 2. Physics-based variables
    # Southward Bz (drives reconnection)
    df['bz_south'] = np.maximum(0.0, -df['Bz'])
    
    # Calculate Pdyn if not present, otherwise use existing
    if 'Pdyn' not in df.columns or df['Pdyn'].isnull().all():
        df['Pdyn'] = 2e-6 * df['N'] * (df['Vsw'] ** 2)
        
    # Calculate Dst proxy
    df['dst_proxy'] = calculate_dst_proxy(df['Vsw'].values, df['Bz'].values, cadence_mins)
    
    # Steps equivalents
    step_30m = int(30 / cadence_mins)
    step_1h = int(60 / cadence_mins)
    step_3h = int(180 / cadence_mins)
    step_6h = int(360 / cadence_mins)
    step_12h = int(720 / cadence_mins)
    
    # 3. Lagged features
    # IMF Bz lags
    df['bz_lag_1h'] = df['Bz'].shift(step_1h)
    df['bz_lag_3h'] = df['Bz'].shift(step_3h)
    
    # Solar wind speed lags
    df['vsw_lag_6h'] = df['Vsw'].shift(step_6h)
    df['vsw_lag_12h'] = df['Vsw'].shift(step_12h)
    
    # Target history (autoregressive inputs)
    df['log_goes_flux_lag_30m'] = df['log_goes_flux'].shift(step_30m)
    df['log_goes_flux_lag_1h'] = df['log_goes_flux'].shift(step_1h)
    df['log_goes_flux_lag_3h'] = df['log_goes_flux'].shift(step_3h)
    
    # 4. Rolling statistics (mean and std of Vsw and Bz)
    df['vsw_roll_mean_3h'] = df['Vsw'].rolling(window=step_3h).mean()
    df['vsw_roll_std_3h'] = df['Vsw'].rolling(window=step_3h).std()
    df['vsw_roll_mean_6h'] = df['Vsw'].rolling(window=step_6h).mean()
    df['vsw_roll_std_6h'] = df['Vsw'].rolling(window=step_6h).std()
    
    df['bz_roll_mean_3h'] = df['Bz'].rolling(window=step_3h).mean()
    df['bz_roll_std_3h'] = df['Bz'].rolling(window=step_3h).std()
    df['bz_roll_mean_6h'] = df['Bz'].rolling(window=step_6h).mean()
    df['bz_roll_std_6h'] = df['Bz'].rolling(window=step_6h).std()
    
    # 5. Cyclic features (sin/cos encoding of time of day and day of year)
    # Time of day
    hours = df.index.hour + df.index.minute / 60.0
    df['sin_time_of_day'] = np.sin(2 * np.pi * hours / 24.0)
    df['cos_time_of_day'] = np.cos(2 * np.pi * hours / 24.0)
    
    # Day of year
    doy = df.index.dayofyear
    df['sin_day_of_year'] = np.sin(2 * np.pi * doy / 365.25)
    df['cos_day_of_year'] = np.cos(2 * np.pi * doy / 365.25)
    
    # 6. Forward fill missing data for small gaps (up to 2h)
    # Longer gaps will naturally contain NaNs and will be skipped in sequence generation
    limit_fwd_fill = int(120 / cadence_mins) # 2 hours
    df.ffill(limit=limit_fwd_fill, inplace=True)
    
    # Save engineered features
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path)
    print(f"Engineered features shape: {df.shape}")
    print(f"Saved engineered features to: {out_path}")
    return df

if __name__ == "__main__":
    engineer_features()
