import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_eda_plots(data_path="data/processed/features_data.parquet", out_dir="plots"):
    """
    Generates all requested EDA plots and saves them as PNG files.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Feature data parquet not found at {data_path}. Run feature_engineer.py first.")
        
    print(f"Loading feature data from {data_path}...")
    df = pd.read_parquet(data_path)
    os.makedirs(out_dir, exist_ok=True)
    
    # Set style
    sns.set_theme(style="darkgrid")
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})
    
    # Define hazard levels
    GREEN_LIMIT = 1e3
    YELLOW_LIMIT = 1e4
    RED_LIMIT = 1e5
    
    # -------------------------------------------------------------
    # Plot 1: Time-Series of Electron Flux with Storm-Time Annotations (Dst proxy)
    # -------------------------------------------------------------
    print("Generating Plot 1: Time Series of Electron Flux & Dst Proxy...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # Let's take a 30-day subset for high-resolution visualization of storms
    subset_df = df.iloc[-int(30 * 24 * 12):] # Last 30 days
    
    ax1.plot(subset_df.index, subset_df['goes_flux'], color='royalblue', label='GOES >2 MeV Flux', linewidth=1.5)
    ax1.set_yscale('log')
    ax1.axhline(YELLOW_LIMIT, color='gold', linestyle='--', linewidth=1.5, label='Yellow Alert (10^4)')
    ax1.axhline(RED_LIMIT, color='red', linestyle='--', linewidth=1.5, label='Red Alert (10^5)')
    ax1.set_ylabel("Flux (e-/cm²/s/sr)")
    ax1.set_title("Geostationary Electron Flux (>2 MeV) & Geomagnetic Storm Index")
    ax1.legend(loc='upper right')
    
    ax2.plot(subset_df.index, subset_df['dst_proxy'], color='crimson', label='Dst Proxy (nT)', linewidth=1.5)
    ax2.axhline(-50, color='orange', linestyle=':', label='Moderate Storm')
    ax2.axhline(-100, color='darkred', linestyle=':', label='Intense Storm')
    ax2.set_ylabel("Dst Proxy (nT)")
    ax2.set_xlabel("UTC Time")
    ax2.legend(loc='lower left')
    
    # Shade storm periods
    ax2.fill_between(subset_df.index, subset_df['dst_proxy'], -50, where=(subset_df['dst_proxy'] < -50),
                     color='orange', alpha=0.3, label='Storm Activity')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "goes_flux_and_storm_time.png"), dpi=150)
    plt.close()
    
    # -------------------------------------------------------------
    # Plot 2: Solar Wind Parameter Panels Aligned with Flux
    # -------------------------------------------------------------
    print("Generating Plot 2: Solar Wind Parameters Panel...")
    fig, axes = plt.subplots(5, 1, figsize=(15, 15), sharex=True)
    
    axes[0].plot(subset_df.index, subset_df['goes_flux'], color='purple')
    axes[0].set_yscale('log')
    axes[0].set_ylabel("Flux (e-/cm²/s/sr)")
    axes[0].set_title("Aligned Space Weather Observation Panel (Last 30 Days)")
    axes[0].axhline(YELLOW_LIMIT, color='gold', linestyle='--')
    
    axes[1].plot(subset_df.index, subset_df['Vsw'], color='teal')
    axes[1].set_ylabel("Vsw (km/s)")
    
    axes[2].plot(subset_df.index, subset_df['Bz'], color='blue')
    axes[2].axhline(0, color='black', linestyle=':', alpha=0.5)
    axes[2].set_ylabel("Bz (nT)")
    
    axes[3].plot(subset_df.index, subset_df['N'], color='green')
    axes[3].set_ylabel("N (cm⁻³)")
    
    axes[4].plot(subset_df.index, subset_df['Pdyn'], color='orange')
    axes[4].set_ylabel("Pdyn (nPa)")
    axes[4].set_xlabel("UTC Time")
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "solar_wind_panels.png"), dpi=150)
    plt.close()
    
    # -------------------------------------------------------------
    # Plot 3: Superposed Epoch Analysis (SEA) around Storm Onsets
    # -------------------------------------------------------------
    print("Generating Plot 3: Superposed Epoch Analysis...")
    # Identify storm onsets using Dst proxy local minima
    # Define storm onset as Dst dropping below -50 nT, separated by at least 7 days
    storm_threshold = -50
    onsets = []
    min_dist = int(7 * 24 * 12) # 7 days in steps
    
    idx = 0
    while idx < len(df) - min_dist:
        if df['dst_proxy'].iloc[idx] < storm_threshold:
            # Find local minimum in next 24h
            local_range = df['dst_proxy'].iloc[idx : idx + int(24 * 12)]
            min_idx = local_range.idxmin()
            onsets.append(min_idx)
            idx += min_dist # skip to avoid duplicates
        else:
            idx += 1
            
    print(f"Identified {len(onsets)} geomagnetic storm events for SEA.")
    
    if onsets:
        # Window: -24 hours to +72 hours (at 5-min resolution)
        pre_hours = 24
        post_hours = 72
        pre_steps = int(pre_hours * 12)
        post_steps = int(post_hours * 12)
        window_size = pre_steps + post_steps + 1
        
        flux_matrix = []
        vsw_matrix = []
        
        for onset_dt in onsets:
            onset_idx = df.index.get_loc(onset_dt)
            if onset_idx < pre_steps or onset_idx > len(df) - post_steps - 1:
                continue
            
            flux_slice = df['goes_flux'].iloc[onset_idx - pre_steps : onset_idx + post_steps + 1].values
            vsw_slice = df['Vsw'].iloc[onset_idx - pre_steps : onset_idx + post_steps + 1].values
            
            if len(flux_slice) == window_size:
                flux_matrix.append(flux_slice)
                vsw_matrix.append(vsw_slice)
                
        flux_matrix = np.array(flux_matrix)
        vsw_matrix = np.array(vsw_matrix)
        
        time_axis = np.linspace(-pre_hours, post_hours, window_size)
        
        fig, (ax_f, ax_v) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot Flux SEA
        med_f = np.nanmedian(flux_matrix, axis=0)
        p10_f = np.nanpercentile(flux_matrix, 10, axis=0)
        p90_f = np.nanpercentile(flux_matrix, 90, axis=0)
        
        ax_f.plot(time_axis, med_f, color='purple', linewidth=2.5, label='Median')
        ax_f.fill_between(time_axis, p10_f, p90_f, color='purple', alpha=0.2, label='10th - 90th Percentile')
        ax_f.set_yscale('log')
        ax_f.axvline(0, color='red', linestyle='--', linewidth=2, label='Storm Onset (t=0)')
        ax_f.set_ylabel("GOES Electron Flux (e-/cm²/s/sr)")
        ax_f.set_title("Superposed Epoch Analysis (SEA) of Storm Events")
        ax_f.legend(loc='upper right')
        
        # Plot Vsw SEA
        med_v = np.nanmedian(vsw_matrix, axis=0)
        p10_v = np.nanpercentile(vsw_matrix, 10, axis=0)
        p90_v = np.nanpercentile(vsw_matrix, 90, axis=0)
        
        ax_v.plot(time_axis, med_v, color='teal', linewidth=2.5, label='Median')
        ax_v.fill_between(time_axis, p10_v, p90_v, color='teal', alpha=0.2, label='10th - 90th Percentile')
        ax_v.axvline(0, color='red', linestyle='--', linewidth=2)
        ax_v.set_ylabel("Solar Wind Speed Vsw (km/s)")
        ax_v.set_xlabel("Hours Relative to Onset")
        ax_v.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "superposed_epoch_analysis.png"), dpi=150)
        plt.close()
        
    # -------------------------------------------------------------
    # Plot 4: Lagged Correlation Heatmap
    # -------------------------------------------------------------
    print("Generating Plot 4: Lagged Correlation Heatmap...")
    # Calculate lags of 0, 1, 3, 6, 12, 24 hours (0, 12, 36, 72, 144, 288 steps)
    lags = [0, 1, 3, 6, 12, 24]
    lag_steps = [int(lh * 12) for lh in lags]
    
    corr_data = {}
    variables = ['Vsw', 'B', 'Bz', 'N', 'Pdyn', 'dst_proxy']
    
    for var in variables:
        for lag, step in zip(lags, lag_steps):
            corr_data[f"{var}_lag_{lag}h"] = df[var].shift(step)
            
    corr_df = pd.DataFrame(corr_data)
    corr_df['log_goes_flux'] = df['log_goes_flux']
    
    # Calculate correlation matrix
    corr_matrix = corr_df.corr()
    
    # Extract only correlations with target log_goes_flux
    target_corr = pd.DataFrame(index=variables, columns=[f"{l}h" for l in lags])
    for var in variables:
        for lag in lags:
            target_corr.loc[var, f"{lag}h"] = corr_matrix.loc[f"{var}_lag_{lag}h", "log_goes_flux"]
            
    target_corr = target_corr.astype(float)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(target_corr, annot=True, cmap='coolwarm', vmin=-1.0, vmax=1.0, fmt=".3f", linewidths=.5)
    plt.title("Correlation with log10(GOES Flux) at Various Lags")
    plt.ylabel("Solar Wind Variables")
    plt.xlabel("Lag Time (hours)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "lagged_correlation_heatmap.png"), dpi=150)
    plt.close()
    
    print(f"All EDA plots saved successfully to folder '{out_dir}'!")

if __name__ == "__main__":
    generate_eda_plots()
