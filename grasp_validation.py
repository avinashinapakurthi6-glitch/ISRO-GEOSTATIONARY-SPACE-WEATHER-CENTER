import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress, pearsonr
import cdflib

from data_reader import read_cdf_variables

def run_grasp_validation(raw_dir="data/raw", processed_dir="data/processed", plots_dir="plots"):
    """
    Reads GRASP CDF files, aligns with GOES observations, computes longitudinal correction factors,
    and visualizes the cross-calibration results.
    """
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Read GRASP data
    grasp_pattern = os.path.join(raw_dir, "grasp", "*.cdf")
    grasp_files = sorted(glob.glob(grasp_pattern))
    
    if not grasp_files:
        print("No GRASP CDF files found. Run generate_mock_data.py first to create them.")
        return
        
    print(f"Reading GRASP CDF files from {raw_dir}/grasp...")
    grasp_dfs = []
    for filepath in grasp_files:
        df = read_cdf_variables(filepath, ["flux"])
        if not df.empty:
            grasp_dfs.append(df)
            
    if not grasp_dfs:
        print("Failed to read any GRASP data.")
        return
        
    grasp_all = pd.concat(grasp_dfs).sort_index()
    grasp_all = grasp_all[~grasp_all.index.duplicated(keep='first')]
    grasp_all.rename(columns={"flux": "grasp_flux"}, inplace=True)
    grasp_all['log_grasp_flux'] = np.log10(np.clip(grasp_all['grasp_flux'], 1.0, 1e7))
    
    # 2. Read aligned GOES data
    aligned_path = os.path.join(processed_dir, "aligned_data.parquet")
    if not os.path.exists(aligned_path):
        print("Aligned parquet data not found. Run data_reader.py first.")
        return
    
    goes_df = pd.read_parquet(aligned_path)[['goes_flux']]
    goes_df['log_goes_flux'] = np.log10(np.clip(goes_df['goes_flux'], 1.0, 1e7))
    
    # 3. Align both datasets in UTC
    print("Aligning GOES and GRASP datasets...")
    combined = pd.concat([goes_df, grasp_all], axis=1).dropna()
    
    if combined.empty:
        print("No overlapping periods found between GOES and GRASP datasets.")
        return
        
    print(f"Overlapping data size: {combined.shape[0]} points.")
    
    # 4. Cross-calibration: Linear Regression in log10 space
    # log10(GRASP) = slope * log10(GOES) + intercept
    slope, intercept, r_value, p_value, std_err = linregress(
        combined['log_goes_flux'], combined['log_grasp_flux']
    )
    
    print("\n=============================================================")
    print("ISRO GRASP VS NOAA GOES CROSS-CALIBRATION RESULTS")
    print("=============================================================")
    print(f"Linear Fit Equation: log10(GRASP) = {slope:.4f} * log10(GOES) + {intercept:.4f}")
    print(f"Pearson Correlation (R): {r_value:.4f}")
    print(f"R-squared: {r_value**2:.4f}")
    print("=============================================================\n")
    
    # Apply calibration correction to GOES
    combined['log_goes_flux_corrected'] = slope * combined['log_goes_flux'] + intercept
    combined['goes_flux_corrected'] = 10 ** combined['log_goes_flux_corrected']
    
    # Save correction factors
    calibration_factors = {'slope': slope, 'intercept': intercept, 'r_value': r_value}
    os.makedirs("models", exist_ok=True)
    np.save("models/grasp_calibration.npy", calibration_factors)
    
    # 5. Visualization: Scatter plots and Time-series overlays
    # Plot A: Scatter Plot (Before & After Calibration)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), sharex=True, sharey=True)
    
    # Raw Scatter
    ax1.scatter(combined['log_goes_flux'], combined['log_grasp_flux'], color='royalblue', alpha=0.1, s=5)
    # Plot 1:1 line
    min_val = min(combined['log_goes_flux'].min(), combined['log_grasp_flux'].min())
    max_val = max(combined['log_goes_flux'].max(), combined['log_grasp_flux'].max())
    ax1.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='1:1 Line')
    ax1.set_xlabel("log10(GOES Flux) [American Longitude]")
    ax1.set_ylabel("log10(GRASP Flux) [Indian Longitude]")
    ax1.set_title("Before Calibration (Raw Comparison)")
    ax1.legend()
    
    # Corrected Scatter
    ax2.scatter(combined['log_goes_flux_corrected'], combined['log_grasp_flux'], color='teal', alpha=0.1, s=5)
    ax2.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='1:1 Line')
    ax2.set_xlabel("Corrected log10(GOES Flux)")
    ax2.set_title("After Cross-Calibration Correction")
    ax2.legend()
    
    plt.suptitle("NOAA GOES vs ISRO GRASP Flux Cross-Calibration", fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "grasp_goes_scatter_calibration.png"), dpi=150)
    plt.close()
    
    # Plot B: Time Series Overlay (Diurnal Shift & Calibration)
    # Take a 3-day window to clearly show the diurnal variations (shifted local noon)
    # GOES local noon is at UTC 17:00; GRASP local noon is at UTC 07:00 (10 hour shift)
    plt.figure(figsize=(15, 7))
    subset = combined.iloc[-int(3 * 24 * 12):] # Last 3 days
    
    plt.plot(subset.index, subset['log_grasp_flux'], color='black', label='ISRO GRASP Observations', linewidth=2.5)
    plt.plot(subset.index, subset['log_goes_flux'], color='royalblue', linestyle=':', label='NOES GOES (Raw)', linewidth=1.5)
    plt.plot(subset.index, subset['log_goes_flux_corrected'], color='crimson', label='GOES (Corrected & Aligned)', linewidth=2.0)
    
    plt.title("GOES vs GRASP Time Series Overlay (Showing ~10h Longitudinal Phase Shift)")
    plt.ylabel("log10(Flux)")
    plt.xlabel("UTC Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "grasp_goes_timeseries_overlay.png"), dpi=150)
    plt.close()
    
    print(f"GRASP validation completed. Plots saved to '{plots_dir}/'. Calibration saved to 'models/'.")

if __name__ == "__main__":
    run_grasp_validation()
