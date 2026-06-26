import os
import glob
import pandas as pd
import numpy as np
import cdflib

def read_cdf_variables(filepath, var_list):
    """
    Reads specific variables and the Epoch from a CDF file.
    Returns a pandas DataFrame.
    """
    print(f"Reading CDF: {filepath}")
    try:
        cdf = cdflib.CDF(filepath)
    except Exception as e:
        print(f"Error opening {filepath}: {e}")
        return pd.DataFrame()
        
    try:
        epoch = cdf.varget("Epoch")
        # Convert epochs to datetime objects
        datetimes = cdflib.cdfepoch.to_datetime(epoch)
        
        # Build dictionary
        data = {'datetime': datetimes}
        for var in var_list:
            try:
                data[var] = cdf.varget(var)
            except Exception as ev:
                print(f"Variable {var} not found in {filepath}: {ev}")
                data[var] = np.nan
        
        df = pd.DataFrame(data)
        df.set_index('datetime', inplace=True)
        return df
    finally:
        try:
            cdf.close()
        except:
            pass

def process_and_align_data(raw_dir="data/raw", processed_dir="data/processed", cadence_mins=5):
    """
    Reads all yearly GOES and Wind CDF files, aligns them to a common UTC axis,
    and caches the result as a Parquet file.
    """
    os.makedirs(processed_dir, exist_ok=True)
    
    # 1. Read GOES electron flux files
    goes_pattern = os.path.join(raw_dir, "goes", "*.cdf")
    goes_files = sorted(glob.glob(goes_pattern))
    
    if not goes_files:
        print(f"No GOES CDF files found in {raw_dir}/goes. Please run generate_mock_data.py first.")
        return None
        
    goes_dfs = []
    for filepath in goes_files:
        df = read_cdf_variables(filepath, ["flux"])
        if not df.empty:
            goes_dfs.append(df)
            
    if not goes_dfs:
        print("Failed to read any GOES data.")
        return None
    goes_all = pd.concat(goes_dfs).sort_index()
    # Remove duplicates
    goes_all = goes_all[~goes_all.index.duplicated(keep='first')]
    
    # Rename flux to goes_flux
    goes_all.rename(columns={"flux": "goes_flux"}, inplace=True)
    
    # 2. Read Wind plasma and IMF files
    wind_pattern = os.path.join(raw_dir, "wind", "*.cdf")
    wind_files = sorted(glob.glob(wind_pattern))
    
    if not wind_files:
        print(f"No Wind CDF files found in {raw_dir}/wind. Please run generate_mock_data.py first.")
        return None
        
    wind_dfs = []
    wind_vars = ["Vsw", "B", "Bz", "N", "Pdyn"]
    for filepath in wind_files:
        df = read_cdf_variables(filepath, wind_vars)
        if not df.empty:
            wind_dfs.append(df)
            
    if not wind_dfs:
        print("Failed to read any Wind data.")
        return None
    wind_all = pd.concat(wind_dfs).sort_index()
    # Remove duplicates
    wind_all = wind_all[~wind_all.index.duplicated(keep='first')]
    
    # 3. Define the common aligned UTC time axis
    start_time = max(goes_all.index.min(), wind_all.index.min())
    end_time = min(goes_all.index.max(), wind_all.index.max())
    
    print(f"Aligning datasets from {start_time} to {end_time} ...")
    
    freq_str = f"{cadence_mins}min"
    common_index = pd.date_range(start=start_time, end=end_time, freq=freq_str)
    
    # 4. Reindex and interpolate GOES
    goes_aligned = goes_all.reindex(common_index)
    goes_aligned = goes_aligned.interpolate(method='linear', limit=int(120 / cadence_mins)) # max 2h gap interpolation
    
    # 5. Reindex and interpolate Wind
    wind_aligned = wind_all.reindex(common_index)
    wind_aligned = wind_aligned.interpolate(method='linear', limit=int(120 / cadence_mins)) # max 2h gap interpolation
    
    # 6. Combine aligned datasets
    combined = pd.concat([goes_aligned, wind_aligned], axis=1)
    combined.index.name = "datetime"
    
    # Save to Parquet
    out_path = os.path.join(processed_dir, "aligned_data.parquet")
    combined.to_parquet(out_path)
    print(f"Successfully aligned and saved to Parquet: {out_path}")
    print(f"Dataset shape: {combined.shape}")
    return combined

if __name__ == "__main__":
    process_and_align_data()
