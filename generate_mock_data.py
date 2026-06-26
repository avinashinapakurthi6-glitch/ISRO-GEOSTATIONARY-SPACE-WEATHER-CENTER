import os
import datetime
import numpy as np
import cdflib

def generate_year_data(year, cadence_mins=5):
    """
    Generates realistic synthetic space weather data for a single year.
    Cadence: 5 minutes (default) or 1 minute.
    """
    start_time = datetime.datetime(year, 1, 1, 0, 0, 0)
    end_time = datetime.datetime(year, 12, 31, 23, 59, 59)
    delta = datetime.timedelta(minutes=cadence_mins)
    
    times = []
    curr = start_time
    while curr <= end_time:
        times.append(curr)
        curr += delta
    
    n_steps = len(times)
    
    # Time arrays for simulation
    t_days = np.array([(t - start_time).total_seconds() / 86400.0 for t in times])
    
    # 1. Simulate Solar Wind Speed Vsw (km/s)
    # Background Vsw with recurrent high speed streams (every 27 days due to solar rotation)
    # plus random transient storm events (CMEs)
    vsw_base = 350.0 + 50.0 * np.sin(2 * np.pi * t_days / 27.0)
    vsw_noise = np.random.normal(0, 15.0, n_steps)
    vsw = vsw_base + vsw_noise
    
    # Add CME storm arrivals (random spikes followed by exponential decay)
    n_storms = np.random.randint(15, 25) # 15 to 25 storms per year
    storm_days = np.random.uniform(0, 365, n_storms)
    for sd in storm_days:
        idx_start = int((sd / 365.0) * n_steps)
        if idx_start >= n_steps: continue
        # Vsw jumps up by 200-450 km/s
        jump = np.random.uniform(200.0, 450.0)
        # Decay constant (e-folding time of ~2.5 days)
        decay_steps = int((2.5 * 24 * 60) / cadence_mins)
        decay = jump * np.exp(-np.arange(n_steps - idx_start) / decay_steps)
        vsw[idx_start:] += decay

    vsw = np.clip(vsw, 250.0, 950.0)
    
    # 2. Simulate IMF Bz (nT)
    # Fluctuate around 0, turns strongly southward (negative) during storms (triggered by CME arrival)
    bz = np.random.normal(0.0, 3.0, n_steps)
    for sd in storm_days:
        idx_start = int((sd / 365.0) * n_steps)
        if idx_start >= n_steps: continue
        # Southward turn duration (6 to 24 hours)
        dur_steps = int(np.random.uniform(6.0, 24.0) * 60.0 / cadence_mins)
        idx_end = min(idx_start + dur_steps, n_steps)
        bz[idx_start:idx_end] -= np.random.uniform(10.0, 25.0)
        # Add some recovery
        recovery_steps = int((1.0 * 24 * 60) / cadence_mins)
        rec_end = min(idx_end + recovery_steps, n_steps)
        bz[idx_end:rec_end] += np.random.uniform(0.0, 5.0)
        
    bz = np.clip(bz, -30.0, 15.0)

    # 3. Simulate IMF B magnitude (nT)
    # B increases when Bz is negative or Vsw is high
    b = np.abs(bz) + 3.0 + np.random.exponential(2.0, n_steps)
    b = np.clip(b, 2.0, 40.0)
    
    # 4. Simulate Density N (cm^-3)
    # Density spikes at storm onset, then drops below baseline (sheath compression followed by cavity)
    n = 5.0 + 2.0 * np.sin(2 * np.pi * t_days / 27.0) + np.random.normal(0, 1.0, n_steps)
    for sd in storm_days:
        idx_start = int((sd / 365.0) * n_steps)
        if idx_start >= n_steps: continue
        # Spike duration (3 to 8 hours)
        dur_steps = int(np.random.uniform(3.0, 8.0) * 60.0 / cadence_mins)
        idx_end = min(idx_start + dur_steps, n_steps)
        n[idx_start:idx_end] += np.random.uniform(15.0, 45.0)
        # Depletion cavity (1 to 2 days)
        dep_steps = int((1.5 * 24 * 60) / cadence_mins)
        dep_end = min(idx_end + dep_steps, n_steps)
        n[idx_end:dep_end] = np.maximum(0.5, n[idx_end:dep_end] - np.random.uniform(2.0, 4.0))

    n = np.clip(n, 0.1, 80.0)
    
    # 5. Simulate Dynamic Pressure Pdyn (nPa)
    # Pdyn = 2e-6 * N * Vsw^2
    pdyn = 2e-6 * n * (vsw ** 2)
    
    # 6. Simulate GOES >2 MeV Electron Flux (electrons/cm^2/s/sr)
    # Physics model: 
    # - Delayed response to Vsw (1-2 day lag)
    # - Prompt dropout due to magnetopause shadowing when Pdyn is extremely high
    # - Diurnal variations based on satellite local time (GOES is at ~75W longitude, so local time = UTC - 5 hours)
    # - Flux in log10 space ranges from 1 to 5.5
    
    # Vsw smooth driver (exponential moving average with 1.5 day decay)
    alpha = 1.0 - np.exp(-cadence_mins / (1.5 * 24 * 60))
    vsw_smooth = np.zeros(n_steps)
    curr_v = 400.0
    for i in range(n_steps):
        curr_v = (1 - alpha) * curr_v + alpha * vsw[i]
        vsw_smooth[i] = curr_v
        
    log_flux_base = 1.0 + 3.5 * (vsw_smooth - 300.0) / 500.0
    log_flux_base = np.clip(log_flux_base, 1.0, 5.0)
    
    # Diurnal variations: GOES is at ~75W longitude.
    # Local noon is at UTC 17:00, local midnight at UTC 05:00.
    utc_hours = np.array([t.hour + t.minute / 60.0 for t in times])
    diurnal = 0.4 * np.cos(2 * np.pi * (utc_hours - 17.0) / 24.0)
    
    log_flux = log_flux_base + diurnal + np.random.normal(0, 0.15, n_steps)
    
    # Shadowing dropouts: when Pdyn > 5 nPa, electron flux drops rapidly
    for i in range(n_steps):
        if pdyn[i] > 6.0:
            log_flux[i] -= np.minimum(2.0, (pdyn[i] - 6.0) * 0.3)
            
    # Southward Bz (geomagnetic activity) drives flux enhancement after a lag of ~12 hours
    # Smooth negative Bz with ~12h delay
    bz_south = np.maximum(0.0, -bz)
    alpha_bz = 1.0 - np.exp(-cadence_mins / (12.0 * 60.0))
    bz_smooth = np.zeros(n_steps)
    curr_bz = 0.0
    for i in range(n_steps):
        curr_bz = (1 - alpha_bz) * curr_bz + alpha_bz * bz_south[i]
        bz_smooth[i] = curr_bz
        
    log_flux += 0.8 * (bz_smooth / 15.0)
    log_flux = np.clip(log_flux, 0.5, 6.0)
    flux = 10 ** log_flux
    
    # 7. Simulate GRASP Electron Flux (Indian longitude ~74E, local time = UTC + 5 hours)
    # Local noon is at UTC 07:00. Diurnal shift of +10 hours relative to GOES.
    # GRASP flux also has cross-calibration difference (e.g. scale factor of 0.85 + small offset)
    grasp_diurnal = 0.4 * np.cos(2 * np.pi * (utc_hours - 7.0) / 24.0)
    grasp_log_flux = 0.95 * log_flux_base + grasp_diurnal + np.random.normal(0, 0.15, n_steps) - 0.1
    # Shadowing dropouts aligned
    for i in range(n_steps):
        if pdyn[i] > 6.0:
            grasp_log_flux[i] -= np.minimum(2.0, (pdyn[i] - 6.0) * 0.3)
    grasp_log_flux = np.clip(grasp_log_flux, 0.5, 6.0)
    grasp_flux = 10 ** grasp_log_flux
    
    # Convert datetimes to CDF Epochs
    # compute_epoch expects a list of [year, month, day, hour, minute, second, millisecond]
    time_components = [[t.year, t.month, t.day, t.hour, t.minute, t.second, int(t.microsecond/1000)] for t in times]
    epoch_vals = cdflib.cdfepoch.compute_epoch(time_components)
    
    return epoch_vals, flux, vsw, b, bz, n, pdyn, grasp_flux

def write_cdf_file(filepath, epoch, variables):
    """
    Writes variables to a CDF file.
    """
    if os.path.exists(filepath):
        os.remove(filepath)
        
    writer = cdflib.cdfwrite.CDF(filepath)
    
    # Write Epoch
    epoch_spec = {
        'Variable': 'Epoch',
        'Data_Type': 31,
        'Num_Elements': 1,
        'Rec_Vary': True,
        'Dim_Sizes': []
    }
    writer.write_var(epoch_spec, var_data=epoch)
    
    # Write other variables
    for var_name, var_data in variables.items():
        var_spec = {
            'Variable': var_name,
            'Data_Type': 33,
            'Num_Elements': 1,
            'Rec_Vary': True,
            'Dim_Sizes': []
        }
        writer.write_var(var_spec, var_data=var_data)
        
    writer.close()
    print(f"Written: {filepath}")

def main():
    np.random.seed(42)
    base_dir = "data/raw"
    os.makedirs(os.path.join(base_dir, "goes"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "wind"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "grasp"), exist_ok=True)
    
    print("Generating simulated space weather CDF data (2010 - 2021)...")
    # For GOES and Wind, generate yearly files from 2010 to 2021
    # For GRASP, generate yearly files for 2020 and 2021 (the validation period)
    for year in range(2010, 2022):
        print(f"Generating data for year {year}...")
        epoch, flux, vsw, b, bz, n, pdyn, grasp_flux = generate_year_data(year)
        
        # Write GOES CDF
        goes_file = os.path.join(base_dir, "goes", f"goes_flux_{year}.cdf")
        write_cdf_file(goes_file, epoch, {'flux': flux})
        
        # Write Wind CDF
        wind_file = os.path.join(base_dir, "wind", f"wind_plasma_imf_{year}.cdf")
        write_cdf_file(wind_file, epoch, {
            'Vsw': vsw,
            'B': b,
            'Bz': bz,
            'N': n,
            'Pdyn': pdyn
        })
        
        # Write GRASP CDF for 2020-2021
        if year >= 2020:
            grasp_file = os.path.join(base_dir, "grasp", f"grasp_flux_{year}.cdf")
            write_cdf_file(grasp_file, epoch, {'flux': grasp_flux})
            
    print("All mock CDF files generated successfully!")

if __name__ == "__main__":
    main()
