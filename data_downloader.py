import os
import urllib.request
import re
from datetime import datetime

def download_file(url, dest_path):
    """Downloads a file with a progress indicator."""
    print(f"Downloading {url} -> {dest_path}")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print("Success.")
        return True
    except Exception as e:
        print(f"Failed to download: {e}")
        return False

def download_cdaweb_data(spacecraft, year, month, dest_dir):
    """
    Downloads CDF data from NASA's CDAWeb directory for a given year and month.
    This demonstrates downloading real Wind SWE/MFI and GOES EPEAD data.
    """
    os.makedirs(dest_dir, exist_ok=True)
    
    # Example URL prefixes
    if spacecraft == "wind_mfi":
        # Wind MFI 1-minute IMF data
        url_base = f"https://cdaweb.gsfc.nasa.gov/pub/data/wind/mfi/mfi_h2/{year}/"
        pattern = rf"wi_h2_mfi_{year}{month:02d}\d{{2}}_v\d{{2}}\.cdf"
    elif spacecraft == "wind_swe":
        # Wind SWE 1-minute solar wind plasma data
        url_base = f"https://cdaweb.gsfc.nasa.gov/pub/data/wind/swe/swe_h1/{year}/"
        pattern = rf"wi_h1_swe_{year}{month:02d}\d{{2}}_v\d{{2}}\.cdf"
    elif spacecraft == "goes15":
        # GOES-15 5-minute electron flux
        url_base = f"https://cdaweb.gsfc.nasa.gov/pub/data/goes/goes15/g15_cpead_5m/{year}/"
        pattern = rf"g15_cpead_5m_{year}{month:02d}\d{{2}}_v\d{{2}}\.cdf"
    else:
        print(f"Unknown spacecraft: {spacecraft}")
        return
    
    print(f"Scanning CDAWeb directory: {url_base} ...")
    try:
        with urllib.request.urlopen(url_base) as response:
            html = response.read().decode('utf-8')
            
        files = re.findall(rf'href="({pattern})"', html)
        # Deduplicate files (take the latest version if duplicates exist)
        files = sorted(list(set(files)))
        
        if not files:
            print("No matching files found.")
            return
        
        print(f"Found {len(files)} files to download.")
        # Download first 2 files as a quick demonstration
        for filename in files[:2]:
            file_url = url_base + filename
            dest_file = os.path.join(dest_dir, filename)
            download_file(file_url, dest_file)
            
    except Exception as e:
        print(f"Error accessing CDAWeb index: {e}")

def main():
    print("=================================================================")
    print("Space Weather Data Downloader (NASA CDAWeb Client)")
    print("=================================================================")
    print("This script is configured to pull raw CDF files from NASA SPDF.")
    print("Since downloading 11 years of high-cadence CDF files requires")
    print("several gigabytes, please run 'generate_mock_data.py' for quick")
    print("local testing and dashboard demonstration.")
    print("=================================================================\n")
    
    choice = input("Do you want to run a quick test download of 2 Wind MFI files? (y/n): ").strip().lower()
    if choice == 'y':
        download_cdaweb_data("wind_mfi", 2020, 1, "data/raw/wind_test")
    else:
        print("Download bypassed. Run 'python generate_mock_data.py' to populate simulated files.")

if __name__ == "__main__":
    main()
