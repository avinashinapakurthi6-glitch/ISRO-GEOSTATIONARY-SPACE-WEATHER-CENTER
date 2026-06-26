import os
import joblib
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# Define features and horizons
FEATURES = [
    'Vsw', 'B', 'Bz', 'N', 'Pdyn', 'bz_south', 'dst_proxy',
    'bz_lag_1h', 'bz_lag_3h', 'vsw_lag_6h', 'vsw_lag_12h',
    'vsw_roll_mean_3h', 'vsw_roll_std_3h', 'vsw_roll_mean_6h', 'vsw_roll_std_6h',
    'bz_roll_mean_3h', 'bz_roll_std_3h', 'bz_roll_mean_6h', 'bz_roll_std_6h',
    'sin_time_of_day', 'cos_time_of_day', 'sin_day_of_year', 'cos_day_of_year',
    'log_goes_flux_lag_30m', 'log_goes_flux_lag_1h', 'log_goes_flux_lag_3h'
]

TARGET = 'log_goes_flux'

HORIZON_STEPS = [6, 72, 144] # 30 min, 6 hours, 12 hours (at 5-min cadence)

class SpaceWeatherDataset(Dataset):
    """
    Custom PyTorch Dataset for on-the-fly sequence generation.
    """
    def __init__(self, features, targets, indices, seq_len=288):
        self.features = features
        self.targets = targets
        self.indices = indices
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        end_idx = start_idx + self.seq_len
        
        x = self.features[start_idx:end_idx]
        # Target is at t + horizon_steps
        y = self.targets[idx]
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def prepare_data(data_path="data/processed/features_data.parquet", seq_len=288, upsample_factor=4, save_scaler_dir="data/processed"):
    """
    Loads parquet data, splits into train/val/test, scales inputs,
    builds sequence indices, and performs storm-time upsampling.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Feature data parquet not found at {data_path}. Run feature_engineer.py first.")
        
    df = pd.read_parquet(data_path)
    
    # Define splits
    train_df = df[df.index < '2019-01-01']
    val_df = df[(df.index >= '2019-01-01') & (df.index < '2020-01-01')]
    test_df = df[df.index >= '2020-01-01']
    
    print(f"Data split: Train={train_df.shape[0]}, Val={val_df.shape[0]}, Test={test_df.shape[0]}")
    
    # Scaler
    scaler = StandardScaler()
    
    # Clean features of NaN values for fitting
    train_features_clean = train_df[FEATURES].dropna()
    scaler.fit(train_features_clean)
    
    # Save scaler
    os.makedirs(save_scaler_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(save_scaler_dir, "scaler.joblib"))
    print("Scaler saved.")
    
    # Scale each dataset
    def scale_df(df_in):
        # We fill NaNs in features with 0 (since they are scaled, 0 is mean) or we forward fill.
        # Note: We already did forward fill in feature_engineer.py, but for remaining NaNs
        # (e.g. at the very start of the dataset due to lags/rolling windows), we fill with mean.
        x_raw = df_in[FEATURES].copy()
        x_scaled = scaler.transform(x_raw.fillna(train_features_clean.mean()))
        
        # Keep target log_goes_flux raw
        y_raw = df_in[TARGET].values
        
        return x_scaled, y_raw
        
    x_train, y_train = scale_df(train_df)
    x_val, y_val = scale_df(val_df)
    x_test, y_test = scale_df(test_df)
    
    # Function to get valid indices
    def get_sequence_indices(x, y, is_train=False):
        # Max steps we need to look ahead
        max_horizon = max(HORIZON_STEPS)
        n_records = len(x)
        limit = n_records - seq_len - max_horizon
        
        if limit <= 0:
            return x, np.empty((0, len(HORIZON_STEPS))), np.empty(0, dtype=int)
            
        # Precompute rows with NaNs in features
        has_nan_row = np.isnan(x).any(axis=1)
        cumsum = np.cumsum(np.insert(has_nan_row.astype(int), 0, 0))
        valid_features = (cumsum[seq_len : limit + seq_len] - cumsum[0 : limit]) == 0
        
        # Precompute rows with NaNs in targets
        is_nan_y = np.isnan(y)
        valid_targets = np.ones(limit, dtype=bool)
        for h in HORIZON_STEPS:
            valid_targets &= ~is_nan_y[seq_len + h - 1 : limit + seq_len + h - 1]
            
        indices = np.where(valid_features & valid_targets)[0]
        
        if len(indices) == 0:
            target_vals = np.empty((0, len(HORIZON_STEPS)))
        else:
            target_vals = np.stack([y[indices + seq_len + h - 1] for h in HORIZON_STEPS], axis=1)
        
        # Storm-time upsampling (flux > 10^4 in log10 space is > 4.0)
        if is_train and upsample_factor > 1 and len(indices) > 0:
            # Check if any of the target horizons exceed 4.0
            is_storm = (target_vals > 4.0).any(axis=1)
            storm_indices = indices[is_storm]
            storm_targets = target_vals[is_storm]
            
            if len(storm_indices) > 0:
                print(f"Original train sequences: {len(indices)}, storm sequences: {len(storm_indices)}")
                # Duplicate storm sequences
                upsampled_indices = []
                upsampled_targets = []
                for _ in range(upsample_factor - 1):
                    upsampled_indices.append(storm_indices)
                    upsampled_targets.append(storm_targets)
                    
                indices = np.concatenate([indices] + upsampled_indices)
                target_vals = np.concatenate([target_vals] + upsampled_targets)
                
                # Shuffle the upsampled indices
                shuffle_idx = np.random.permutation(len(indices))
                indices = indices[shuffle_idx]
                target_vals = target_vals[shuffle_idx]
                print(f"Upsampled train sequences: {len(indices)}")
                
        return x, target_vals, indices
        
    # Build sequences
    x_tr, y_tr, idx_tr = get_sequence_indices(x_train, y_train, is_train=True)
    x_va, y_va, idx_va = get_sequence_indices(x_val, y_val, is_train=False)
    x_te, y_te, idx_te = get_sequence_indices(x_test, y_test, is_train=False)
    
    # Create datasets
    train_dataset = SpaceWeatherDataset(x_tr, y_tr, idx_tr, seq_len)
    val_dataset = SpaceWeatherDataset(x_va, y_va, idx_va, seq_len)
    test_dataset = SpaceWeatherDataset(x_te, y_te, idx_te, seq_len)
    
    return train_dataset, val_dataset, test_dataset, scaler

if __name__ == "__main__":
    prepare_data()
