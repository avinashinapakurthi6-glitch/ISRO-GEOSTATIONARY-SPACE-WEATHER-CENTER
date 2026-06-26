import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from scipy.stats import pearsonr
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

from data_utils import prepare_data, FEATURES, HORIZON_STEPS, TARGET
from models import TemporalFusionTransformer, Seq2SeqLSTM

def compute_classification_metrics(y_true, y_pred, threshold=4.0):
    """
    Computes POD, FAR, and HSS for binary storm classification.
    """
    obs_storm = y_true > threshold
    pred_storm = y_pred > threshold
    
    tp = np.sum(obs_storm & pred_storm)
    fp = np.sum((~obs_storm) & pred_storm)
    fn = np.sum(obs_storm & (~pred_storm))
    tn = np.sum((~obs_storm) & (~obs_storm)) # standard count
    
    # Correct formula for total elements
    n = len(y_true)
    tn = n - (tp + fp + fn)
    
    pod = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    far = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Heidke Skill Score (HSS)
    expected_correct = ((tp + fn) * (tp + fp) + (tn + fn) * (tn + fp)) / n
    hss = (tp + tn - expected_correct) / (n - expected_correct) if (n - expected_correct) > 0 else 0.0
    
    return {
        'POD': pod,
        'FAR': far,
        'HSS': hss,
        'TP': int(tp),
        'FP': int(fp),
        'FN': int(fn),
        'TN': int(tn)
    }

def run_evaluation(data_path="data/processed/features_data.parquet", models_dir="models", out_dir="plots"):
    """
    Evaluates the Persistence, XGBoost, LSTM, and TFT models on the test set.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Load data
    print("Loading test data...")
    _, _, test_dataset, scaler = prepare_data(data_path, seq_len=288)
    
    import sys
    is_verify = "--verify" in sys.argv
    if is_verify:
        print("Running in verification mode. Subsetting test dataset to first 2000 samples.")
        test_dataset.indices = test_dataset.indices[:2000]
        test_dataset.targets = test_dataset.targets[:2000]
        
    # Load raw feature data to extract alignment and persistence
    df = pd.read_parquet(data_path)
    test_df = df[df.index >= '2020-01-01'].dropna()
    
    # For seq models, we have PyTorch Datasets
    # Let's extract all y_true and inputs from test_dataset
    print("Extracting test targets...")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False)
    
    y_test_true = test_dataset.targets # (N, n_horizons)
    n_samples = len(y_test_true)
    print(f"Number of test samples: {n_samples}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # -------------------------------------------------------------
    # 2. Load Models and Run Inference
    # -------------------------------------------------------------
    preds_dict = {}
    
    # A. Persistence model: target(t + h) = target(t)
    # We find the current flux (which is log_goes_flux at final step of input history, index seq_len-1)
    df_raw = pd.read_parquet(data_path)
    test_df_raw = df_raw[df_raw.index >= '2020-01-01']
    y_test_raw = test_df_raw[TARGET].values
    
    idx_te = test_dataset.indices
    seq_len = 288
    
    # y(t) is at idx_te + seq_len - 1
    y_t_current = y_test_raw[idx_te + seq_len - 1]
    
    # Persistence predicts y(t) for all horizons
    preds_dict['Persistence'] = np.stack([y_t_current, y_t_current, y_t_current], axis=1)
    
    # B. XGBoost Model
    # Since XGBoost was trained on tabular features, we evaluate it on tabular features.
    # The tabular features for each sample are the values at time t (which is at idx_te + seq_len - 1).
    print("Running XGBoost Inference...")
    xgb_preds = np.zeros((n_samples, 3, 3)) # (samples, horizons, quantiles)
    horizons = ["30m", "6h", "12h"]
    quantiles = [0.1, 0.5, 0.9]
    
    x_test_raw = test_df_raw[FEATURES].fillna(0).values
    X_test_tabular = pd.DataFrame(x_test_raw[idx_te + seq_len - 1], columns=FEATURES)
    
    xgb_loaded = True
    for h_idx, h_name in enumerate(horizons):
        for q_idx, q in enumerate(quantiles):
            model_name = f"xgb_h{h_name}_q{int(q*100)}"
            model_path = os.path.join(models_dir, f"{model_name}.json")
            if os.path.exists(model_path):
                model = xgb.XGBRegressor()
                model.load_model(model_path)
                xgb_preds[:, h_idx, q_idx] = model.predict(X_test_tabular)
            else:
                xgb_loaded = False
                
    if xgb_loaded:
        preds_dict['XGBoost'] = xgb_preds
    else:
        print("XGBoost models not found. Skipping.")
        
    # C. LSTM Model
    print("Running LSTM Inference...")
    lstm_path = os.path.join(models_dir, "lstm_model.pt")
    if os.path.exists(lstm_path):
        lstm_model = Seq2SeqLSTM(num_features=len(FEATURES), d_model=64, n_horizons=3, n_quantiles=3)
        lstm_model.load_state_dict(torch.load(lstm_path, map_location=device))
        lstm_model.to(device)
        lstm_model.eval()
        
        lstm_preds_list = []
        with torch.no_grad():
            for bx, _ in test_loader:
                bx = bx.to(device)
                pred_batch = lstm_model(bx).cpu().numpy()
                lstm_preds_list.append(pred_batch)
        preds_dict['LSTM'] = np.concatenate(lstm_preds_list, axis=0)
    else:
        print("LSTM model not found. Skipping.")
        
    # D. TFT Model
    print("Running TFT Inference...")
    tft_path = os.path.join(models_dir, "tft_model.pt")
    if os.path.exists(tft_path):
        tft_model = TemporalFusionTransformer(num_features=len(FEATURES), seq_len=seq_len, d_model=64, n_horizons=3, n_quantiles=3)
        tft_model.load_state_dict(torch.load(tft_path, map_location=device))
        tft_model.to(device)
        tft_model.eval()
        
        tft_preds_list = []
        with torch.no_grad():
            for bx, _ in test_loader:
                bx = bx.to(device)
                pred_batch, _, _ = tft_model(bx)
                tft_preds_list.append(pred_batch.cpu().numpy())
        preds_dict['TFT'] = np.concatenate(tft_preds_list, axis=0)
    else:
        print("TFT model not found. Skipping.")
        
    # -------------------------------------------------------------
    # 3. Compute Metrics
    # -------------------------------------------------------------
    metrics_summary = []
    
    for model_name, preds in preds_dict.items():
        # preds can have shape (N, 3) (Persistence) or (N, 3, 3) (quantiles)
        for h_idx, (h_name, h_step) in enumerate(zip(horizons, HORIZON_STEPS)):
            y_true = y_test_true[:, h_idx]
            
            # Predict value (use 50th percentile/median if multi-quantile, else raw)
            if preds.ndim == 3:
                y_pred = preds[:, h_idx, 1] # 50th percentile (index 1)
            else:
                y_pred = preds[:, h_idx]
                
            # Regression metrics
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r_val, _ = pearsonr(y_true, y_pred)
            
            # Skill Score vs Persistence
            # SS = 1 - (MSE_model / MSE_persistence)
            y_pers = preds_dict['Persistence'][:, h_idx]
            mse_pers = mean_squared_error(y_true, y_pers)
            mse_model = mean_squared_error(y_true, y_pred)
            ss = 1.0 - (mse_model / mse_pers) if mse_pers > 0 else 0.0
            
            # Classification/Storm metrics (flux > 10^4 in log10 is > 4.0)
            class_metrics = compute_classification_metrics(y_true, y_pred, threshold=4.0)
            
            metrics_summary.append({
                'Model': model_name,
                'Horizon': h_name,
                'RMSE': rmse,
                'MAE': mae,
                'Pearson_R': r_val,
                'Skill_Score_vs_Pers': ss,
                'POD': class_metrics['POD'],
                'FAR': class_metrics['FAR'],
                'HSS': class_metrics['HSS']
            })
            
    metrics_df = pd.DataFrame(metrics_summary)
    print("\n=============================================================")
    print("EVALUATION METRICS SUMMARY ON TEST SET (2020-2021)")
    print("=============================================================")
    print(metrics_df.to_string(index=False))
    
    # Save CSV
    metrics_df.to_csv("models/evaluation_metrics.csv", index=False)
    
    # -------------------------------------------------------------
    # 4. Generate Reliability Diagrams
    # -------------------------------------------------------------
    print("\nGenerating Reliability Diagrams...")
    # For models with quantiles, we check what fraction of y_true is below the predicted quantile
    quantile_models = [m for m in ['XGBoost', 'LSTM', 'TFT'] if m in preds_dict]
    
    if quantile_models:
        plt.figure(figsize=(10, 8))
        target_quantiles = [0.1, 0.5, 0.9]
        
        for model_name in quantile_models:
            preds = preds_dict[model_name] # (N, 3, 3)
            # Evaluate across all horizons combined for robustness
            actual_fractions = []
            for q_idx, q in enumerate(target_quantiles):
                count_below = 0
                total_count = 0
                for h_idx in range(3):
                    y_true = y_test_true[:, h_idx]
                    y_pred_q = preds[:, h_idx, q_idx]
                    count_below += np.sum(y_true <= y_pred_q)
                    total_count += len(y_true)
                actual_fractions.append(count_below / total_count)
                
            plt.plot(target_quantiles, actual_fractions, marker='o', label=model_name, linewidth=2)
            
        plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Perfect Calibration')
        plt.title("Reliability Diagram (Quantile Calibration)")
        plt.xlabel("Target Quantile")
        plt.ylabel("Observed Fraction Below Quantile")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "quantile_reliability_diagram.png"), dpi=150)
        plt.close()
        
    # -------------------------------------------------------------
    # 5. Generate Time-Series Comparison Plot
    # -------------------------------------------------------------
    print("Generating Forecast Comparison Plot...")
    # Plot a 5-day storm window in the test set to compare predictions
    plt.figure(figsize=(15, 8))
    
    # Find a storm period in the test set (where true flux > 4.0)
    storm_indices = np.where(y_test_true[:, 1] > 4.0)[0]
    if len(storm_indices) > 0:
        start_idx = max(0, storm_indices[0] - 288) # 24h before
        end_idx = min(n_samples, storm_indices[0] + 576) # 48h after
    else:
        start_idx = 0
        end_idx = min(n_samples, 288 * 5)
        
    subset_slice = slice(start_idx, end_idx)
    time_index = test_df.index[idx_te[subset_slice] + seq_len - 1]
    
    plt.plot(time_index, y_test_true[subset_slice, 1], color='black', label='Observed Flux', linewidth=2)
    
    colors = {'Persistence': 'gray', 'XGBoost': 'orange', 'LSTM': 'green', 'TFT': 'blue'}
    for model_name, preds in preds_dict.items():
        if model_name == 'Persistence': continue
        
        # 6h horizon (index 1)
        if preds.ndim == 3:
            y_pred = preds[subset_slice, 1, 1] # Median
            # Shade 10th-90th percentile for TFT or LSTM
            if model_name in ['TFT', 'LSTM']:
                y_pred_10 = preds[subset_slice, 1, 0]
                y_pred_90 = preds[subset_slice, 1, 2]
                plt.fill_between(time_index, y_pred_10, y_pred_90, color=colors[model_name], alpha=0.15)
        else:
            y_pred = preds[subset_slice, 1]
            
        plt.plot(time_index, y_pred, color=colors[model_name], label=f"{model_name} (6h Forecast)", linewidth=1.5)
        
    plt.title("6-Hour Forecast Comparison during Storm Window")
    plt.ylabel("log10(GOES >2 MeV Flux)")
    plt.xlabel("UTC Time")
    plt.axhline(4.0, color='red', linestyle='--', alpha=0.7, label='Hazard Threshold (10^4)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "forecast_comparison_storm_window.png"), dpi=150)
    plt.close()
    
    print("Evaluation completed. Plots saved to 'plots/' directory.")

if __name__ == "__main__":
    run_evaluation()
