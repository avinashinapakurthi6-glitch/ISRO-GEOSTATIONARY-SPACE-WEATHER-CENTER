import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow
import sys

# Define features and horizons
from data_utils import FEATURES, TARGET, HORIZON_STEPS

def train_xgboost(data_path="data/processed/features_data.parquet", models_dir="models"):
    """
    Trains XGBoost regressors for each forecast horizon (30m, 6h, 12h)
    and each target quantile (10th, 50th, 90th percentiles).
    """
    os.makedirs(models_dir, exist_ok=True)
    
    print("Loading feature data...")
    df = pd.read_parquet(data_path)
    
    is_verify = "--verify" in sys.argv
    if is_verify:
        print("Running in verification mode. Training dataset subset to 2018.")
        train_df = df[(df.index >= '2018-01-01') & (df.index < '2019-01-01')].dropna()
    else:
        train_df = df[df.index < '2019-01-01'].dropna()
        
    val_df = df[(df.index >= '2019-01-01') & (df.index < '2020-01-01')].dropna()
    
    X_train = train_df[FEATURES]
    X_val = val_df[FEATURES]
    
    # Initialize MLflow
    try:
        mlflow.set_experiment("ISRO_Space_Weather_XGBoost")
    except Exception as e:
        print(f"MLflow experiment setup bypassed: {e}")
        
    horizons = ["30m", "6h", "12h"]
    quantiles = [0.1, 0.5, 0.9]
    
    # Store trained models
    trained_models = {}
    
    for h_idx, (h_name, h_step) in enumerate(zip(horizons, HORIZON_STEPS)):
        # Targets are shifted backwards to align features (t) with targets (t + h_step)
        y_train = train_df[TARGET].shift(-h_step).dropna()
        y_val = val_df[TARGET].shift(-h_step).dropna()
        
        # Align features and targets after shift dropouts
        common_train_idx = X_train.index.intersection(y_train.index)
        common_val_idx = X_val.index.intersection(y_val.index)
        
        X_tr, y_tr = X_train.loc[common_train_idx], y_train.loc[common_train_idx]
        X_va, y_va = X_val.loc[common_val_idx], y_val.loc[common_val_idx]
        
        # Implement upsampling of storm periods (y > 4.0) in training set
        is_storm = y_tr > 4.0
        X_tr_storm = X_tr[is_storm]
        y_tr_storm = y_tr[is_storm]
        
        if len(y_tr_storm) > 0:
            # Upsample storm periods 4x
            X_tr = pd.concat([X_tr] + [X_tr_storm] * 3)
            y_tr = pd.concat([y_tr] + [y_tr_storm] * 3)
            
            # Shuffle
            shuffled_indices = np.random.permutation(len(y_tr))
            X_tr = X_tr.iloc[shuffled_indices]
            y_tr = y_tr.iloc[shuffled_indices]
            
        print(f"\nTraining models for Horizon: {h_name} (+{h_step} steps)...")
        print(f"Training shape: {X_tr.shape}, Validation shape: {X_va.shape}")
        
        for q in quantiles:
            run_name = f"xgb_h{h_name}_q{int(q*100)}"
            print(f"Fitting model: {run_name} (quantile {q})...")
            
            # Use quantile loss objective if available in xgboost
            # objective = reg:quantileerror, quantile_alpha = q
            params = {
                'objective': 'reg:quantileerror',
                'quantile_alpha': q,
                'max_depth': 2 if is_verify else 6,
                'learning_rate': 0.1 if is_verify else 0.05,
                'n_estimators': 5 if is_verify else 150,
                'random_state': 42,
                'n_jobs': -1
            }
            
            # Start MLflow run
            try:
                run = mlflow.start_run(run_name=run_name)
                mlflow.log_params(params)
                mlflow.log_param("horizon", h_name)
                mlflow.log_param("quantile", q)
            except Exception as e:
                run = None
                
            model = xgb.XGBRegressor(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                verbose=False
            )
            
            # Evaluate on validation set
            preds = model.predict(X_va)
            mse = mean_squared_error(y_va, preds)
            mae = mean_absolute_error(y_va, preds)
            
            # Log metrics
            print(f"Validation MAE: {mae:.4f}, MSE: {mse:.4f}")
            try:
                if run:
                    mlflow.log_metric("val_mae", mae)
                    mlflow.log_metric("val_mse", mse)
                    mlflow.end_run()
            except:
                pass
                
            # Save model
            model_path = os.path.join(models_dir, f"{run_name}.json")
            model.save_model(model_path)
            
    print("\nAll XGBoost models trained and saved successfully!")

if __name__ == "__main__":
    train_xgboost()
