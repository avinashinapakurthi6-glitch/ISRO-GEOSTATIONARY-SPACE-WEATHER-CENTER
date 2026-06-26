import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import mlflow
import sys

from data_utils import prepare_data, FEATURES
from models import TemporalFusionTransformer, QuantileLoss

def train_tft_model(data_path="data/processed/features_data.parquet", models_dir="models"):
    """
    Trains the custom Temporal Fusion Transformer (TFT) model using Quantile Loss.
    """
    os.makedirs(models_dir, exist_ok=True)
    
    is_verify = "--verify" in sys.argv
    
    # 1. Prepare Datasets
    print("Preparing sequence datasets for TFT...")
    train_dataset, val_dataset, _, _ = prepare_data(data_path, seq_len=288, upsample_factor=1 if is_verify else 4)
    
    # Dataloaders
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Initialize Model & Training Configurations
    num_features = len(FEATURES)
    seq_len = 288
    
    model = TemporalFusionTransformer(
        num_features=num_features,
        seq_len=seq_len,
        d_model=64,
        n_horizons=3,
        n_quantiles=3,
        num_heads=4,
        dropout=0.1
    )
    model = model.to(device)
    
    criterion = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4) # TFT can use slightly lower lr
    
    # Cosine Annealing learning rate schedule
    epochs = 1 if is_verify else 50
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Early stopping settings
    patience = 10
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    # MLflow tracking
    try:
        mlflow.set_experiment("ISRO_Space_Weather_TFT")
        mlflow.start_run(run_name="tft_primary_training")
        mlflow.log_params({
            "batch_size": batch_size,
            "d_model": 64,
            "num_heads": 4,
            "lr": 5e-4,
            "weight_decay": 1e-4,
            "optimizer": "AdamW",
            "epochs": epochs
        })
    except Exception as e:
        print(f"MLflow tracking bypassed: {e}")
        
    print("Starting TFT training loop...")
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_count = 0
        for idx, (batch_x, batch_y) in enumerate(train_loader):
            if is_verify and idx >= 5:
                break
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            preds, _, _ = model(batch_x) # returns preds, selection_weights, attn_weights
            loss = criterion(preds, batch_y)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            train_count += batch_x.size(0)
            
        train_loss /= train_count
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for idx, (batch_x, batch_y) in enumerate(val_loader):
                if is_verify and idx >= 5:
                    break
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                preds, _, _ = model(batch_x)
                loss = criterion(preds, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                val_count += batch_x.size(0)
                
        val_loss /= val_count
        
        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Log to MLflow
        try:
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
        except:
            pass
            
        # Early stopping & model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save best checkpoint
            torch.save(model.state_dict(), os.path.join(models_dir, "tft_model.pt"))
            print("  --> Best model saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
                
    try:
        mlflow.end_run()
    except:
        pass
        
    print("TFT training completed!")

if __name__ == "__main__":
    train_tft_model()
