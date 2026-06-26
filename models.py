import torch
import torch.nn as nn
import numpy as np
import xgboost as xgb

# =====================================================================
# 1. Persistence Baseline Model
# =====================================================================
class PersistenceModel:
    """
    Persistence baseline: flux(t + dt) = flux(t)
    """
    def __init__(self):
        pass
        
    def predict(self, current_flux, horizon_steps):
        # Simply returns the current flux repeated for the forecast horizon
        # shape: (n_samples,) -> returns (n_samples,)
        return current_flux

# =====================================================================
# 2. PyTorch Helper Modules for LSTM and TFT
# =====================================================================
class GLU(nn.Module):
    """Gated Linear Unit"""
    def __init__(self, d_in, d_out):
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out)
        self.linear2 = nn.Linear(d_in, d_out)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        return self.sigmoid(self.linear1(x)) * self.linear2(x)

class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN) as defined in the TFT paper"""
    def __init__(self, d_in, d_hidden, d_out, dropout=0.1, context_dim=None):
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_hidden)
        if context_dim is not None:
            self.context_linear = nn.Linear(context_dim, d_hidden, bias=False)
        else:
            self.context_linear = None
        self.linear2 = nn.Linear(d_hidden, d_hidden)
        self.glu = GLU(d_hidden, d_out)
        self.layernorm = nn.LayerNorm(d_out)
        self.dropout = nn.Dropout(dropout)
        
        # Skip connection projection if dimensions don't match
        if d_in != d_out:
            self.skip_proj = nn.Linear(d_in, d_out)
        else:
            self.skip_proj = nn.Identity()
            
    def forward(self, x, c=None):
        h = self.linear1(x)
        if c is not None and self.context_linear is not None:
            h = h + self.context_linear(c)
        h = torch.relu(h)
        h = self.linear2(h)
        h = self.dropout(h)
        h = self.glu(h)
        return self.layernorm(self.skip_proj(x) + h)

class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network (VSN) for feature selection"""
    def __init__(self, num_features, d_model, dropout=0.1):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        
        # Processors for individual variables
        self.single_grns = nn.ModuleList([
            GatedResidualNetwork(d_model, d_model, d_model, dropout=dropout)
            for _ in range(num_features)
        ])
        
        # Flattened feature representation to weights
        self.flatten_grn = GatedResidualNetwork(num_features * d_model, d_model, num_features, dropout=dropout)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x_list):
        # x_list is a list of tensors of shape (batch, seq, d_model), length = num_features
        # Or shape (batch, d_model) if sequence dimension is 1
        
        # Process each variable separately
        processed = [grn(x) for grn, x in zip(self.single_grns, x_list)]
        
        # Stack variables: (batch, seq, num_features, d_model)
        stacked = torch.stack(processed, dim=-2)
        
        # Concatenate for selection weights: (batch, seq, num_features * d_model)
        concat_features = torch.cat(processed, dim=-1)
        
        # Get selection weights: (batch, seq, num_features)
        weights = self.flatten_grn(concat_features)
        weights = self.softmax(weights).unsqueeze(-1) # (batch, seq, num_features, 1)
        
        # Weighted sum: (batch, seq, d_model)
        out = torch.sum(weights * stacked, dim=-2)
        return out, weights.squeeze(-1)

# Quantile Loss function (10th, 50th, 90th percentiles)
class QuantileLoss(nn.Module):
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles
        
    def forward(self, preds, target):
        # preds: (batch, n_horizons, n_quantiles)
        # target: (batch, n_horizons)
        losses = []
        for i, q in enumerate(self.quantiles):
            pred_q = preds[..., i] # (batch, n_horizons)
            error = target - pred_q
            loss = torch.max((q - 1) * error, q * error)
            losses.append(loss.mean())
        return torch.stack(losses).sum()

# =====================================================================
# 3. Seq2Seq LSTM Model
# =====================================================================
class Seq2SeqLSTM(nn.Module):
    """
    Sequence-to-Sequence Encoder-Decoder LSTM model for multi-horizon forecast.
    """
    def __init__(self, num_features, d_model=64, n_horizons=3, n_quantiles=3):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        self.n_horizons = n_horizons
        self.n_quantiles = n_quantiles
        
        # Input projection
        self.input_proj = nn.Linear(num_features, d_model)
        
        # Encoder LSTM
        self.encoder = nn.LSTM(d_model, d_model, batch_first=True, num_layers=2)
        
        # Decoder LSTM
        self.decoder = nn.LSTM(d_model, d_model, batch_first=True, num_layers=2)
        
        # Prediction heads
        # Output: (batch, n_horizons, n_quantiles)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 32),
                nn.ReLU(),
                nn.Linear(32, n_quantiles)
            ) for _ in range(n_horizons)
        ])
        
    def forward(self, x):
        # x: (batch, seq_len, num_features)
        x_proj = self.input_proj(x)
        
        # Encode
        _, (h, c) = self.encoder(x_proj)
        
        # Decode step-by-step for horizons
        # Decoder input starts as zero vector
        batch_size = x.size(0)
        dec_input = torch.zeros(batch_size, 1, self.d_model, device=x.device)
        
        dec_states = (h, c)
        dec_outputs = []
        
        for _ in range(self.n_horizons):
            dec_out, dec_states = self.decoder(dec_input, dec_states)
            dec_outputs.append(dec_out.squeeze(1)) # (batch, d_model)
            dec_input = dec_out # Teacher forcing or feedback
            
        # Predict quantiles for each horizon
        preds = []
        for i in range(self.n_horizons):
            pred_h = self.heads[i](dec_outputs[i]) # (batch, n_quantiles)
            preds.append(pred_h)
            
        return torch.stack(preds, dim=1) # (batch, n_horizons, n_quantiles)

# =====================================================================
# 4. Custom Temporal Fusion Transformer (TFT) Model
# =====================================================================
class TemporalFusionTransformer(nn.Module):
    """
    Custom lightweight Temporal Fusion Transformer (TFT) in PyTorch.
    Optimized for multi-horizon forecasting, feature selection, and quantile loss.
    """
    def __init__(self, num_features, seq_len, d_model=64, n_horizons=3, n_quantiles=3, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_features = num_features
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_horizons = n_horizons
        self.n_quantiles = n_quantiles
        
        # Feature embeddings (linear projection for numerical variables)
        self.feature_projections = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(num_features)
        ])
        
        # Variable Selection Network for historical time-series
        self.vsn = VariableSelectionNetwork(num_features, d_model, dropout=dropout)
        
        # Locality enhancement (LSTM layer)
        self.lstm = nn.LSTM(d_model, d_model, batch_first=True, num_layers=1)
        self.lstm_gate = GLU(d_model, d_model)
        self.lstm_layernorm = nn.LayerNorm(d_model)
        
        # Temporal Self-Attention
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.attn_gate = GLU(d_model, d_model)
        self.attn_layernorm = nn.LayerNorm(d_model)
        
        # Position-wise Feed-Forward Network
        self.grn_ff = GatedResidualNetwork(d_model, d_model, d_model, dropout=dropout)
        
        # Decoder/Prediction component
        # We project the temporal representations from the last time step to each horizon
        self.horizon_projs = nn.ModuleList([
            nn.Sequential(
                GatedResidualNetwork(d_model, d_model, d_model, dropout=dropout),
                nn.Linear(d_model, n_quantiles)
            ) for _ in range(n_horizons)
        ])
        
    def forward(self, x):
        # x: (batch, seq_len, num_features)
        batch_size, seq_len, num_features = x.size()
        
        # 1. Input embedding projection
        projected = []
        for i in range(num_features):
            feat_single = x[:, :, i].unsqueeze(-1) # (batch, seq_len, 1)
            proj_feat = self.feature_projections[i](feat_single) # (batch, seq_len, d_model)
            projected.append(proj_feat)
            
        # 2. Variable Selection Network
        vsn_out, selection_weights = self.vsn(projected) # vsn_out: (batch, seq_len, d_model)
        
        # 3. Locality enhancement via LSTM
        lstm_out, _ = self.lstm(vsn_out) # (batch, seq_len, d_model)
        # Gated residual connection
        lstm_out = self.lstm_layernorm(vsn_out + self.lstm_gate(lstm_out))
        
        # 4. Temporal Self-Attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out) # (batch, seq_len, d_model)
        attn_out = self.attn_layernorm(lstm_out + self.attn_gate(attn_out))
        
        # 5. Position-wise Feed Forward (GRN)
        ff_out = self.grn_ff(attn_out) # (batch, seq_len, d_model)
        
        # 6. Predict multi-horizon percentiles from the final step's hidden state
        final_state = ff_out[:, -1, :] # (batch, d_model)
        
        preds = []
        for i in range(self.n_horizons):
            pred_h = self.horizon_projs[i](final_state) # (batch, n_quantiles)
            preds.append(pred_h)
            
        # Stack output: (batch, n_horizons, n_quantiles)
        preds_stacked = torch.stack(preds, dim=1)
        
        return preds_stacked, selection_weights, attn_weights
