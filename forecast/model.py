"""Traffic inflow forecasting models: LSTM and Temporal Transformer.

Input : the last FORECAST_HISTORY bins (30s each) for one intersection.
        Per bin: vehicles that entered each phase's approach lanes
        (MAX_PHASES values), the share of time upstream signals were
        green toward each phase (MAX_PHASES values), and normalized simulation time.
Output: vehicles expected to arrive on each phase's approach in each of
        the next FORECAST_HORIZON bins.

Both models support multi-intersection parameter sharing.
"""
import math
import numpy as np
import torch
import torch.nn as nn

from config import (FORECAST_D_MODEL, FORECAST_HISTORY, FORECAST_HORIZON,
                    FORECAST_NHEAD, FORECAST_NUM_LAYERS, MAX_PHASES)

IN_DIM = 2 * MAX_PHASES + 1
COUNT_SCALE = 10.0


def normalise(x):
    x = np.array(x, dtype=np.float32, copy=True)
    x[..., :MAX_PHASES] /= COUNT_SCALE
    return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequence modeling."""

    def __init__(self, d_model, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x shape: (B, T, D)
        return x + self.pe[:, :x.size(1)]


class LSTMForecaster(nn.Module):
    """Recurrent LSTM forecaster."""

    def __init__(self, hidden=64, layers=1):
        super().__init__()
        self.lstm = nn.LSTM(IN_DIM, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, FORECAST_HORIZON * MAX_PHASES)
        )

    def forward(self, x):  # x: (B, H, IN_DIM), normalised
        out, _ = self.lstm(x)
        y = self.head(out[:, -1])
        return y.view(-1, FORECAST_HORIZON, MAX_PHASES)


class TransformerForecaster(nn.Module):
    """Temporal Transformer with Multi-Head Self-Attention for inflow forecasting."""

    def __init__(self, d_model=FORECAST_D_MODEL, nhead=FORECAST_NHEAD,
                 num_layers=FORECAST_NUM_LAYERS, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(IN_DIM, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=FORECAST_HISTORY + 5)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu"
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, FORECAST_HORIZON * MAX_PHASES)
        )

    def forward(self, x):  # x: (B, H, IN_DIM), normalised
        h = self.input_proj(x)
        h = self.pos_encoder(h)
        encoded = self.transformer_encoder(h)
        # Pool temporal representation (mean pooling across history window)
        pooled = encoded.mean(dim=1)
        y = self.head(pooled)
        return y.view(-1, FORECAST_HORIZON, MAX_PHASES)


class Forecaster:
    """Inference wrapper used inside TrafficEnv, supporting both architectures."""

    def __init__(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        arch = ckpt.get("arch", "lstm")
        kwargs = ckpt.get("kwargs", {})
        if arch == "transformer":
            self.model = TransformerForecaster(**kwargs)
        else:
            self.model = LSTMForecaster(**kwargs)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.arch = arch

    @torch.no_grad()
    def predict(self, history):
        """history: (N, FORECAST_HISTORY, IN_DIM) raw -> (N, horizon, P) vehicles."""
        x = torch.from_numpy(normalise(history))
        return np.clip(self.model(x).numpy() * COUNT_SCALE, 0, None)


def make_windows(series):
    """Turn one (T, IN_DIM) series into (X, Y) training windows."""
    series = np.asarray(series, dtype=np.float32)
    pad = np.zeros((FORECAST_HISTORY, IN_DIM), dtype=np.float32)
    s = np.concatenate([pad, series])
    X, Y = [], []
    for end in range(FORECAST_HISTORY, len(s) - FORECAST_HORIZON + 1):
        X.append(s[end - FORECAST_HISTORY:end])
        Y.append(s[end:end + FORECAST_HORIZON, :MAX_PHASES])
    return np.array(X), np.array(Y)
