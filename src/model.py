"""
model.py — Manipal-Climate-RNN
═══════════════════════════════════════════════════════════════════════════════
Stacked Bidirectional LSTM for multivariate climate forecasting.

Architecture (exact spec)
  Input       : (batch, 30, 12)           — 30-day window, 12 features
  BiLSTM ×2   : hidden_dim=64, dropout=0.2, bidirectional=True
  Concat      : h_n[-2] ‖ h_n[-1]        — forward + backward last hidden
                → (batch, 128)
  Dropout(0.2)
  FC          : Linear(128, 2)            — [temp_°C, precip_mm]

Hidden-state extraction
  h_n has shape (num_layers*2, batch, 64).
  For 2 stacked BiLSTM layers:
    h_n[0] = layer-0 forward,  h_n[1] = layer-0 backward
    h_n[2] = layer-1 forward,  h_n[3] = layer-1 backward
  We take h_n[-2] (last-layer fwd) and h_n[-1] (last-layer bwd).
  Concatenation gives 64+64 = 128 dims.
"""

import torch
import torch.nn as nn


class StackedBiLSTM(nn.Module):
    """
    Stacked Bidirectional LSTM forecaster.

    Parameters
    ----------
    input_dim  : int    number of input features        (12 for this dataset)
    hidden_dim : int    LSTM hidden size per direction   (64, spec)
    num_layers : int    stacked LSTM depth               (2, spec)
    dropout    : float  inter-layer + FC dropout rate   (0.2, spec)
    output_dim : int    forecast targets                 (2: temp + precip)
    """

    def __init__(self,
                 input_dim:  int   = 12,
                 hidden_dim: int   = 64,
                 num_layers: int   = 2,
                 dropout:    float = 0.2,
                 output_dim: int   = 2):
        super().__init__()
        self.hidden_dim     = hidden_dim
        self.num_layers     = num_layers
        self.num_directions = 2   # bidirectional

        # ── BiLSTM ────────────────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )

        # ── Head ──────────────────────────────────────────────────────────────
        self.dropout = nn.Dropout(p=dropout)
        # 64 (fwd) + 64 (bwd) = 128 → 2
        self.fc = nn.Linear(hidden_dim * self.num_directions, output_dim)

        self._init_weights()

    # ── forward ───────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x   : (batch, seq_len=30, input_dim=12)
        out : (batch, 2)
        """
        # h_n : (num_layers*2, batch, hidden_dim)  =  (4, B, 64)
        _, (h_n, _) = self.lstm(x)

        # Last-layer: forward = h_n[-2], backward = h_n[-1]
        h_fwd = h_n[-2]                       # (B, 64)
        h_bwd = h_n[-1]                       # (B, 64)
        h_cat = torch.cat([h_fwd, h_bwd], 1)  # (B, 128)
        h_cat = self.dropout(h_cat)
        return self.fc(h_cat)                 # (B, 2)

    # ── weight init ───────────────────────────────────────────────────────────
    def _init_weights(self):
        for name, p in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p.data)
            elif "bias" in name:
                p.data.fill_(0)
                n = p.size(0)
                p.data[n // 4 : n // 2].fill_(1.0)   # forget-gate bias = 1
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        return (f"StackedBiLSTM(input={self.lstm.input_size}, "
                f"hidden={self.hidden_dim}×2={self.hidden_dim*2}, "
                f"layers={self.num_layers}, "
                f"fc_in=128, fc_out=2, "
                f"params={self.count_parameters():,})")


# ── quick sanity ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    m = StackedBiLSTM(input_dim=12)
    print(m)
    x = torch.randn(8, 30, 12)
    y = m(x)
    print(f"Input {x.shape} → Output {y.shape}")
    assert y.shape == (8, 2)
    print("Shape assertion passed.")
