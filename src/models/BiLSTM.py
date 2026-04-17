"""
Classic PyTorch Bidirectional LSTM head.
"""

import torch
import torch.nn as nn


class BiLSTM(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        output_dim: int,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(2 * hidden_size, output_dim)

    def forward(self, sequence_features: torch.Tensor, **kwargs) -> torch.Tensor:
        _, (hidden, _) = self.lstm(sequence_features)
        last = hidden[-2:, :, :]
        pooled = torch.cat((last[0], last[1]), dim=1)
        pooled = self.dropout(pooled)
        return self.head(pooled)
