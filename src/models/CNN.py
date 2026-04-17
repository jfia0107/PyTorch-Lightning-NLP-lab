"""
Classic PyTorch 1D CNN with multi-scale filters and global max pooling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class CNN(nn.Module):

    def __init__(
        self,
        input_size: int,
        num_filters: int,
        filter_sizes: List[int],
        dropout: float,
        output_dim: int,
    ):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=input_size, out_channels=num_filters, kernel_size=k)
            for k in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(len(filter_sizes) * num_filters, output_dim)

    def forward(self, sequence_features: torch.Tensor, **kwargs) -> torch.Tensor:
        x = sequence_features.permute(0, 2, 1)
        pooled = [
            F.max_pool1d(F.relu(conv(x)), kernel_size=x.shape[2] - conv.kernel_size[0] + 1).squeeze(2)
            for conv in self.convs
        ]
        out = torch.cat(pooled, dim=1)
        return self.head(self.dropout(out))
