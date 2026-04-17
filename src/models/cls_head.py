"""
Takes CLS token to create a linear projection. For pure BERT-type models as they need a classification head for a specific task.
"""

import torch
import torch.nn as nn


class CLSHead(nn.Module):

    def __init__(self, input_size: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # linear out:
        self.head = nn.Linear(input_size, output_dim)

    def forward(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        if features.ndim == 3:
            features = features[:, 0, :]
        return self.head(self.dropout(features))
