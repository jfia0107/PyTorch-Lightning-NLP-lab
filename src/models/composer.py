"""
Universal backbone + head container.

"""

import torch
import torch.nn as nn


class ComposedNLPModel(nn.Module):

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        name: str = "",
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.name = name

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:

        if hasattr(self.backbone, "config"):
            # Transformer backbone
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            sequence_features = outputs.last_hidden_state
        else:
            # Embedding backbone (vocab-based)
            sequence_features = self.backbone(input_ids)

        return self.head(sequence_features, attention_mask=attention_mask, **kwargs)
