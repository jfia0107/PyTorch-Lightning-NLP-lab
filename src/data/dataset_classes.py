"""

Dataset classes for PyTorch dataloaders.
Encoder and CDL lanes same as "prepare_data.py"
"""

import torch
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
from typing import Dict, Any


class CDLDataset(Dataset):
    # Vocabulary based CDL lane

    def __init__(self, hf_dataset: HFDataset) -> None:
        self.dataset = hf_dataset
        self.labels = hf_dataset["label"]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.dataset[idx]
        return {
            "input_ids": example["input_ids"].detach().clone().long(),
            "label": example["label"].detach().clone().float(),
        }


class EncoderDataset(Dataset):
    # Encoder lane, added attention mask

    def __init__(self, hf_dataset: HFDataset) -> None:
        self.dataset = hf_dataset
        self.labels = hf_dataset["label"]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.dataset[idx]
        return {
            "input_ids": example["input_ids"].detach().clone().long(),
            "label": example["label"].detach().clone().float(),
            "attention_mask": example["attention_mask"].detach().clone().to(torch.long),
        }
