"""
Unified Lightning DataModule for both lanes (encoder, CDL).

"""

import os
import logging
from typing import Optional

import numpy as np
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from datasets import load_from_disk
from omegaconf import DictConfig

from src.data.dataset_classes import CDLDataset, EncoderDataset

log = logging.getLogger(__name__)


class UnifiedDataModule(pl.LightningDataModule):

    def __init__(self, cfg: DictConfig,  **kwargs):
        super().__init__()
        self.cfg = cfg
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self._sampler = None

    def prepare_data(self):
        #Assert preprocessed data exists. Always run prepare_data.py first
        processed_path = self._processed_path()
        if not os.path.exists(processed_path):
            raise FileNotFoundError(
                f"Processed data not found at {processed_path}. "
                f"Run prepare_data.py."
            )
        log.info(f"Processed data found at {processed_path}")

    def setup(self, stage: Optional[str] = None):
        dataset = load_from_disk(self._processed_path())
        dataset.set_format(type="torch")

        DatasetClass = EncoderDataset if self._tokenizer_name() else CDLDataset

        self.train_ds = DatasetClass(dataset["train"])
        self.val_ds = DatasetClass(dataset["validation"])
        self.test_ds = DatasetClass(dataset["test"])

    def train_dataloader(self) -> DataLoader:
        dc = self.cfg.data
        sampler_cfg = self.cfg.get("sampler", {})
        sampler_name = sampler_cfg.get("name", "random")

        # Base kwargs that apply to ALL dataloaders
        dl_kwargs = {
            "dataset": self.train_ds,
            "num_workers": dc.num_workers,
            "pin_memory": True,
            "persistent_workers": dc.num_workers > 0,
        }

        sampler = None

        if sampler_name == "stratified":
            from src.utils.samplers import StratifiedOversampleSampler
            labels = np.array(self.train_ds.labels, dtype=np.float32)
            self._sampler = StratifiedOversampleSampler(
                labels=labels,
                batch_size=dc.batch_size,
                minority_ratio=sampler_cfg.get("minority_ratio", 0.5),
            )
            sampler = self._sampler

        dl_kwargs["batch_size"] = dc.batch_size
        dl_kwargs["shuffle"] = sampler is None
        dl_kwargs["sampler"] = sampler
        dl_kwargs["drop_last"] = True

        return DataLoader(**dl_kwargs)

    def val_dataloader(self) -> DataLoader:
        dc = self.cfg.data

        dl_kwargs = {
            "dataset": self.val_ds,
            "batch_size": dc.batch_size,
            "shuffle": False,
            "num_workers": dc.num_workers,
            "pin_memory": True,
            "persistent_workers": dc.num_workers > 0,
        }

        return DataLoader(**dl_kwargs)

    def test_dataloader(self) -> DataLoader:
        dc = self.cfg.data

        dl_kwargs = {
            "dataset": self.test_ds,
            "batch_size": dc.batch_size,
            "shuffle": False,
            "num_workers": dc.num_workers,
            "pin_memory": True,
            "persistent_workers": dc.num_workers > 0,
        }

        return DataLoader(**dl_kwargs)

    def on_train_epoch_start(self):
        if self._sampler is not None and hasattr(self._sampler, "set_epoch"):
            self._sampler.set_epoch(self.trainer.current_epoch)

    def _processed_path(self) -> str:
        lane = self.cfg.data.get("lane", "default")
        return f"./data/{self.cfg.data.output_name}_{lane}_processed"

    def _tokenizer_name(self) -> Optional[str]:
        model = self.cfg.get("model")
        if model is not None:
            backbone = model.get("backbone")
            if backbone is not None:
                return backbone.get("pretrained_model_name_or_path")
        return None
