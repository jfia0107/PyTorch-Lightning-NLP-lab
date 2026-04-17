"""
Main training script.
Usage:
    python -m train model=bert task=binary data=encoder
    or
    python -m train with current Hydra params
"""

import os
import hydra
import torch
import lightning.pytorch as pl
from src.utils.set_vars import set_vars
from omegaconf import DictConfig, ListConfig, OmegaConf
from src.models.composer import ComposedNLPModel



@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):

    pl.seed_everything(cfg.seed, workers=True)

    # Set several QoL Pytorch/Lightning variables
    set_vars()

    # Build DataModule
    datamodule = hydra.utils.instantiate(cfg.data, cfg=cfg, _recursive_=False)
    # Build Model via ComposedNLPModel
    model = hydra.utils.instantiate(cfg.model)
    model = model.float()

    # Build Lightning Task
    task = hydra.utils.instantiate(cfg.task.task_class, model=model, cfg=cfg, _recursive_=False)

    # Instantiate Lightning Callbacks (early stopping, checkpoints, ...)
    callbacks = [hydra.utils.instantiate(cb) for cb in cfg.trainer.callbacks]
    
    # Instantiate Lightning Task
    logger = hydra.utils.instantiate(cfg.logger)
        
    
    # Instantiate Lightning Trainer
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
        _convert_="partial",
    )
    # torch.compile the built model for greater performance, if selected. May cause issues with DDP and other parallel strategies.
    if cfg.get("compile", False):
        model = torch.compile(model)

    # Lightning Fit
    trainer.fit(model=task, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    # Return monitored metric for Optuna sweeping
    metric = trainer.callback_metrics.get(cfg.task.monitor_metric)
    return metric.item() if metric is not None else 0.0


if __name__ == "__main__":
    main()
