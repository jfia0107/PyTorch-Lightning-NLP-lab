"""
Parent task - all shared Lightning Task logic lives here.
standard_task.StandardTask is a child of this class
"""

import logging
from typing import Any, Dict, List
from contextlib import nullcontext
import torch
import torch.nn as nn
import hydra
import lightning.pytorch as pl
from omegaconf import DictConfig
from torchmetrics import F1Score, Precision, Recall, AUROC
import numpy as np

log = logging.getLogger(__name__)


class BaseTask(pl.LightningModule):

    def __init__(self, model: nn.Module, cfg: DictConfig):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.loss_fn = hydra.utils.instantiate(cfg.task.loss_fn)
        self._freeze_backbone()

        # Binary classification metrics
        self.val_f1 = F1Score(task="binary")
        self.val_precision = Precision(task="binary")
        self.val_recall = Recall(task="binary")
        self.val_auc = AUROC(task="binary")

        # Epoch-level accumulators
        self._val_preds: List[torch.Tensor] = []
        self._val_targets: List[torch.Tensor] = []

        self.save_hyperparameters(ignore=["model"])

    def on_train_start(self):
        # This is an attempt to fail fast on misconfigured runs
        # validate dtypes, shapes, and forward+loss on first batch
        batch = next(iter(self.trainer.train_dataloader))
        device = next(self.model.parameters()).device
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Assert Input shape
        assert batch["input_ids"].ndim == 2, f"input_ids shape {batch['input_ids'].shape}, expected (B, T)"

        # Dry-run forward
        if self.trainer.precision in ("16-mixed", "bf16-mixed"):
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16 if "bf16" in self.trainer.precision else torch.float16)
        else:
            autocast_ctx = nullcontext()

        with torch.no_grad(), autocast_ctx:
            try:
                self._dry_run(batch)
            except Exception as e:
                log.error(f"Dry-run forward+loss failed: {e}")
                raise
        if self.cfg.get("gradient_checkpointing", False):
            if hasattr(self.model, "backbone") and hasattr(self.model.backbone, "gradient_checkpointing_enable"):
                self.model.backbone.gradient_checkpointing_enable()
                log.info("Gradient checkpointing enabled")

    def _dry_run(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        labels = batch["label"]

        outputs = self.model(input_ids, attention_mask)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        if logits.shape != labels.shape:
            logits = logits.squeeze(-1)
        loss = self.loss_fn(logits, labels)
        log.info(f"Dry-run passed.")

    def _log_val_metrics(self, loss, preds_probs, targets):
        # Update validation metrics
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

        # Binarize targets for classification metrics if continuous, hardcoded to 0.5
        binary_targets = (targets >= 0.5).long() if targets.is_floating_point() else targets

        self.val_f1.update(preds_probs, binary_targets)
        self.val_precision.update(preds_probs, binary_targets)
        self.val_recall.update(preds_probs, binary_targets)
        self.val_auc.update(preds_probs, binary_targets)

        self._val_preds.append(preds_probs.detach().cpu())
        self._val_targets.append(targets.detach().cpu())

    def on_validation_epoch_end(self):
        
        # Torchmetrics:
        self.log("val_f1", self.val_f1.compute(), prog_bar=True, sync_dist=True)
        self.log("val_precision", self.val_precision.compute(), sync_dist=True)
        self.log("val_recall", self.val_recall.compute(), sync_dist=True)
        self.log("val_auc", self.val_auc.compute(), prog_bar=True, sync_dist=True)

        self.val_f1.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_auc.reset()

        if len(self._val_preds) == 0:
            return

        preds = torch.cat(self._val_preds).float().numpy()
        targets = torch.cat(self._val_targets).float().numpy()

        # Class separation metric
        binary_targets = (targets >= 0.5).astype(int)
        pos_preds = preds[binary_targets == 1]
        neg_preds = preds[binary_targets == 0]
        if len(pos_preds) > 0 and len(neg_preds) > 0:
            self.log("val_class_separation", pos_preds.mean() - neg_preds.mean())

        self._val_preds = []
        self._val_targets = []

    def on_train_epoch_start(self):
        # Backbone freezong logic activation, set in hydra
        if self.cfg.get("freeze_backbone", False) and hasattr(self.model, "backbone"):
            self.model.backbone.eval()

    def configure_optimizers(self):
        # setup optimizer
        cfg_opt = self.cfg.optimizer

        bert_lr = float(cfg_opt.get("bert_lr", cfg_opt.get("lr", 1e-5)))
        head_lr = float(cfg_opt.get("head_lr", cfg_opt.get("lr", 1e-3)))
        w_decay = float(cfg_opt.get("weight_decay", 0.0))

        param_groups = []

        # Backbone params
        if hasattr(self.model, "backbone") and self.model.backbone is not None:
            backbone_params = [p for p in self.model.backbone.parameters() if p.requires_grad]
            if backbone_params:
                param_groups.append({"params": backbone_params, "lr": bert_lr})

        # Head params
        if hasattr(self.model, "head"):
            head_params = [p for p in self.model.head.parameters() if p.requires_grad]
            if head_params:
                param_groups.append({"params": head_params, "lr": head_lr})

        # Fallback if nothing was added
        if not param_groups:
            param_groups = [{"params": [p for p in self.model.parameters() if p.requires_grad], "lr": head_lr}]

        opt_class = hydra.utils.get_class(cfg_opt._target_)
        optimizer = opt_class(params=param_groups, weight_decay=w_decay)
        # scheduler setup (if selected in hydra - probably only for transformers)
        cfg_sched = self.cfg.get("scheduler")
        if cfg_sched and cfg_sched.get("_target_"):
            total_steps = self.trainer.estimated_stepping_batches
            warmup_steps = int(total_steps * self.cfg.get("warmup_ratio", 0.1))

            scheduler = hydra.utils.instantiate(
                cfg_sched,
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }

        return optimizer

    def _freeze_backbone(self):
        # backbone freezing logic
        if not hasattr(self.model, "backbone") or self.model.backbone is None:
            return
        if not self.cfg.get("freeze_backbone", False):
            log.info("Backbone: fully trainable")
            return

        for param in self.model.backbone.parameters():
            param.requires_grad = False

        unfreeze_n = self.cfg.get("unfreeze_top_n", 0)
        if unfreeze_n > 0 and hasattr(self.model.backbone, "encoder"):
            for layer in self.model.backbone.encoder.layer[-unfreeze_n:]:
                for param in layer.parameters():
                    param.requires_grad = True
            log.info(f"Backbone: frozen, top {unfreeze_n} layers unfrozen")
        else:
            log.info("Backbone: completely frozen")
