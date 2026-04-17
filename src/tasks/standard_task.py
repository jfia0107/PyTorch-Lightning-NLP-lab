"""

StandardTask for all experiments.

Covers: bert+cls, bert+bilstm, bert+cnn, bilstm, cnn, ...
"""

import torch
from omegaconf import DictConfig
from src.tasks.base_task import BaseTask


class StandardTask(BaseTask):

    def __init__(self, model, cfg: DictConfig):
        super().__init__(model=model, cfg=cfg)

    def _shared_step(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        labels = batch["label"]

        outputs = self.model(input_ids, attention_mask)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs

        if logits.shape != labels.shape:
            logits = logits.squeeze(-1)

        loss = self.loss_fn(logits, labels.float())
        return loss, logits, labels

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, targets = self._shared_step(batch)
        preds_probs = torch.sigmoid(logits)
        self._log_val_metrics(loss, preds_probs, targets)
        return loss
