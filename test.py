"""
Run inference on test split, compute metrics, save predictions into csv.

Usage:
    python -m test ckpt_path=checkpoint.ckpt or set ckpt_path in Hydra config

All outputs can be found in the created results/ folder.
"""
# Threshold for F1
THRESHOLD = 0.5




import os
import json
import logging
import hydra
import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from omegaconf import DictConfig
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

log = logging.getLogger(__name__)




def run_inference(model, dataloader, device):
    all_preds = []
    all_labels = []

    model.eval()

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Running inference", leave=False):
            input_ids = batch["input_ids"].to(device)
            
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            labels = batch["label"]
            outputs = model(input_ids, attention_mask)

            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            preds = torch.sigmoid(logits).squeeze(-1) 
            
            all_preds.append(preds.cpu().float())
            all_labels.append(labels.cpu())

    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model checkpoint
    ckpt_path = cfg.get("ckpt_path")
    assert ckpt_path is not None, "Set ckpt_path=path/to/checkpoint.ckpt"
    log.info(f"Loading checkpoint: {ckpt_path}")
    # Build model and task as if this was train.py but load model from the checkpoint and put model into eval mode
    from src.tasks.standard_task import StandardTask

    model_skeleton = hydra.utils.instantiate(cfg.model)
    task = StandardTask.load_from_checkpoint(ckpt_path, model=model_skeleton, cfg=cfg, weights_only=False)

    task.model = task.model.to(device).float()
    task.model.eval()
    model = torch.compile(task.model)

    lane = cfg.data.get("lane", "encoder")
    processed_path = f"./data/{cfg.data.output_name}_{lane}_processed"

    assert os.path.exists(processed_path), f"Data not found: {processed_path}"
    dataset = load_from_disk(processed_path)
    test_data = dataset["test"]
    test_data.set_format(type="torch")

    # Select dataset class based on CDL/encoder
    tokenizer_name = cfg.get("model", {}).get("backbone", {}).get("pretrained_model_name_or_path")
    if tokenizer_name:
        from src.data.dataset_classes import EncoderDataset
        test_ds = EncoderDataset(test_data)
    else:
        from src.data.dataset_classes import CDLDataset
        test_ds = CDLDataset(test_data)

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.data.get("batch_size", 128),
        shuffle=False,
        num_workers=cfg.data.get("num_workers", 4),
        pin_memory=True,
    )

    log.info(f"Test samples amount: {len(test_ds)}")

    # Run inference
    preds, labels = run_inference(model, test_loader, device)
    log.info(f"Inference complete: {len(preds)} predictions")
    

    binary_labels = (labels >= THRESHOLD).astype(int)
    binary_preds = (preds >= THRESHOLD).astype(int)
    binary_labels_strict = (labels > 0.0).astype(int)
    
    # Core metrics
    auc = roc_auc_score(binary_labels, preds)
    ap_score = average_precision_score(binary_labels_strict, preds)
    f1 = f1_score(binary_labels, binary_preds, zero_division=0)
    precision_val = precision_score(binary_labels, binary_preds, zero_division=0)
    recall_val = recall_score(binary_labels, binary_preds, zero_division=0)
    # Confusion matrix
    optimal_binary_preds = (preds >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(binary_labels, optimal_binary_preds).ravel()
    # Calculate optimal threshold for maximum F1
    precisions, recalls, thresholds_arr = precision_recall_curve(binary_labels, preds)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_f1_idx = np.argmax(f1_scores)
    optimal_threshold = float(thresholds_arr[best_f1_idx]) if best_f1_idx < len(thresholds_arr) else 0.5
    best_f1 = float(f1_scores[best_f1_idx])
    
    


    metrics = {
        "auc": float(auc),
        "f1_baseline": float(f1),
        "f1_optimal": best_f1,
        "optimal_threshold": optimal_threshold,
        "precision": float(precision_val),
        "recall": float(recall_val),
        "pred_mean": float(preds.mean()),
        "pred_std": float(preds.std()),
        "n_test": len(preds),
        "checkpoint": ckpt_path,
        "model": cfg.model.name,
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn)
    }

    # CLI logging
    log.info("RESULTS:")
    log.info(f"AUC:              {auc:.4f}")
    log.info(f"Average Precision:{ap_score:.4f}")
    log.info(f"F1 (Baseline):    {f1:.4f} (at threshold 0.5)")
    log.info(f"F1 (Optimal):     {best_f1:.4f} (at threshold {optimal_threshold:.4f})")
    log.info(f"Precision:        {precision_val:.4f}")
    log.info(f"Recall:           {recall_val:.4f}")
    log.info(f"True Positives:   {int(tp)}")
    log.info(f"True Negatives:   {int(tn)}")
    log.info(f"False Positives:  {int(fp)}")
    log.info(f"False Negatives:  {int(fn)}")

    # Create results folder
    out_dir = f"./results/{cfg.model.name}"
    os.makedirs(out_dir, exist_ok=True)

    # Save metrics as json
    metrics_path = f"{out_dir}/test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Metrics saved to {metrics_path}")

    # Save preds with text as csv
    raw_dataset = load_from_disk(processed_path)["test"]
    texts = raw_dataset["text"] if "text" in raw_dataset.column_names else [""] * len(preds)

    df = pd.DataFrame({
        "text": texts[:len(preds)],
        "label": labels,
        "binary_label": binary_labels,
        "prediction_prob": preds,
        "binary_pred": binary_preds,
    })

    df = df.sort_values(by="label", ascending=False)

    preds_path = f"{out_dir}/test_predictions.csv"
    df.to_csv(preds_path, index=False)
    log.info(f"Predictions saved to {preds_path}")

    return metrics

if __name__ == "__main__":
    main()
