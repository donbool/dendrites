# Comprehensive evaluation and metrics computation

import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
import numpy as np


def evaluate_model(model, data_loader, model_type="bp", device="cuda"):
    """
    Comprehensive evaluation of a trained model on sequence tagging task.

    Args:
        model: Trained model (BPRNNSentiment or DLLSentimentRNN)
        data_loader: DataLoader for evaluation
        model_type: "bp" or "dll"
        device: "cuda" or "cpu"

    Returns:
        dict with metrics: acc, precision, recall, f1, loss, auc, conf_matrix
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_logits = []
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in data_loader:
            token_ids = batch[0].to(device)
            labels = batch[1].to(device)
            mask = batch[2].to(device)  # (B, T) - 1 for real tokens, 0 for padding

            # Forward pass
            logits = model(token_ids)  # (B, T, num_classes)

            # Flatten for loss computation
            B, T, C = logits.shape
            logits_flat = logits.reshape(B * T, C)
            labels_flat = labels.reshape(B * T)
            mask_flat = mask.reshape(B * T)

            # Loss (CE) with masking - only compute on real tokens
            loss_flat = F.cross_entropy(logits_flat, labels_flat, reduction='none')
            loss = (loss_flat * mask_flat).sum() / mask_flat.sum()
            total_loss += loss.item()
            num_batches += 1

            # Predictions - only on real tokens
            preds = torch.argmax(logits, dim=-1)  # (B, T)
            preds_flat = preds.reshape(B * T)

            # Apply mask: only keep predictions for real tokens
            valid_preds = preds_flat[mask_flat.bool()].cpu().numpy()
            valid_labels = labels_flat[mask_flat.bool()].cpu().numpy()
            valid_logits = logits_flat[mask_flat.bool()].cpu().numpy()

            # Store for metrics
            all_preds.append(valid_preds)
            all_labels.append(valid_labels)
            all_logits.append(valid_logits)

    # Concatenate all batches
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_logits = np.concatenate(all_logits)
    avg_loss = total_loss / num_batches

    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0, average='weighted')
    recall = recall_score(all_labels, all_preds, zero_division=0, average='weighted')
    f1 = f1_score(all_labels, all_preds, zero_division=0, average='weighted')
    conf_mat = confusion_matrix(all_labels, all_preds)

    # AUC (only if binary classification, otherwise set to 0)
    num_classes = all_logits.shape[1]
    if num_classes == 2 and len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_logits[:, 1])
    else:
        auc = 0.0

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "loss": avg_loss,
        "auc": auc,
        "confusion_matrix": conf_mat.tolist(),
        "num_correct": int(np.sum(all_preds == all_labels)),
        "num_total": len(all_labels),
    }


def print_metrics(metrics, split="val"):
    """Pretty-print evaluation metrics."""
    print(f"\n{'='*60}")
    print(f"  {split.upper()} METRICS")
    print(f"{'='*60}")
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  F1 Score:   {metrics['f1']:.4f}")
    print(f"  AUC-ROC:    {metrics['auc']:.4f}")
    print(f"  Loss:       {metrics['loss']:.4f}")
    print(f"  Correct:    {metrics['num_correct']}/{metrics['num_total']}")

    # Confusion matrix
    cm = metrics["confusion_matrix"]
    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0][0]}, FP={cm[0][1]}")
    print(f"    FN={cm[1][0]}, TP={cm[1][1]}")
    print(f"{'='*60}\n")


def compare_models(bp_metrics, dll_metrics):
    """Compare BP vs DLL performance."""
    print(f"\n{'='*70}")
    print(f"  BP vs DLL COMPARISON")
    print(f"{'='*70}")

    metrics_to_compare = ["accuracy", "precision", "recall", "f1", "auc", "loss"]

    for metric in metrics_to_compare:
        bp_val = bp_metrics[metric]
        dll_val = dll_metrics[metric]
        diff = dll_val - bp_val
        pct = (diff / bp_val * 100) if bp_val != 0 else 0

        winner = "DLL" if (metric != "loss" and diff > 0) or (metric == "loss" and diff < 0) else "BP"

        print(f"\n  {metric.upper()}")
        print(f"    BP:       {bp_val:.4f}")
        print(f"    DLL:      {dll_val:.4f}")
        print(f"    Diff:     {diff:+.4f} ({pct:+.2f}%) → {winner}")

    print(f"{'='*70}\n")
