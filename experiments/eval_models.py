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
    Comprehensive evaluation of a trained model.

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
        for token_ids, labels in data_loader:
            token_ids = token_ids.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(token_ids)  # (B, 2)

            # Loss (CE)
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item()
            num_batches += 1

            # Predictions
            preds = torch.argmax(logits, dim=-1)  # (B,)

            # Store for metrics
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_logits.append(logits.cpu().numpy())

    # Concatenate all batches
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_logits = np.concatenate(all_logits)
    avg_loss = total_loss / num_batches

    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    conf_mat = confusion_matrix(all_labels, all_preds)

    # AUC (for positive class)
    if len(np.unique(all_labels)) > 1:
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
