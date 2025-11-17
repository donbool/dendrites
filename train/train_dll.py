# train/train_dll.py

import torch
import torch.nn.functional as F
from experiments.metrics import MetricsLogger


def train_dll(
    model,
    train_loader,
    val_loader=None,
    num_epochs=5,
    device="cuda",
    verbose=True,
    metrics_logger=None,
):
    """
    Training loop for DLL, matching the DLL algorithm in the paper.

    Args:
        model: DLLSentimentRNN (wrapper around DLL_RNN_Model)
        train_loader: dataloader yielding (token_ids, labels)
        val_loader: optional validation loader
        num_epochs: int
        device: "cuda" or "cpu"
    """

    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()

        total_loss = 0
        total_correct = 0
        total_examples = 0

        for token_ids, labels in train_loader:

            token_ids = token_ids.to(device)
            labels = labels.to(device)

            # --- 1. Forward pass ---
            logits = model(token_ids)  # (B, C)

            # --- 2. Logging CE loss (monitoring only) ---
            with torch.no_grad():
                log_probs = F.log_softmax(logits, dim=-1)
                ce_loss = -(log_probs[torch.arange(labels.size(0)), labels]).mean()

            total_loss += ce_loss.item()

            # --- 3. Accuracy ---
            preds = torch.argmax(logits, dim=-1)
            total_correct += (preds == labels).sum().item()
            total_examples += labels.size(0)

            # --- 4. DLL update (actual learning happens here, no backprop) ---
            model.dll_update(labels, epoch)

        avg_loss = total_loss / len(train_loader)
        train_acc = total_correct / total_examples

        if verbose:
            print(f"[Epoch {epoch+1}/{num_epochs}] DLL Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f}")

        # Log metrics
        if metrics_logger is not None:
            val_acc_log = None
            val_loss_log = None

            # ----------------------------------
            # Optional validation
            # ----------------------------------
            if val_loader is not None:
                val_acc_log, val_loss_log = evaluate_dll(model, val_loader, device)
                if verbose:
                    print(f"                   DLL Val   Loss: {val_loss_log:.4f} | Val Acc:   {val_acc_log:.4f}")

            metrics_logger.log_epoch(epoch, avg_loss, train_acc, val_loss_log, val_acc_log)
        else:
            # ----------------------------------
            # Optional validation
            # ----------------------------------
            if val_loader is not None:
                val_acc, val_loss = evaluate_dll(model, val_loader, device)
                if verbose:
                    print(f"                   DLL Val   Loss: {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

    return model



def evaluate_dll(model, data_loader, device="cuda"):
    """
    Standard evaluation loop (same as BP evaluation, DLL has no special eval).
    """
    model.eval()
    total_loss = 0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for token_ids, labels in data_loader:

            token_ids = token_ids.to(device)
            labels = labels.to(device)

            logits = model(token_ids)
            log_probs = F.log_softmax(logits, dim=-1)
            ce_loss = -(log_probs[torch.arange(labels.size(0)), labels]).mean()

            preds = torch.argmax(logits, dim=-1)
            total_correct += (preds == labels).sum().item()
            total_examples += labels.size(0)

            total_loss += ce_loss.item()

    avg_loss = total_loss / len(data_loader)
    acc = total_correct / total_examples

    return acc, avg_loss
