# train/train_hybrid.py
# Training loop for Hybrid DLL (DLL + temporal credit via eligibility traces)

import torch
import torch.nn.functional as F
from experiments.metrics import MetricsLogger


def train_hybrid(
    model,
    train_loader,
    val_loader=None,
    num_epochs=5,
    device="cuda",
    verbose=True,
    metrics_logger=None,
):
    """
    Training loop for Hybrid DLL model on POS tagging (DLL with eligibility traces).

    Args:
        model: HybridDLLRNN (hybrid variant with eligibility traces)
        train_loader: dataloader yielding (token_ids, labels, mask)
        val_loader: optional validation loader
        num_epochs: int
        device: "cuda", "mps", or "cpu"
    """
    # Ensure device is set up correctly (mps, cuda, or cpu)
    if not device:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()

        total_loss = 0
        total_correct = 0
        total_examples = 0

        for batch in train_loader:

            token_ids = batch[0].to(device)
            labels = batch[1].to(device)
            mask = batch[2].to(device)  # (B, T)

            # --- 1. Forward pass ---
            logits = model(token_ids)  # (B, T, C)

            # --- 2. Logging CE loss (monitoring only) ---
            with torch.no_grad():
                B, T, C = logits.shape
                logits_flat = logits.reshape(B * T, C)
                labels_flat = labels.reshape(B * T)
                mask_flat = mask.reshape(B * T)

                log_probs = F.log_softmax(logits_flat, dim=-1)
                ce_loss_flat = -(log_probs[torch.arange(B * T), labels_flat])
                ce_loss = (ce_loss_flat * mask_flat).sum() / mask_flat.sum()

            total_loss += ce_loss.item()

            # --- 3. Accuracy (only on non-padding tokens) ---
            preds = torch.argmax(logits, dim=-1)  # (B, T)
            correct = ((preds == labels) * mask.bool()).sum().item()
            examples = mask.sum().item()

            total_correct += correct
            total_examples += examples

            # --- 4. Hybrid update (DLL + temporal credit via traces) ---
            model.dll_update(labels, mask, epoch)

        avg_loss = total_loss / len(train_loader)
        train_acc = total_correct / total_examples

        if verbose:
            print(f"[Epoch {epoch+1}/{num_epochs}] Hybrid Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f}")

        # Log metrics
        if metrics_logger is not None:
            val_acc_log = None
            val_loss_log = None

            # ----------------------------------
            # Optional validation
            # ----------------------------------
            if val_loader is not None:
                val_acc_log, val_loss_log = evaluate_hybrid(model, val_loader, device)
                if verbose:
                    print(f"                     Hybrid Val   Loss: {val_loss_log:.4f} | Val Acc:   {val_acc_log:.4f}")

            metrics_logger.log_epoch(epoch, avg_loss, train_acc, val_loss_log, val_acc_log)
        else:
            # ----------------------------------
            # Optional validation
            # ----------------------------------
            if val_loader is not None:
                val_acc, val_loss = evaluate_hybrid(model, val_loader, device)
                if verbose:
                    print(f"                     Hybrid Val   Loss: {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

    return model


def evaluate_hybrid(model, data_loader, device="cuda"):
    """
    Evaluation loop for Hybrid DLL on sequence tagging task.
    """
    model.eval()
    total_loss = 0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for batch in data_loader:

            token_ids = batch[0].to(device)
            labels = batch[1].to(device)
            mask = batch[2].to(device)  # (B, T)

            logits = model(token_ids)  # (B, T, C)

            B, T, C = logits.shape
            logits_flat = logits.reshape(B * T, C)
            labels_flat = labels.reshape(B * T)
            mask_flat = mask.reshape(B * T)

            log_probs = F.log_softmax(logits_flat, dim=-1)
            ce_loss_flat = -(log_probs[torch.arange(B * T), labels_flat])
            ce_loss = (ce_loss_flat * mask_flat).sum() / mask_flat.sum()

            preds = torch.argmax(logits, dim=-1)  # (B, T)
            correct = ((preds == labels) * mask.bool()).sum().item()
            examples = mask.sum().item()

            total_correct += correct
            total_examples += examples
            total_loss += ce_loss.item()

    avg_loss = total_loss / len(data_loader)
    acc = total_correct / total_examples

    return acc, avg_loss
