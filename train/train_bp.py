# train/train_bp.py

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from experiments.metrics import MetricsLogger


def train_bp(
    model,
    train_loader,
    val_loader=None,
    num_epochs=5,
    lr=1e-3,
    max_grad_norm=5.0,
    device="cuda",
    verbose=True,
    metrics_logger=None,
):
    """
    Standard backpropagation training loop for the BP RNN baseline.

    Args:
        model: BPRNNSentiment
        train_loader: DataLoader yielding (token_ids, labels)
        val_loader: optional DataLoader for validation
        num_epochs: int
        lr: learning rate
        max_grad_norm: gradient clipping (important for RNN stability)
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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(reduction='none')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_examples = 0

        for batch in train_loader:
            token_ids = batch[0].to(device)
            labels = batch[1].to(device)
            mask = batch[2].to(device)  # (B, T) - 1 for real tokens, 0 for padding

            optimizer.zero_grad()

            logits = model(token_ids)   # (B, T, num_classes)

            # Flatten for loss computation
            B, T, C = logits.shape
            logits_flat = logits.reshape(B * T, C)
            labels_flat = labels.reshape(B * T)
            mask_flat = mask.reshape(B * T)

            loss_flat = criterion(logits_flat, labels_flat)
            # Apply mask: only compute loss on real tokens (non-padding)
            loss = (loss_flat * mask_flat).sum() / mask_flat.sum()
            loss.backward()

            # optional gradient clipping
            clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

            optimizer.step()

            # compute accuracy (only on non-padding tokens)
            preds = logits.argmax(dim=-1)  # (B, T)
            correct = ((preds == labels) * mask.bool()).sum().item()
            examples = mask.sum().item()

            total_correct += correct
            total_examples += examples
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_acc = total_correct / total_examples

        if verbose:
            print(f"[Epoch {epoch+1}/{num_epochs}] BP Train Loss: {avg_loss:.4f}  |  Train Acc: {train_acc:.4f}")

        # Log metrics
        if metrics_logger is not None:
            val_acc_log = None
            val_loss_log = None

            # -------------------------
            # Optional validation pass
            # -------------------------
            if val_loader is not None:
                val_acc_log, val_loss_log = evaluate_bp(model, val_loader, criterion, device)
                if verbose:
                    print(f"                BP Val   Loss: {val_loss_log:.4f}  |  Val Acc:   {val_acc_log:.4f}")

            metrics_logger.log_epoch(epoch, avg_loss, train_acc, val_loss_log, val_acc_log)
        else:
            # -------------------------
            # Optional validation pass
            # -------------------------
            if val_loader is not None:
                val_acc, val_loss = evaluate_bp(model, val_loader, criterion, device)
                if verbose:
                    print(f"                BP Val   Loss: {val_loss:.4f}  |  Val Acc:   {val_acc:.4f}")

    return model


def evaluate_bp(model, data_loader, criterion, device="cuda"):
    """
    Runs validation or testing on sequence tagging task.
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

            logits = model(token_ids)  # (B, T, num_classes)

            # Flatten for loss computation
            B, T, C = logits.shape
            logits_flat = logits.reshape(B * T, C)
            labels_flat = labels.reshape(B * T)
            mask_flat = mask.reshape(B * T)

            loss_flat = criterion(logits_flat, labels_flat)
            # Apply mask: only compute loss on real tokens
            loss = (loss_flat * mask_flat).sum() / mask_flat.sum()

            preds = logits.argmax(dim=-1)  # (B, T)
            correct = ((preds == labels) * mask.bool()).sum().item()
            examples = mask.sum().item()

            total_correct += correct
            total_examples += examples
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    acc = total_correct / total_examples

    return acc, avg_loss
