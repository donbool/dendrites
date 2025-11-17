# train/train_bp.py

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_


def train_bp(
    model,
    train_loader,
    val_loader=None,
    num_epochs=5,
    lr=1e-3,
    max_grad_norm=5.0,
    device="cuda",
    verbose=True,
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
        device: "cuda" or "cpu"
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_examples = 0

        for token_ids, labels in train_loader:
            token_ids = token_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(token_ids)   # (B, num_classes)
            loss = criterion(logits, labels)
            loss.backward()

            # optional gradient clipping
            clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

            optimizer.step()

            # compute accuracy
            preds = logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_examples += labels.size(0)

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_acc = total_correct / total_examples

        if verbose:
            print(f"[Epoch {epoch+1}/{num_epochs}] BP Train Loss: {avg_loss:.4f}  |  Train Acc: {train_acc:.4f}")

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
    Runs validation or testing.
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
            loss = criterion(logits, labels)

            preds = logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_examples += labels.size(0)

            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    acc = total_correct / total_examples

    return acc, avg_loss
