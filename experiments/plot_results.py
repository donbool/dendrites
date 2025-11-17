# Comprehensive plotting and visualization for paper

import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np


def load_log(path):
    """Load metrics JSON log."""
    with open(path, "r") as f:
        return json.load(f)


def find_latest_logs(log_dir="./results"):
    """Find latest BP and DLL log files."""
    bp_logs = sorted(glob.glob(os.path.join(log_dir, "bp_*.json")))
    dll_logs = sorted(glob.glob(os.path.join(log_dir, "dll_*.json")))

    if not bp_logs or not dll_logs:
        raise FileNotFoundError(
            f"No logs found in {log_dir}. Run training first with metrics logging."
        )

    return bp_logs[-1], dll_logs[-1]


def plot_training_curves(bp_log_path, dll_log_path, output_dir="./results"):
    """Plot training curves for both models."""
    bp = load_log(bp_log_path)
    dll = load_log(dll_log_path)

    os.makedirs(output_dir, exist_ok=True)

    # ----- 1. Validation Accuracy -----
    plt.figure(figsize=(10, 6))
    plt.plot(bp["epochs"], bp["val_acc"], "o-", linewidth=2, markersize=5, label="BP")
    plt.plot(dll["epochs"], dll["val_acc"], "s-", linewidth=2, markersize=5, label="DLL")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Validation Accuracy: BP vs DLL", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "val_accuracy_comparison.png"), dpi=300)
    print("Saved → val_accuracy_comparison.png")
    plt.close()

    # ----- 2. Training Accuracy -----
    plt.figure(figsize=(10, 6))
    plt.plot(bp["epochs"], bp["train_acc"], "o-", linewidth=2, markersize=5, label="BP")
    plt.plot(dll["epochs"], dll["train_acc"], "s-", linewidth=2, markersize=5, label="DLL")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Training Accuracy: BP vs DLL", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "train_accuracy_comparison.png"), dpi=300)
    print("Saved → train_accuracy_comparison.png")
    plt.close()

    # ----- 3. Validation Loss -----
    plt.figure(figsize=(10, 6))
    plt.plot(bp["epochs"], bp["val_loss"], "o-", linewidth=2, markersize=5, label="BP")
    plt.plot(dll["epochs"], dll["val_loss"], "s-", linewidth=2, markersize=5, label="DLL")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Validation Loss: BP vs DLL", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "val_loss_comparison.png"), dpi=300)
    print("Saved → val_loss_comparison.png")
    plt.close()

    # ----- 4. Training Loss -----
    plt.figure(figsize=(10, 6))
    plt.plot(bp["epochs"], bp["train_loss"], "o-", linewidth=2, markersize=5, label="BP")
    plt.plot(dll["epochs"], dll["train_loss"], "s-", linewidth=2, markersize=5, label="DLL")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training Loss: BP vs DLL", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "train_loss_comparison.png"), dpi=300)
    print("Saved → train_loss_comparison.png")
    plt.close()

    # ----- 5. Combined 2x2 Grid -----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(bp["epochs"], bp["train_loss"], "o-", label="BP", linewidth=2)
    axes[0, 0].plot(dll["epochs"], dll["train_loss"], "s-", label="DLL", linewidth=2)
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training Loss", fontweight="bold")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(bp["epochs"], bp["val_loss"], "o-", label="BP", linewidth=2)
    axes[0, 1].plot(dll["epochs"], dll["val_loss"], "s-", label="DLL", linewidth=2)
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].set_title("Validation Loss", fontweight="bold")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(bp["epochs"], bp["train_acc"], "o-", label="BP", linewidth=2)
    axes[1, 0].plot(dll["epochs"], dll["train_acc"], "s-", label="DLL", linewidth=2)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].set_title("Training Accuracy", fontweight="bold")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(bp["epochs"], bp["val_acc"], "o-", label="BP", linewidth=2)
    axes[1, 1].plot(dll["epochs"], dll["val_acc"], "s-", label="DLL", linewidth=2)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].set_title("Validation Accuracy", fontweight="bold")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("BP vs DLL: Training Metrics", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_summary_2x2.png"), dpi=300)
    print("Saved → training_summary_2x2.png")
    plt.close()

    return bp, dll


if __name__ == "__main__":
    print("Finding latest logs...")
    bp_path, dll_path = find_latest_logs()
    print(f"BP log:  {bp_path}")
    print(f"DLL log: {dll_path}")

    print("\nGenerating plots...")
    plot_training_curves(bp_path, dll_path)
