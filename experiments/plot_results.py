# accuracy/loss curves

import json
import matplotlib.pyplot as plt


def load_log(path):
    with open(path, "r") as f:
        return json.load(f)


def plot_curves(bp_log_path, dll_log_path):
    bp = load_log(bp_log_path)
    dll = load_log(dll_log_path)

    # -----------------------
    # Accuracy curves
    # -----------------------
    plt.figure(figsize=(8, 5))
    plt.plot(bp["val_acc"], label="BP Val Acc")
    plt.plot(dll["val_acc"], label="DLL Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy: BP vs DLL")
    plt.legend()
    plt.grid(True)
    plt.savefig("val_accuracy_bp_vs_dll.png")
    print("Saved plot → val_accuracy_bp_vs_dll.png")

    # -----------------------
    # Training Acc curves
    # -----------------------
    plt.figure(figsize=(8, 5))
    plt.plot(bp["train_acc"], label="BP Train Acc")
    plt.plot(dll["train_acc"], label="DLL Train Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train Accuracy: BP vs DLL")
    plt.legend()
    plt.grid(True)
    plt.savefig("train_accuracy_bp_vs_dll.png")
    print("Saved plot → train_accuracy_bp_vs_dll.png")
