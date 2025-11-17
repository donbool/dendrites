# Training metrics logging and tracking

import json
import os
from datetime import datetime


class MetricsLogger:
    """Log training metrics to JSON for later analysis and plotting."""

    def __init__(self, log_dir="./results", model_name="model"):
        self.log_dir = log_dir
        self.model_name = model_name
        os.makedirs(log_dir, exist_ok=True)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(
            log_dir, f"{model_name}_{self.timestamp}.json"
        )

        self.logs = {
            "model_name": model_name,
            "timestamp": self.timestamp,
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "epochs": [],
        }

    def log_epoch(self, epoch, train_loss, train_acc, val_loss=None, val_acc=None):
        """Log metrics for one epoch."""
        self.logs["epochs"].append(epoch)
        self.logs["train_loss"].append(train_loss)
        self.logs["train_acc"].append(train_acc)

        if val_loss is not None:
            self.logs["val_loss"].append(val_loss)
        if val_acc is not None:
            self.logs["val_acc"].append(val_acc)

    def save(self):
        """Save logs to JSON file."""
        with open(self.log_file, "w") as f:
            json.dump(self.logs, f, indent=2)
        print(f"Metrics saved → {self.log_file}")
        return self.log_file

    def get_logs(self):
        """Return logs dict."""
        return self.logs
