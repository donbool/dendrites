"""
Visualization and Analysis for DLL RNN Research Paper

Generates publication-quality figures and tables for comparing:
- Backprop baseline
- Standard DLL
- Hierarchical DLL
- Hybrid DLL (with traces)
- Temporal DLL
- Predictive Coding DLL
- Bidirectional DLL
"""

import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import pandas as pd

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("colorblind")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13


class ResultsVisualizer:
    """
    Load metrics from all trained models and generate publication figures.
    """

    def __init__(self, results_dir="./results"):
        self.results_dir = Path(results_dir)
        self.metrics = {}
        self.load_all_metrics()

    def load_all_metrics(self):
        """Load metrics JSON files for all models."""
        json_files = glob.glob(str(self.results_dir / "*.json"))

        for json_file in json_files:
            filename = Path(json_file).stem
            # Extract model name (before timestamp)
            # Format: modelname_YYYYMMDD_HHMMSS.json
            parts = filename.split('_')
            if len(parts) >= 2:
                model_name = parts[0]
            else:
                model_name = filename

            with open(json_file, 'r') as f:
                data = json.load(f)

            # Store with timestamp to handle multiple runs
            self.metrics[filename] = {
                'model_name': model_name,
                'data': data
            }

    def get_latest_run(self, model_name):
        """Get the most recent run for a given model."""
        matching = [(k, v) for k, v in self.metrics.items()
                   if v['model_name'] == model_name]

        if not matching:
            return None

        # Sort by timestamp (embedded in filename)
        matching.sort(key=lambda x: x[0], reverse=True)
        return matching[0][1]['data']

    def plot_training_curves(self, models=None, save_path="figures/training_curves.png"):
        """
        Plot training and validation accuracy/loss curves for all models.

        Creates 2x2 subplot:
        - Top left: Training accuracy
        - Top right: Validation accuracy
        - Bottom left: Training loss
        - Bottom right: Validation loss
        """
        if models is None:
            models = ['bp', 'dll', 'hierarchical', 'hybrid', 'predictive', 'bidirectional']

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
        model_display_names = {
            'bp': 'Backprop',
            'dll': 'DLL',
            'hierarchical': 'Hierarchical DLL',
            'hybrid': 'Hybrid DLL',
            'temporal': 'Temporal DLL',
            'predictive': 'Predictive DLL',
            'bidirectional': 'Bidirectional DLL'
        }

        for idx, model in enumerate(models):
            data = self.get_latest_run(model)
            if data is None:
                continue

            epochs = data['epochs']
            train_acc = data.get('train_acc', [])
            val_acc = data.get('val_acc', [])
            train_loss = data.get('train_loss', [])
            val_loss = data.get('val_loss', [])

            label = model_display_names.get(model, model.upper())
            color = colors[idx]

            # Training accuracy
            axes[0, 0].plot(epochs, train_acc, label=label, color=color, linewidth=2)

            # Validation accuracy
            if val_acc:
                axes[0, 1].plot(epochs, val_acc, label=label, color=color, linewidth=2)

            # Training loss
            axes[1, 0].plot(epochs, train_loss, label=label, color=color, linewidth=2)

            # Validation loss
            if val_loss:
                axes[1, 1].plot(epochs, val_loss, label=label, color=color, linewidth=2)

        # Formatting
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Training Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Training Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Validation Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved training curves → {save_path}")
        plt.close()

    def plot_final_comparison(self, models=None, save_path="figures/final_comparison.png"):
        """
        Bar chart comparing final validation accuracy across all models.
        """
        if models is None:
            models = ['bp', 'dll', 'hierarchical', 'hybrid', 'predictive', 'bidirectional']

        model_display_names = {
            'bp': 'Backprop',
            'dll': 'DLL',
            'hierarchical': 'Hierarchical\nDLL',
            'hybrid': 'Hybrid\nDLL',
            'temporal': 'Temporal\nDLL',
            'predictive': 'Predictive\nDLL',
            'bidirectional': 'Bidirectional\nDLL'
        }

        accuracies = []
        labels = []
        colors_list = []

        # Define colors: BP in one color, DLL variants in another
        bp_color = '#1f77b4'  # Blue
        dll_color = '#ff7f0e'  # Orange

        for model in models:
            data = self.get_latest_run(model)
            if data is None:
                continue

            val_acc = data.get('val_acc', [])
            if val_acc:
                final_acc = val_acc[-1] * 100  # Convert to percentage
                accuracies.append(final_acc)
                labels.append(model_display_names.get(model, model.upper()))

                # Color coding
                if model == 'bp':
                    colors_list.append(bp_color)
                else:
                    colors_list.append(dll_color)

        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(labels, accuracies, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.2)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}%',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
        ax.set_title('Final Validation Accuracy Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(True, axis='y', alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=bp_color, edgecolor='black', label='Backprop (baseline)'),
            Patch(facecolor=dll_color, edgecolor='black', label='DLL variants')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved final comparison → {save_path}")
        plt.close()

    def plot_dll_variants_comparison(self, save_path="figures/dll_variants_only.png"):
        """
        Focused comparison: Standard DLL vs all extensions.
        Shows how each extension performs relative to baseline DLL.
        """
        models = ['dll', 'hierarchical', 'hybrid', 'predictive', 'bidirectional']

        model_display_names = {
            'dll': 'Standard DLL\n(baseline)',
            'hierarchical': 'Hierarchical\nDLL',
            'hybrid': 'Hybrid\nDLL',
            'predictive': 'Predictive\nDLL',
            'bidirectional': 'Bidirectional\nDLL'
        }

        # Get DLL baseline
        dll_data = self.get_latest_run('dll')
        if dll_data is None:
            print("Warning: No DLL baseline found")
            return

        dll_acc = dll_data.get('val_acc', [])[-1] * 100 if dll_data.get('val_acc') else 0

        accuracies = []
        deltas = []
        labels = []

        for model in models:
            data = self.get_latest_run(model)
            if data is None:
                continue

            val_acc = data.get('val_acc', [])
            if val_acc:
                final_acc = val_acc[-1] * 100
                delta = final_acc - dll_acc

                accuracies.append(final_acc)
                deltas.append(delta)
                labels.append(model_display_names.get(model, model.upper()))

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Absolute accuracy
        colors = ['#2ca02c' if model == 'dll' else '#ff7f0e' for model in models[:len(labels)]]
        bars1 = axes[0].bar(labels, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)

        for bar in bars1:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}%',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

        axes[0].set_ylabel('Validation Accuracy (%)', fontsize=12)
        axes[0].set_title('Absolute Performance', fontsize=13, fontweight='bold')
        axes[0].set_ylim(0, 100)
        axes[0].grid(True, axis='y', alpha=0.3)

        # Right: Relative to DLL baseline
        colors_delta = ['green' if d >= 0 else 'red' for d in deltas]
        bars2 = axes[1].bar(labels, deltas, color=colors_delta, alpha=0.7, edgecolor='black', linewidth=1.2)

        for bar, delta in zip(bars2, deltas):
            height = bar.get_height()
            va = 'bottom' if delta >= 0 else 'top'
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{delta:+.2f}%',
                        ha='center', va=va, fontsize=9, fontweight='bold')

        axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1.5)
        axes[1].set_ylabel('Accuracy Difference vs Standard DLL (%)', fontsize=12)
        axes[1].set_title('Relative Improvement', fontsize=13, fontweight='bold')
        axes[1].grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved DLL variants comparison → {save_path}")
        plt.close()

    def create_results_table(self, models=None, save_path="figures/results_table.txt"):
        """
        Create LaTeX-formatted results table.
        """
        if models is None:
            models = ['bp', 'dll', 'hierarchical', 'hybrid', 'predictive', 'bidirectional']

        model_display_names = {
            'bp': 'Backprop',
            'dll': 'Standard DLL',
            'hierarchical': 'Hierarchical DLL',
            'hybrid': 'Hybrid DLL',
            'temporal': 'Temporal DLL',
            'predictive': 'Predictive DLL',
            'bidirectional': 'Bidirectional DLL'
        }

        results = []

        for model in models:
            data = self.get_latest_run(model)
            if data is None:
                continue

            val_acc = data.get('val_acc', [])
            val_loss = data.get('val_loss', [])

            if val_acc and val_loss:
                final_acc = val_acc[-1] * 100
                final_loss = val_loss[-1]

                results.append({
                    'Model': model_display_names.get(model, model.upper()),
                    'Accuracy': final_acc,
                    'Loss': final_loss
                })

        # Create pandas dataframe
        df = pd.DataFrame(results)

        # Sort by accuracy (descending)
        df = df.sort_values('Accuracy', ascending=False)

        # Add rank
        df.insert(0, 'Rank', range(1, len(df) + 1))

        # Format for LaTeX
        latex_table = df.to_latex(
            index=False,
            float_format="%.2f",
            caption="Performance comparison of DLL variants on Penn Treebank POS tagging",
            label="tab:results",
            column_format="cccc"
        )

        # Save
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            f.write("=== LATEX TABLE ===\n\n")
            f.write(latex_table)
            f.write("\n\n=== MARKDOWN TABLE ===\n\n")
            f.write(df.to_markdown(index=False, floatfmt=".2f"))
            f.write("\n\n=== CSV ===\n\n")
            f.write(df.to_csv(index=False))

        print(f"Saved results table → {save_path}")
        print("\nResults Summary:")
        print(df.to_string(index=False))

    def plot_convergence_speed(self, models=None, save_path="figures/convergence_speed.png"):
        """
        Plot how quickly each model reaches 60% validation accuracy.
        """
        if models is None:
            models = ['bp', 'dll', 'hierarchical', 'hybrid', 'predictive', 'bidirectional']

        model_display_names = {
            'bp': 'Backprop',
            'dll': 'DLL',
            'hierarchical': 'Hierarchical DLL',
            'hybrid': 'Hybrid DLL',
            'predictive': 'Predictive DLL',
            'bidirectional': 'Bidirectional DLL'
        }

        threshold = 0.60  # 60% accuracy
        epochs_to_threshold = []
        labels = []

        for model in models:
            data = self.get_latest_run(model)
            if data is None:
                continue

            val_acc = data.get('val_acc', [])
            if not val_acc:
                continue

            # Find first epoch where accuracy exceeds threshold
            for epoch_idx, acc in enumerate(val_acc):
                if acc >= threshold:
                    epochs_to_threshold.append(epoch_idx + 1)
                    labels.append(model_display_names.get(model, model.upper()))
                    break
            else:
                # Never reached threshold
                epochs_to_threshold.append(len(val_acc))
                labels.append(model_display_names.get(model, model.upper()) + "*")

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(labels)))
        bars = ax.barh(labels, epochs_to_threshold, color=colors, alpha=0.8, edgecolor='black')

        for bar, epochs in zip(bars, epochs_to_threshold):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f' {int(epochs)} epochs',
                   ha='left', va='center', fontsize=9, fontweight='bold')

        ax.set_xlabel('Epochs to Reach 60% Validation Accuracy', fontsize=12)
        ax.set_title('Convergence Speed Comparison', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, axis='x', alpha=0.3)

        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved convergence speed plot → {save_path}")
        plt.close()

    def generate_all_figures(self):
        """Generate all publication-quality figures."""
        print("\n========== GENERATING FIGURES ==========\n")

        self.plot_training_curves()
        self.plot_final_comparison()
        self.plot_dll_variants_comparison()
        self.plot_convergence_speed()
        self.create_results_table()

        print("\n========== ALL FIGURES GENERATED ==========")
        print("Check the ./figures/ directory for outputs")


if __name__ == "__main__":
    viz = ResultsVisualizer(results_dir="./results")
    viz.generate_all_figures()
