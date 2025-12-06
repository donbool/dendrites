"""
Run Full Experimental Suite for Research Paper

Trains all models and generates comprehensive results:
1. Runs all DLL variants + backprop baseline
2. Collects metrics
3. Generates visualizations
4. Creates results tables
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_experiments(max_len=32, batch_size=16, epochs=5):
    """
    Run all experiments sequentially.

    Args:
        max_len: Maximum sequence length
        batch_size: Batch size (must match DLL requirements)
        epochs: Number of training epochs
    """

    models = [
        'bp',
        'dll',
        'hierarchical',
        'temporal',
        'predictive',
        'bidirectional'
    ]

    print("=" * 80)
    print(f"RUNNING FULL EXPERIMENTAL SUITE")
    print(f"Sequence length: {max_len}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print("=" * 80)

    for model in models:
        print(f"\n{'=' * 80}")
        print(f"TRAINING: {model.upper()}")
        print(f"{'=' * 80}\n")

        cmd = [
            sys.executable,
            "experiments/run_exp.py",
            f"--run_{model}",
            f"--epochs_{model}",
            str(epochs),
            "--batch_size",
            str(batch_size),
            "--max_len",
            str(max_len),
        ]

        # Add model-specific hyperparameters
        if model == 'predictive':
            cmd.extend(["--pred_weight", "0.05"])

        result = subprocess.run(cmd, capture_output=False, text=True)

        if result.returncode != 0:
            print(f"\n⚠️  WARNING: {model.upper()} training failed!")
        else:
            print(f"\n✓ {model.upper()} training completed successfully")

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 80)


def generate_visualizations():
    """Generate all figures and tables."""
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80 + "\n")

    cmd = [sys.executable, "experiments/visualize_results.py"]
    subprocess.run(cmd)


def run_comparisons():
    """
    Run pairwise comparisons between key models.
    """
    comparisons = [
        ("bp", "dll", "Backprop vs Standard DLL"),
        ("dll", "bidirectional", "Unidirectional vs Bidirectional DLL"),
        ("dll", "temporal", "Standard vs Temporal DLL"),
        ("dll", "predictive", "Standard vs Predictive DLL"),
    ]

    print("\n" + "=" * 80)
    print("KEY COMPARISONS")
    print("=" * 80 + "\n")

    for model1, model2, description in comparisons:
        print(f"\n--- {description} ---")

        cmd = [
            sys.executable,
            "experiments/run_exp.py",
            f"--run_{model1}",
            f"--run_{model2}",
            "--epochs_" + model1,
            "5",
            "--epochs_" + model2,
            "5",
            "--batch_size",
            "16",
            "--max_len",
            "32",
        ]

        if model2 == 'predictive':
            cmd.extend(["--pred_weight", "0.05"])

        subprocess.run(cmd, capture_output=False, text=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run full experimental suite for DLL RNN research paper"
    )

    parser.add_argument(
        "--max_len",
        type=int,
        default=32,
        help="Maximum sequence length"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size (fixed for DLL)"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip training, only generate visualizations"
    )

    parser.add_argument(
        "--only_comparisons",
        action="store_true",
        help="Only run key comparisons"
    )

    args = parser.parse_args()

    if args.only_comparisons:
        run_comparisons()
    elif args.skip_training:
        generate_visualizations()
    else:
        # Run full pipeline
        run_experiments(
            max_len=args.max_len,
            batch_size=args.batch_size,
            epochs=args.epochs
        )
        generate_visualizations()

    print("\n" + "=" * 80)
    print("✓ EXPERIMENTAL PIPELINE COMPLETE")
    print("=" * 80)
    print("\nResults saved to:")
    print("  - Metrics: ./results/*.json")
    print("  - Figures: ./figures/*.png")
    print("  - Tables:  ./figures/results_table.txt")
