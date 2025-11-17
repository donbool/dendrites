# CLI pipeline: train BP, train DLL, train Hybrid, evaluate

import sys
import os
import torch
import argparse

# Add parent directory to path so we can import models, train, data
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.bp_rnn import BPRNN
from models.dll_rnn import DLLRNN
from models.hybrid_rnn import HybridDLLRNN
from train.train_bp import train_bp
from train.train_dll import train_dll
from train.train_hybrid import train_hybrid
from data.dataset_pos_tagging import load_pos_tagging
from experiments.metrics import MetricsLogger
from experiments.eval_models import evaluate_model, print_metrics, compare_models


def run_experiment(args):
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    print("Loading Penn Treebank POS tagging dataset...")
    train_loader, val_loader, word_vocab, pos_vocab = load_pos_tagging(
        batch_size=args.batch_size,
        max_len=args.max_len
    )

    # --------------------------
    # 1. BACKPROP BASELINE
    # --------------------------
    bp_metrics_final = None
    if args.run_bp:
        print("\n========== TRAINING BP BASELINE ==========\n")

        bp_model = BPRNN(
            vocab_size=len(word_vocab),
            embed_dim=args.embed_dim,
            hidden_size=args.hidden_size,
            num_classes=len(pos_vocab),
            seq_len=args.max_len,
            device=device
        )

        bp_logger = MetricsLogger(log_dir="./results", model_name="bp")

        train_bp(
            model=bp_model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs_bp,
            lr=args.lr_bp,
            device=device,
            metrics_logger=bp_logger,
        )

        bp_logger.save()
        torch.save(bp_model.state_dict(), "results/bp_model.pt")
        print("Saved BP model → results/bp_model.pt")

        # Evaluate on validation set
        bp_metrics_final = evaluate_model(bp_model, val_loader, model_type="bp", device=device)
        print_metrics(bp_metrics_final, split="validation")

    # --------------------------
    # 2. DENDRITIC LOCAL LEARNING (DLL)
    # --------------------------
    dll_metrics_final = None
    if args.run_dll:
        print("\n========== TRAINING DLL MODEL ==========\n")

        dll_model = DLLRNN(
            vocab_size=len(word_vocab),
            embed_dim=args.embed_dim,
            hidden_size=args.hidden_size,
            num_classes=len(pos_vocab),
            seq_len=args.max_len,
            batch_size=args.batch_size,
            device=device,
            weight_lr=args.lr_dll
        )

        dll_logger = MetricsLogger(log_dir="./results", model_name="dll")

        train_dll(
            model=dll_model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs_dll,
            device=device,
            metrics_logger=dll_logger,
        )

        dll_logger.save()
        torch.save(dll_model.dll_core.Wy, "results/dll_Wy.pt")
        torch.save(dll_model.dll_core.Wh, "results/dll_Wh.pt")
        print("Saved DLL weights → results/dll_Wy.pt, results/dll_Wh.pt")

        # Evaluate on validation set
        dll_metrics_final = evaluate_model(dll_model, val_loader, model_type="dll", device=device)
        print_metrics(dll_metrics_final, split="validation")

    # --------------------------
    # 3. HYBRID DLL (DLL + Temporal Credit via Traces)
    # --------------------------
    hybrid_metrics_final = None
    if args.run_hybrid:
        print("\n========== TRAINING HYBRID DLL MODEL ==========\n")

        hybrid_model = HybridDLLRNN(
            vocab_size=len(word_vocab),
            embed_dim=args.embed_dim,
            hidden_size=args.hidden_size,
            num_classes=len(pos_vocab),
            seq_len=args.max_len,
            batch_size=args.batch_size,
            device=device,
            weight_lr=args.lr_hybrid,
            use_traces=args.use_traces,
            e_decay=args.e_decay,
            e_clip=args.e_clip,
        )

        hybrid_logger = MetricsLogger(log_dir="./results", model_name="hybrid")

        train_hybrid(
            model=hybrid_model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs_hybrid,
            device=device,
            metrics_logger=hybrid_logger,
        )

        hybrid_logger.save()
        torch.save(hybrid_model.dll_core.Wy, "results/hybrid_Wy.pt")
        torch.save(hybrid_model.dll_core.Wh, "results/hybrid_Wh.pt")
        print("Saved Hybrid weights → results/hybrid_Wy.pt, results/hybrid_Wh.pt")

        # Evaluate on validation set
        hybrid_metrics_final = evaluate_model(hybrid_model, val_loader, model_type="dll", device=device)
        print_metrics(hybrid_metrics_final, split="validation")

    # --------------------------
    # 4. COMPARISON
    # --------------------------
    if bp_metrics_final is not None and dll_metrics_final is not None:
        compare_models(bp_metrics_final, dll_metrics_final)

    if bp_metrics_final is not None and hybrid_metrics_final is not None:
        print("\n========== BP vs HYBRID COMPARISON ==========")
        compare_models(bp_metrics_final, hybrid_metrics_final)

    if dll_metrics_final is not None and hybrid_metrics_final is not None:
        print("\n========== DLL vs HYBRID COMPARISON ==========")
        compare_models(dll_metrics_final, hybrid_metrics_final)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--embed_dim", type=int, default=150)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=32)

    parser.add_argument("--epochs_bp", type=int, default=5)
    parser.add_argument("--epochs_dll", type=int, default=5)
    parser.add_argument("--epochs_hybrid", type=int, default=5)

    parser.add_argument("--lr_bp", type=float, default=1e-2)
    parser.add_argument("--lr_dll", type=float, default=5e-4)
    parser.add_argument("--lr_hybrid", type=float, default=5e-4)

    # Hybrid-specific hyperparameters
    parser.add_argument("--use_traces", type=bool, default=True, help="Enable eligibility traces in hybrid model")
    parser.add_argument("--e_decay", type=float, default=0.92, help="Eligibility trace decay (lambda)")
    parser.add_argument("--e_clip", type=float, default=0.1, help="Eligibility trace clipping value")

    parser.add_argument("--run_bp", action="store_true")
    parser.add_argument("--run_dll", action="store_true")
    parser.add_argument("--run_hybrid", action="store_true")

    args = parser.parse_args()

    run_experiment(args)
