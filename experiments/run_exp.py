# CLI pipeline: train BP, train DLL, train Hybrid, train Hierarchical, train Temporal, evaluate

import sys
import os
import torch
import argparse

# Add parent directory to path so we can import models, train, data
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.bp_rnn import BPRNN
from models.dll_rnn import DLLRNN
from models.hybrid_rnn import HybridDLLRNN
from models.hierarchical_rnn import HierarchicalDLLRNN
from models.temporal_rnn import TemporalDLLRNN
from models.predictive_coding_rnn import PredictiveCoderRNN
from train.train_bp import train_bp
from train.train_dll import train_dll
from train.train_hybrid import train_hybrid
from train.train_hierarchical import train_hierarchical
from train.train_temporal import train_temporal
from train.train_predictive import train_predictive
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
            trace_strength=args.trace_strength,
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
    # 4. HIERARCHICAL DLL (Multi-level Theta for Temporal Credit)
    # --------------------------
    hierarchical_metrics_final = None
    if args.run_hierarchical:
        print("\n========== TRAINING HIERARCHICAL DLL MODEL ==========\n")

        hierarchical_model = HierarchicalDLLRNN(
            vocab_size=len(word_vocab),
            embed_dim=args.embed_dim,
            hidden_size=args.hidden_size,
            num_classes=len(pos_vocab),
            seq_len=args.max_len,
            batch_size=args.batch_size,
            device=device,
            weight_lr=args.lr_hierarchical,
        )

        hierarchical_logger = MetricsLogger(log_dir="./results", model_name="hierarchical")

        train_hierarchical(
            model=hierarchical_model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs_hierarchical,
            device=device,
            metrics_logger=hierarchical_logger,
        )

        hierarchical_logger.save()
        torch.save(hierarchical_model.dll_core.Wh, "results/hierarchical_Wh.pt")
        torch.save(hierarchical_model.dll_core.theta_h_short, "results/hierarchical_theta_short.pt")
        torch.save(hierarchical_model.dll_core.theta_h_medium, "results/hierarchical_theta_medium.pt")
        torch.save(hierarchical_model.dll_core.theta_h_long, "results/hierarchical_theta_long.pt")
        print("Saved Hierarchical weights → results/hierarchical_*.pt")

        # Evaluate on validation set
        hierarchical_metrics_final = evaluate_model(hierarchical_model, val_loader, model_type="dll", device=device)
        print_metrics(hierarchical_metrics_final, split="validation")

    # --------------------------
    # 5. TEMPORAL DLL (Recurrent Error Paths for Temporal Credit)
    # --------------------------
    temporal_metrics_final = None
    if args.run_temporal:
        print("\n========== TRAINING TEMPORAL DLL MODEL ==========\n")

        temporal_model = TemporalDLLRNN(
            vocab_size=len(word_vocab),
            embed_dim=args.embed_dim,
            hidden_size=args.hidden_size,
            num_classes=len(pos_vocab),
            seq_len=args.max_len,
            batch_size=args.batch_size,
            device=device,
            weight_lr=args.lr_temporal,
            temporal_decay=args.temporal_decay,
        )

        temporal_logger = MetricsLogger(log_dir="./results", model_name="temporal")

        train_temporal(
            model=temporal_model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs_temporal,
            device=device,
            metrics_logger=temporal_logger,
        )

        temporal_logger.save()
        torch.save(temporal_model.dll_core.Wh, "results/temporal_Wh.pt")
        torch.save(temporal_model.dll_core.theta_h, "results/temporal_theta_h.pt")
        torch.save(temporal_model.dll_core.theta_h_recurrent, "results/temporal_theta_h_recurrent.pt")
        print("Saved Temporal weights → results/temporal_*.pt")

        # Evaluate on validation set
        temporal_metrics_final = evaluate_model(temporal_model, val_loader, model_type="dll", device=device)
        print_metrics(temporal_metrics_final, split="validation")

    # --------------------------
    # 6. PREDICTIVE CODING DLL (Dual Objectives for Temporal Credit)
    # --------------------------
    predictive_metrics_final = None
    if args.run_predictive:
        print("\n========== TRAINING PREDICTIVE CODING DLL MODEL ==========\n")

        predictive_model = PredictiveCoderRNN(
            vocab_size=len(word_vocab),
            embed_dim=args.embed_dim,
            hidden_size=args.hidden_size,
            num_classes=len(pos_vocab),
            seq_len=args.max_len,
            batch_size=args.batch_size,
            device=device,
            weight_lr=args.lr_predictive,
            pred_weight=args.pred_weight,
        )

        predictive_logger = MetricsLogger(log_dir="./results", model_name="predictive")

        train_predictive(
            model=predictive_model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs_predictive,
            device=device,
            metrics_logger=predictive_logger,
        )

        predictive_logger.save()
        torch.save(predictive_model.dll_core.Wh, "results/predictive_Wh.pt")
        torch.save(predictive_model.dll_core.W_pred, "results/predictive_W_pred.pt")
        torch.save(predictive_model.dll_core.theta_h, "results/predictive_theta_h.pt")
        torch.save(predictive_model.dll_core.theta_h_pred, "results/predictive_theta_h_pred.pt")
        print("Saved Predictive Coding weights → results/predictive_*.pt")

        # Evaluate on validation set
        predictive_metrics_final = evaluate_model(predictive_model, val_loader, model_type="dll", device=device)
        print_metrics(predictive_metrics_final, split="validation")

    # --------------------------
    # 7. COMPARISON
    # --------------------------
    if bp_metrics_final is not None and dll_metrics_final is not None:
        compare_models(bp_metrics_final, dll_metrics_final)

    if bp_metrics_final is not None and hybrid_metrics_final is not None:
        print("\n========== BP vs HYBRID COMPARISON ==========")
        compare_models(bp_metrics_final, hybrid_metrics_final)

    if dll_metrics_final is not None and hybrid_metrics_final is not None:
        print("\n========== DLL vs HYBRID COMPARISON ==========")
        compare_models(dll_metrics_final, hybrid_metrics_final)

    if dll_metrics_final is not None and hierarchical_metrics_final is not None:
        print("\n========== DLL vs HIERARCHICAL COMPARISON ==========")
        compare_models(dll_metrics_final, hierarchical_metrics_final)

    if bp_metrics_final is not None and hierarchical_metrics_final is not None:
        print("\n========== BP vs HIERARCHICAL COMPARISON ==========")
        compare_models(bp_metrics_final, hierarchical_metrics_final)

    if dll_metrics_final is not None and temporal_metrics_final is not None:
        print("\n========== DLL vs TEMPORAL COMPARISON ==========")
        compare_models(dll_metrics_final, temporal_metrics_final)

    if bp_metrics_final is not None and temporal_metrics_final is not None:
        print("\n========== BP vs TEMPORAL COMPARISON ==========")
        compare_models(bp_metrics_final, temporal_metrics_final)

    if dll_metrics_final is not None and predictive_metrics_final is not None:
        print("\n========== DLL vs PREDICTIVE CODING COMPARISON ==========")
        compare_models(dll_metrics_final, predictive_metrics_final)

    if bp_metrics_final is not None and predictive_metrics_final is not None:
        print("\n========== BP vs PREDICTIVE CODING COMPARISON ==========")
        compare_models(bp_metrics_final, predictive_metrics_final)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--embed_dim", type=int, default=150)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=32)

    parser.add_argument("--epochs_bp", type=int, default=5)
    parser.add_argument("--epochs_dll", type=int, default=5)
    parser.add_argument("--epochs_hybrid", type=int, default=5)
    parser.add_argument("--epochs_hierarchical", type=int, default=5)
    parser.add_argument("--epochs_temporal", type=int, default=5)
    parser.add_argument("--epochs_predictive", type=int, default=5)

    parser.add_argument("--lr_bp", type=float, default=1e-2)
    parser.add_argument("--lr_dll", type=float, default=5e-4)
    parser.add_argument("--lr_hybrid", type=float, default=5e-4)
    parser.add_argument("--lr_hierarchical", type=float, default=5e-4)
    parser.add_argument("--lr_temporal", type=float, default=5e-4)
    parser.add_argument("--lr_predictive", type=float, default=5e-4)

    # Hybrid-specific hyperparameters
    parser.add_argument("--use_traces", type=bool, default=False, help="Enable eligibility traces in hybrid model")
    parser.add_argument("--e_decay", type=float, default=0.92, help="Eligibility trace decay (lambda)")
    parser.add_argument("--e_clip", type=float, default=1.0, help="Eligibility trace clipping value")
    parser.add_argument("--trace_strength", type=float, default=0.0, help="Scale factor for eligibility trace contribution to Wh update")

    # Temporal-specific hyperparameters
    parser.add_argument("--temporal_decay", type=float, default=0.9, help="Discount factor for recurrent error paths (0.0-1.0)")

    # Predictive Coding-specific hyperparameters
    parser.add_argument("--pred_weight", type=float, default=0.5, help="Balance between task and prediction objectives (0.0-1.0)")

    parser.add_argument("--run_bp", action="store_true")
    parser.add_argument("--run_dll", action="store_true")
    parser.add_argument("--run_hybrid", action="store_true")
    parser.add_argument("--run_hierarchical", action="store_true")
    parser.add_argument("--run_temporal", action="store_true")
    parser.add_argument("--run_predictive", action="store_true")

    args = parser.parse_args()

    run_experiment(args)
