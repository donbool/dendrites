# run_experiment.py

import torch
import argparse

from models.bp_rnn import BPSentimentRNN
from models.dll_rnn import DLLSentimentRNN
from train.train_bp import train_bp
from train.train_dll import train_dll
from data.dataset_sst2 import load_sst2


def run_experiment(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading SST-2 dataset...")
    train_loader, val_loader, vocab = load_sst2(
        batch_size=args.batch_size,
        max_len=args.max_len
    )

    # --------------------------
    # 1. BACKPROP BASELINE
    # --------------------------
    if args.run_bp:
        print("\n========== TRAINING BP BASELINE ==========\n")
        
        bp_model = BPSentimentRNN(
            vocab_size=len(vocab),
            embed_dim=args.embed_dim,
            hidden_size=args.hidden_size,
            num_classes=2,
            device=device
        )

        train_bp(
            model=bp_model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs_bp,
            lr=args.lr_bp,
            device=device
        )

        torch.save(bp_model.state_dict(), "bp_model.pt")
        print("Saved BP model → bp_model.pt")

    # --------------------------
    # 2. DENDRITIC LOCAL LEARNING (DLL)
    # --------------------------
    if args.run_dll:
        print("\n========== TRAINING DLL MODEL ==========\n")

        dll_model = DLLSentimentRNN(
            vocab_size=len(vocab),
            embed_dim=args.embed_dim,
            hidden_size=args.hidden_size,
            output_size=2,
            device=device
        )

        train_dll(
            model=dll_model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs_dll,
            device=device
        )

        torch.save(dll_model.dll_core.Wy, "dll_Wy.pt")
        torch.save(dll_model.dll_core.Wh, "dll_Wh.pt")
        print("Saved DLL weights → dll_Wy.pt, dll_Wh.pt")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--embed_dim", type=int, default=300)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=32)

    parser.add_argument("--epochs_bp", type=int, default=5)
    parser.add_argument("--epochs_dll", type=int, default=5)

    parser.add_argument("--lr_bp", type=float, default=1e-3)

    parser.add_argument("--run_bp", action="store_true")
    parser.add_argument("--run_dll", action="store_true")

    args = parser.parse_args()

    run_experiment(args)
