#!/usr/bin/env python3
"""
Diagnostic script to check why models are performing poorly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from data.dataset_sst2 import load_sst2
import numpy as np

print("=" * 70)
print("  SST-2 DATASET DIAGNOSTICS")
print("=" * 70)

# Load data
train_loader, val_loader, vocab = load_sst2(batch_size=32, max_len=32)

print(f"\n1. VOCAB SIZE: {len(vocab)}")
print(f"   (Typical: 10k-50k, yours: {len(vocab)})")

# Check sequence lengths
print(f"\n2. SEQUENCE LENGTH DISTRIBUTION:")
from data.dataset_sst2 import tokenize
from datasets import load_dataset
dataset = load_dataset("glue", "sst2", cache_dir="./data/cache")
train_sentences = dataset["train"]["sentence"]

lengths = [len(tokenize(s)) for s in train_sentences[:1000]]
print(f"   Min length: {min(lengths)}")
print(f"   Max length: {max(lengths)}")
print(f"   Mean length: {np.mean(lengths):.1f}")
print(f"   Median length: {np.median(lengths):.1f}")
print(f"   90th percentile: {np.percentile(lengths, 90):.1f}")
print(f"   95th percentile: {np.percentile(lengths, 95):.1f}")
print(f"\n   ⚠️  Your max_len=32 covers {np.mean(np.array(lengths) <= 32) * 100:.1f}% of data")

# Check batch
print(f"\n3. SAMPLE BATCH:")
token_ids, labels = next(iter(train_loader))
print(f"   Shape: token_ids={token_ids.shape}, labels={labels.shape}")
print(f"   Label distribution: {torch.bincount(labels)}")

# Check randomness
print(f"\n4. LABEL BALANCE:")
all_labels = torch.cat([labels for _, labels in val_loader])
print(f"   Positive (1): {(all_labels == 1).sum().item()} / {len(all_labels)}")
print(f"   Negative (0): {(all_labels == 0).sum().item()} / {len(all_labels)}")
pos_ratio = (all_labels == 1).sum().item() / len(all_labels)
print(f"   Positive ratio: {pos_ratio:.2%}")
print(f"   (Balanced = 50%, Random baseline accuracy = {max(pos_ratio, 1-pos_ratio):.2%})")

# Check tokenization quality
print(f"\n5. TOKENIZATION QUALITY:")
sample_sent = train_sentences[0]
tokens = tokenize(sample_sent)
print(f"   Original: {sample_sent[:80]}")
print(f"   Tokens:   {tokens[:15]}")
print(f"   (Are you losing info? Check if this looks reasonable)")

print(f"\n6. RECOMMENDATIONS:")
if np.mean(np.array(lengths) <= 32) < 0.9:
    print(f"   ⚠️  INCREASE max_len to 48 or 64 (you're truncating {(1 - np.mean(np.array(lengths) <= 32)) * 100:.1f}% of sequences)")
if len(vocab) < 1000:
    print(f"   ⚠️  vocab is very small ({len(vocab)}). Check if tokenizer is working")
if len(vocab) > 100000:
    print(f"   ⚠️  vocab is very large ({len(vocab)}). Consider limiting it")

print(f"\n7. WHAT TO TRY:")
print(f"   1. Increase max_len: --max_len 64 or 96")
print(f"   2. Check your tokenizer (might be too simple)")
print(f"   3. Try higher learning rates: --lr_bp 1e-2 or 5e-3")
print(f"   4. Train longer: --epochs_bp 20 --epochs_dll 20")
print(f"   5. Verify embedding dimension (300 is reasonable)")

print("\n" + "=" * 70)
