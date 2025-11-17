import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import re
from collections import defaultdict


def tokenize(sentence):
    """Simple regex-based tokenizer."""
    return re.findall(r'\w+', sentence.lower())


def build_vocab(train_sentences):
    """Build vocabulary from training sentences."""
    word_freq = defaultdict(int)

    for sentence in train_sentences:
        tokens = tokenize(sentence)
        for token in tokens:
            word_freq[token] += 1

    # Create vocab with special tokens
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, _ in sorted(word_freq.items(), key=lambda x: x[1], reverse=True):
        if word not in vocab:
            vocab[word] = len(vocab)

    return vocab


def encode_sentence(sentence, vocab, max_len):
    """Encode sentence to token IDs with padding."""
    tokens = tokenize(sentence)
    ids = [vocab.get(token, vocab["<unk>"]) for token in tokens][:max_len]

    # Pad to max_len
    if len(ids) < max_len:
        ids += [vocab["<pad>"]] * (max_len - len(ids))

    return torch.tensor(ids)


def load_sst2(batch_size=32, max_len=32):
    """
    Returns:
        - train_loader
        - val_loader
        - vocab
    """

    import os
    cache_dir = "./data/cache"

    # Check if dataset is already cached
    if os.path.exists(cache_dir) and len(os.listdir(cache_dir)) > 0:
        print(f"✓ Found cached dataset in {cache_dir}")
    else:
        print(f"✗ Dataset not cached. Downloading from HuggingFace...")

    dataset = load_dataset("glue", "sst2", cache_dir=cache_dir)

    train_sentences = dataset["train"]["sentence"]
    train_labels = dataset["train"]["label"]
    val_sentences   = dataset["validation"]["sentence"]
    val_labels      = dataset["validation"]["label"]

    vocab = build_vocab(train_sentences)

    train_data = [
        (encode_sentence(sent, vocab, max_len),
         torch.tensor(label))
        for sent, label in zip(train_sentences, train_labels)
    ]

    val_data = [
        (encode_sentence(sent, vocab, max_len),
         torch.tensor(label))
        for sent, label in zip(val_sentences, val_labels)
    ]

    # drop_last=True is CRITICAL for DLL (requires fixed batch size)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader, vocab
