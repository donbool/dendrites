import torch
from torch.utils.data import DataLoader, TensorDataset
import nltk
from collections import defaultdict

# Download Penn Treebank if not already downloaded
try:
    nltk.data.find('corpora/treebank')
except LookupError:
    print("Downloading Penn Treebank corpus...")
    nltk.download('treebank')


def load_pos_tagging(batch_size=32, max_len=50):
    """
    Load Penn Treebank POS tagging dataset.

    Returns:
        - train_loader: DataLoader with (token_ids, pos_ids, mask)
        - val_loader: DataLoader with (token_ids, pos_ids, mask)
        - word_vocab: dict mapping word -> id
        - pos_vocab: dict mapping POS tag -> id
    """
    from nltk.corpus import treebank

    print("Loading Penn Treebank POS tagging data...")

    # Load all sentences with POS tags
    tagged_sents = treebank.tagged_sents()
    print(f"✓ Loaded {len(tagged_sents)} sentences")

    # Split into train/val (80/20)
    split_idx = int(0.8 * len(tagged_sents))
    train_sents = tagged_sents[:split_idx]
    val_sents = tagged_sents[split_idx:]

    print(f"  Train: {len(train_sents)} sentences")
    print(f"  Val:   {len(val_sents)} sentences")

    # Build vocabularies
    word_vocab = {"<pad>": 0, "<unk>": 1}
    pos_vocab = {"<pad>": 0}

    # Count word and POS frequencies
    word_freq = defaultdict(int)
    pos_freq = defaultdict(int)

    for sent in train_sents:
        for word, pos in sent:
            word = word.lower()
            word_freq[word] += 1
            pos_freq[pos] += 1

    # Build word vocab (include all words from train)
    for word in sorted(word_freq.keys()):
        if word not in word_vocab:
            word_vocab[word] = len(word_vocab)

    # Build POS vocab (include all tags)
    for pos in sorted(pos_freq.keys()):
        if pos not in pos_vocab:
            pos_vocab[pos] = len(pos_vocab)

    print(f"  Vocab size: {len(word_vocab)} words, {len(pos_vocab)} POS tags")

    # Encode sentences
    def encode_sentence(sent, word_vocab, pos_vocab, max_len):
        """Encode a sentence to token IDs and POS IDs with padding."""
        token_ids = []
        pos_ids = []

        for word, pos in sent[:max_len]:
            word = word.lower()
            token_id = word_vocab.get(word, word_vocab["<unk>"])
            pos_id = pos_vocab.get(pos, pos_vocab["<pad>"])
            token_ids.append(token_id)
            pos_ids.append(pos_id)

        # Pad
        pad_len = max_len - len(token_ids)
        token_ids += [word_vocab["<pad>"]] * pad_len
        pos_ids += [pos_vocab["<pad>"]] * pad_len

        # Create mask (1 for real tokens, 0 for padding)
        mask = [1] * (max_len - pad_len) + [0] * pad_len

        return torch.tensor(token_ids), torch.tensor(pos_ids), torch.tensor(mask, dtype=torch.float32)

    # Encode train and val sets
    train_data = []
    for sent in train_sents:
        token_ids, pos_ids, mask = encode_sentence(sent, word_vocab, pos_vocab, max_len)
        train_data.append((token_ids, pos_ids, mask))

    val_data = []
    for sent in val_sents:
        token_ids, pos_ids, mask = encode_sentence(sent, word_vocab, pos_vocab, max_len)
        val_data.append((token_ids, pos_ids, mask))

    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=True)

    print(f"✓ Created DataLoaders (batch_size={batch_size}, max_len={max_len})")

    return train_loader, val_loader, word_vocab, pos_vocab
