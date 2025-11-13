from datasets import load_dataset
import torch
from torch.utils.data import DataLoader

def load_sst2(batch_size=32, max_len=32):
    dataset = load_dataset("glue", "sst2")

    train_texts = dataset["train"]["sentence"]
    train_labels = dataset["train"]["label"]

    val_texts = dataset["validation"]["sentence"]
    val_labels = dataset["validation"]["label"]

    # Simple tokenizer (you can swap in GPT2/BERT later)
    from torchtext.vocab import build_vocab_from_iterator
    from torchtext.data.utils import get_tokenizer
    tokenizer = get_tokenizer("basic_english")

    def yield_tokens(data_iter):
        for text in data_iter:
            yield tokenizer(text)

    vocab = build_vocab_from_iterator(yield_tokens(train_texts), specials=["<pad>", "<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    def encode(text):
        tokens = tokenizer(text)
        ids = [vocab[token] for token in tokens][:max_len]
        ids = ids + [vocab["<pad>"]] * (max_len - len(ids))
        return torch.tensor(ids)

    train_data = [(encode(t), torch.tensor(l)) for t, l in zip(train_texts, train_labels)]
    val_data   = [(encode(t), torch.tensor(l)) for t, l in zip(val_texts, val_labels)]

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_data, batch_size=batch_size)

    return train_loader, val_loader, vocab
