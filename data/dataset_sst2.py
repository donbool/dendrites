import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


def build_vocab(train_sentences, tokenizer):
    def yield_tokens(data_iter):
        for sentence in data_iter:
            yield tokenizer(sentence)

    vocab = build_vocab_from_iterator(
        yield_tokens(train_sentences),
        specials=["<pad>", "<unk>"]
    )
    vocab.set_default_index(vocab["<unk>"])
    return vocab


def encode_sentence(sentence, tokenizer, vocab, max_len):
    tokens = tokenizer(sentence)
    ids = [vocab[token] for token in tokens][:max_len]

    # pad to max_len
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

    print("Downloading SST-2 from HuggingFace...")
    dataset = load_dataset("glue", "sst2")

    train_sentences = dataset["train"]["sentence"]
    train_labels = dataset["train"]["label"]
    val_sentences   = dataset["validation"]["sentence"]
    val_labels      = dataset["validation"]["label"]

    tokenizer = get_tokenizer("basic_english")

    vocab = build_vocab(train_sentences, tokenizer)

    train_data = [
        (encode_sentence(sent, tokenizer, vocab, max_len),
         torch.tensor(label))
        for sent, label in zip(train_sentences, train_labels)
    ]

    val_data = [
        (encode_sentence(sent, tokenizer, vocab, max_len),
         torch.tensor(label))
        for sent, label in zip(val_sentences, val_labels)
    ]

    # drop_last=True is CRITICAL for DLL (requires fixed batch size)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader, vocab
