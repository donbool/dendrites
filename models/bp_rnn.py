# models/bp_rnn.py
import torch
import torch.nn as nn


class BPRNNSentiment(nn.Module):
    """
    A simple, clean RNN baseline using standard backpropagation.

    - Embedding layer
    - Single-layer tanh RNN or GRU
    - Classifier on final hidden state
    - Uses PyTorch's autograd (BP) unlike DLL

    Args:
        vocab_size      int
        embed_dim       int
        hidden_size     int
        num_classes     int
        seq_len         int  (must match DLL for fair comparison)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_classes: int,
        seq_len: int,
        device="cuda",
        rnn_type="rnn"   # "rnn" or "gru"
    ):
        super().__init__()

        self.device = device
        self.seq_len = seq_len
        self.hidden_size = hidden_size

        # ----- Embedding -----
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # ----- RNN (classic tanh) or GRU -----
        if rnn_type == "rnn":
            self.rnn = nn.RNN(
                input_size=embed_dim,
                hidden_size=hidden_size,
                batch_first=True,
                nonlinearity="tanh",
            )
        elif rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=embed_dim,
                hidden_size=hidden_size,
                batch_first=True,
            )
        else:
            raise ValueError("rnn_type must be 'rnn' or 'gru'")

        # ----- Classifier (last hidden → logits) -----
        self.classifier = nn.Linear(hidden_size, num_classes)

        self.to(device)

    def forward(self, token_ids):
        """
        token_ids: (B, T)
        Returns:
            logits: (B, T, num_classes) for sequence tagging
        """
        B, T = token_ids.shape
        assert T == self.seq_len, "Pad/truncate sequences to seq_len"

        emb = self.embedding(token_ids.to(self.device))  # (B, T, D)

        # RNN forward
        outputs, h_last = self.rnn(emb)  # outputs: (B, T, H)

        # Apply classifier to each timestep
        logits = self.classifier(outputs)  # (B, T, num_classes)
        return logits
