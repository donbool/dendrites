# models/dll_rnn.py
import torch
import torch.nn as nn
from types import SimpleNamespace

# ---- basic activations (mirroring the original repo's utils) ----

def tanh(x):
    return torch.tanh(x)

def tanh_deriv(x):
    return 1 - torch.tanh(x) ** 2

def linear(x):
    return x

def linear_deriv(x):
    return torch.ones_like(x)


# ======================================================================
#  Core DLL RNN MODEL (copied from original, only stylistic tweaks)
#  This is the thing you are "studying" / comparing against BP.
# ======================================================================

class DLL_RNN_Model(object):
    """
    Faithful implementation of the Dendritic Local Learning RNN from the
    original GitHub, kept as-is so its learning dynamics match the paper.

    Shapes follow the original:
        - seq_len: T
        - hidden_size: H
        - batch_size: B
        - input_size: D_in
        - output_size: D_out

        inputs_seq: (T, input_size, B)
        hu, hx:     (T+1, hidden_size, B)
        y:          (T, output_size, B)
        target_seq: (T, output_size, B)
    """

    def __init__(self, args, device='cuda') -> None:
        self.args = args
        self.device = device
        self.seq_len = args.seq_len
        self.hidden_size = args.hidden_size
        self.batch_size = args.batch_size
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.fn = args.fn
        self.fn_deriv = args.fn_deriv
        self.weight_learning_rate = args.weight_learning_rate
        self.clamp_val = 50
        self.theta_update_discount = args.theta_update_discount
        self.noclamp = args.noclamp
        self.fix_theta_until = args.fix_theta_until
        self.noise = args.noise

        # weights
        self.Wh = torch.empty([self.hidden_size, self.hidden_size]).normal_(
            mean=0.0, std=0.05).to(self.device)
        self.Wx = torch.empty([self.hidden_size, self.input_size]).normal_(
            mean=0.0, std=0.05).to(self.device)
        self.Wy = torch.empty([self.output_size, self.hidden_size]).normal_(
            mean=0.0, std=0.05).to(self.device)
        self.h0 = torch.empty([self.hidden_size, self.batch_size]).normal_(
            mean=0.0, std=0.05).to(self.device)

        self.hu = torch.empty(
            [self.seq_len+1, self.hidden_size, self.batch_size]
        ).to(self.device)
        self.hx = torch.zeros_like(self.hu).to(self.device)
        self.y = torch.empty(
            [self.seq_len, self.output_size, self.batch_size]
        ).to(self.device)

        self.theta_h = torch.zeros_like(self.Wh).normal_(
            mean=0.0, std=0.05).to(self.device)
        self.theta_y = torch.zeros_like(self.Wy).normal_(
            mean=0.0, std=0.05).to(self.device)

    def forward(self, inputs_seq):
        """
        inputs_seq: (T, input_size, B)
        returns:
            y: (T, output_size, B)
        """
        with torch.no_grad():
            self.inputs = inputs_seq.clone()
            self.hu[0] = self.h0.clone()
            self.hx[0] = self.h0.clone()
            for i, inp in enumerate(inputs_seq):
                self.hu[i+1] = self.fn(self.Wh @ self.hu[i] + self.Wx @ inp)
                self.hx[i+1] = self.hu[i+1].clone()
                self.y[i] = linear(self.Wy @ self.hu[i+1])
            return self.y

    def update_weights(self, target_seq, epoch_num):
        """
        target_seq: (T, output_size, B)
        """
        with torch.no_grad():
            # the last dim of ehs is not used, because we don't need hx[0] - hu[0]
            ehs = torch.zeros_like(self.hu).to(self.device)
            eys = torch.zeros_like(self.y).to(self.device)

            # backward pass for dendritic errors
            for i, tar in reversed(list(enumerate(target_seq))):
                eys[i] = tar - self.y[i]
                deltah = self.theta_y.T @ (
                    eys[i] * linear_deriv(self.Wy @ self.hu[i+1])
                )
                if i < len(target_seq)-1:
                    fn_deriv = self.fn_deriv(
                        self.Wh @ self.hu[i+1] + self.Wx @ self.inputs[i]
                    )  # current layer
                    deltah += self.theta_h.T @ (ehs[i+1] * fn_deriv)
                ehs[i] = deltah

            if self.noise:
                ehs = ehs * (1 + 2 * self.noise * (torch.rand_like(ehs) - 0.5))

            dWy = torch.zeros_like(self.Wy).to(self.device)
            dWx = torch.zeros_like(self.Wx).to(self.device)
            dWh = torch.zeros_like(self.Wh).to(self.device)
            dtheta_y_T = torch.zeros_like(self.Wy.T).to(self.device)
            dtheta_h_T = torch.zeros_like(self.Wh.T).to(self.device)

            for i, inp in reversed(list(enumerate(self.inputs))):
                fn_deriv = self.fn_deriv(self.Wh @ self.hu[i] + self.Wx @ inp)
                dWy += (eys[i] * linear_deriv(self.Wy @ self.hu[i+1])) @ self.hu[i+1].T
                dWx += (ehs[i] * fn_deriv) @ inp.T
                dWh += (ehs[i] * fn_deriv) @ self.hu[i].T
                dtheta_y_T -= ehs[i] @ (
                    eys[i] * linear_deriv(self.Wy @ self.hu[i+1])
                ).T
                if i >= 1:
                    dtheta_h_T -= ehs[i-1] @ (ehs[i] * fn_deriv).T

            if not self.noclamp:
                dWy = torch.clamp(dWy, -self.clamp_val, self.clamp_val)
                dWx = torch.clamp(dWx, -self.clamp_val, self.clamp_val)
                dWh = torch.clamp(dWh, -self.clamp_val, self.clamp_val)
                dtheta_y_T = torch.clamp(
                    dtheta_y_T, -self.clamp_val, self.clamp_val)
                dtheta_h_T = torch.clamp(
                    dtheta_h_T, -self.clamp_val, self.clamp_val)

            self.Wy += self.weight_learning_rate * dWy
            self.Wx += self.weight_learning_rate * dWx
            self.Wh += self.weight_learning_rate * dWh

            if epoch_num >= self.fix_theta_until:
                self.theta_y += (
                    self.weight_learning_rate * dtheta_y_T.T /
                    self.theta_update_discount
                )
                self.theta_h += (
                    self.weight_learning_rate * dtheta_h_T.T /
                    self.theta_update_discount
                )


# ======================================================================
#  Thin PyTorch-friendly wrapper for classification (e.g., SST-2)
# ======================================================================

class DLLSentimentRNN(nn.Module):
    """
    Clean wrapper around DLL_RNN_Model for sequence classification.

    - Uses an embedding layer over token ids
    - Runs DLL_RNN_Model on the sequence of embeddings
    - Returns logits from the LAST time step (classification mode)
    - Provides a .dll_update(...) method that calls the exact DLL update rule

    Important constraints (to keep life simple & faithful):
        - Uses a FIXED seq_len (max sequence length after padding/truncation)
        - Uses a FIXED batch_size; your DataLoader should use `drop_last=True`
          so every batch is full-sized.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_classes: int,
        seq_len: int,
        batch_size: int,
        device: str = "cuda",
        weight_lr: float = 1e-3,
        theta_update_discount: float = 10.0,
        fix_theta_until: int = 1,
        noclamp: bool = False,
        noise: float = 0.0,
    ):
        super().__init__()
        self.device = device
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_classes = num_classes

        # Embedding for tokens
        self.embedding = nn.Embedding(vocab_size, embed_dim).to(device)

        # Build args for the core DLL_RNN_Model (matching original fields)
        args = SimpleNamespace()
        args.seq_len = seq_len
        args.hidden_size = hidden_size
        args.batch_size = batch_size
        args.input_size = embed_dim
        args.output_size = num_classes
        args.fn = tanh
        args.fn_deriv = tanh_deriv
        args.weight_learning_rate = weight_lr
        args.theta_update_discount = theta_update_discount
        args.fix_theta_until = fix_theta_until
        args.noclamp = noclamp
        args.noise = noise

        # Core DLL engine
        self.dll_core = DLL_RNN_Model(args, device=device)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: (B, T) with B == batch_size, T == seq_len

        Returns:
            logits: (B, num_classes)  (from the last time step)
        """
        B, T = token_ids.shape
        assert B == self.batch_size, (
            f"Batch size mismatch: expected {self.batch_size}, got {B}. "
            "Use DataLoader with drop_last=True and matching batch_size."
        )
        assert T == self.seq_len, (
            f"Seq len mismatch: expected {self.seq_len}, got {T}. "
            "Pad/truncate your inputs to a fixed length."
        )

        emb = self.embedding(token_ids.to(self.device))  # (B, T, D)
        # DLL_RNN_Model expects (T, input_size, B)
        inputs_seq = emb.permute(1, 2, 0)  # (T, D, B)

        y_seq = self.dll_core.forward(inputs_seq)  # (T, C, B)
        # use last time step for classification
        y_last = y_seq[-1]  # (C, B)
        logits = y_last.transpose(0, 1)  # (B, C)
        return logits

    def dll_update(self, labels: torch.Tensor, epoch: int):
        """
        Apply DLL weight + theta updates for a batch.

        labels: (B,) int64 class labels
        epoch:  int, current training epoch (for fix_theta_until)
        """
        B = labels.shape[0]
        assert B == self.batch_size, (
            f"Batch size mismatch in dll_update: expected {self.batch_size}, got {B}"
        )

        device = self.device
        T = self.dll_core.seq_len
        C = self.dll_core.output_size

        # one-hot targets: (B, C)
        targets_onehot = torch.zeros((B, C), device=device)
        targets_onehot[torch.arange(B, device=device), labels] = 1.0

        # For classification, we broadcast the label across all time steps:
        # target_seq: (T, C, B)
        target_seq = torch.zeros((T, C, B), device=device)
        # broadcasting: same label at each time step
        for t in range(T):
            target_seq[t] = targets_onehot.T  # (C, B)

        # Call the exact DLL update rule
        self.dll_core.update_weights(target_seq, epoch)
