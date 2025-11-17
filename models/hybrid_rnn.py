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
#  Core DLL RNN MODEL + Hybrid Temporal Credit via Eligibility Traces
# ======================================================================

class DLL_RNN_Model(object):
    """
    Dendritic Local Learning RNN from the original GitHub, with a small,
    biologically-inspired extension:

        - Original DLL provides *spatial* credit assignment (local losses)
        - We add *temporal* credit assignment via eligibility traces on Wh

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

        # Optional hybrid flags / hyperparams (with defaults)
        self.use_traces = getattr(args, "use_traces", True)
        self.e_decay = getattr(args, "e_decay", 0.92)  # λ
        self.e_clip = getattr(args, "e_clip", 0.05)  # Conservative clipping for stability
        self.trace_strength = getattr(args, "trace_strength", 0.1)  # Scale factor for trace contribution

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

        # ------------------------------------------------------------------
        # Eligibility traces for the recurrent weights Wh across time.
        # e_Wh_hist[t] is the eligibility trace matrix at time t.
        # t ranges from 0..T (T+1 entries); at t=0 it's all zeros.
        # ------------------------------------------------------------------
        if self.use_traces:
            self.e_Wh_hist = torch.zeros(
                self.seq_len + 1, self.hidden_size, self.hidden_size,
                device=self.device
            )

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

            # reset eligibility history for this sequence
            if self.use_traces:
                self.e_Wh_hist.zero_()  # e_Wh_hist[0] = 0 by construction

            for i, inp in enumerate(inputs_seq):
                # Standard DLL RNN forward
                self.hu[i+1] = self.fn(self.Wh @ self.hu[i] + self.Wx @ inp)
                self.hx[i+1] = self.hu[i+1].clone()
                self.y[i] = linear(self.Wy @ self.hu[i+1])

                # ---------------------------------------------------------
                # Hybrid extension: update eligibility trace for Wh at t=i+1
                # e_Wh_hist[t] = λ e_Wh_hist[t-1] + Hebbian(pre, post)
                #   pre  ~ hu[i]    (previous hidden state)
                #   post ~ hu[i+1]  (current hidden state)
                # We average over batch and use an outer product.
                # ---------------------------------------------------------
                if self.use_traces:
                    pre = self.hu[i]       # (H, B)
                    post = self.hu[i+1]    # (H, B)

                    pre_b = pre.mean(dim=1, keepdim=True)   # (H, 1)
                    post_b = post.mean(dim=1, keepdim=True) # (H, 1)

                    hebbian_Wh = post_b @ pre_b.T           # (H, H)

                    # Decayed trace from previous time step
                    prev_trace = self.e_Wh_hist[i]          # (H, H)
                    new_trace = self.e_decay * prev_trace + hebbian_Wh

                    # Clip for stability
                    new_trace = torch.clamp(new_trace, -self.e_clip, self.e_clip)

                    self.e_Wh_hist[i+1] = new_trace

            return self.y

    def update_weights(self, target_seq, epoch_num, mask_seq=None):
        """
        target_seq: (T, output_size, B)
        mask_seq: (T, B) optional mask (1 for real tokens, 0 for padding)
        """
        with torch.no_grad():
            # the last dim of ehs is not used, because we don't need hx[0] - hu[0]
            ehs = torch.zeros_like(self.hu).to(self.device)
            eys = torch.zeros_like(self.y).to(self.device)

            # backward pass for dendritic errors (original DLL logic)
            for i, tar in reversed(list(enumerate(target_seq))):
                eys[i] = tar - self.y[i]

                # Apply mask: zero out errors for padding positions
                if mask_seq is not None:
                    eys[i] = eys[i] * mask_seq[i].unsqueeze(0)  # (C, B) * (1, B)

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

            # Extra accumulator for temporal credit via traces
            if self.use_traces:
                temporal_grad_Wh = torch.zeros_like(self.Wh).to(self.device)

            # Main gradient accumulation loop (original + hybrid bits)
            for i, inp in reversed(list(enumerate(self.inputs))):
                fn_deriv = self.fn_deriv(self.Wh @ self.hu[i] + self.Wx @ inp)

                # Original DLL weight updates
                dWy += (eys[i] * linear_deriv(self.Wy @ self.hu[i+1])) @ self.hu[i+1].T
                dWx += (ehs[i] * fn_deriv) @ inp.T
                dWh += (ehs[i] * fn_deriv) @ self.hu[i].T

                dtheta_y_T -= ehs[i] @ (
                    eys[i] * linear_deriv(self.Wy @ self.hu[i+1])
                ).T
                if i >= 1:
                    dtheta_h_T -= ehs[i-1] @ (ehs[i] * fn_deriv).T

                # ---------------------------------------------------------
                # Hybrid temporal credit assignment:
                #   temporal_grad_Wh += mod_i * e_Wh_hist[i+1]
                # where mod_i is a scalar "error magnitude" at timestep i.
                # ---------------------------------------------------------
                if self.use_traces:
                    # eys[i]: (C, B). Use its mean absolute value as modulatory signal
                    mod_i = torch.mean(torch.abs(eys[i]))  # scalar
                    # Scale trace contribution to avoid dominating DLL signal
                    temporal_grad_Wh += self.trace_strength * mod_i * self.e_Wh_hist[i+1]

            if not self.noclamp:
                dWy = torch.clamp(dWy, -self.clamp_val, self.clamp_val)
                dWx = torch.clamp(dWx, -self.clamp_val, self.clamp_val)
                dWh = torch.clamp(dWh, -self.clamp_val, self.clamp_val)
                dtheta_y_T = torch.clamp(
                    dtheta_y_T, -self.clamp_val, self.clamp_val)
                dtheta_h_T = torch.clamp(
                    dtheta_h_T, -self.clamp_val, self.clamp_val)

                # Combine DLL gradient with temporal credit gradient (AFTER clamping)
                if self.use_traces:
                    temporal_grad_Wh = torch.clamp(temporal_grad_Wh, -self.clamp_val, self.clamp_val)
                    dWh += temporal_grad_Wh
            else:
                # Combine DLL gradient with temporal credit gradient (when no clamping)
                if self.use_traces:
                    dWh += temporal_grad_Wh

            # Apply weight updates
            self.Wy += self.weight_learning_rate * dWy
            self.Wx += self.weight_learning_rate * dWx
            self.Wh += self.weight_learning_rate * dWh

            # Theta updates (unchanged from original)
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
#  Thin PyTorch-friendly wrapper for sequence tagging (e.g., POS)
# ======================================================================

class HybridDLLRNN(nn.Module):
    """
    Clean wrapper around DLL_RNN_Model with eligibility traces for sequence tagging (e.g., POS tagging).

    - Uses an embedding layer over token ids
    - Runs DLL_RNN_Model on the sequence of embeddings with temporal credit via traces
    - Returns logits for ALL time steps (sequence tagging mode)
    - Provides a .dll_update(...) method that calls the DLL update rule with trace-based temporal credit

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
        use_traces: bool = True,
        e_decay: float = 0.92,
        e_clip: float = 0.1,
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

        # Hybrid-specific flags / hyperparams
        args.use_traces = use_traces
        args.e_decay = e_decay
        args.e_clip = e_clip

        # Core DLL engine (now hybrid-capable)
        self.dll_core = DLL_RNN_Model(args, device=device)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: (B, T) with B == batch_size, T == seq_len

        Returns:
            logits: (B, T, num_classes) for sequence tagging (per-token)
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
        # return all timesteps for sequence tagging
        logits = y_seq.permute(2, 0, 1)  # (B, T, C)
        return logits

    def dll_update(self, labels: torch.Tensor, mask: torch.Tensor, epoch: int):
        """
        Apply DLL weight + theta updates for a batch (sequence tagging).

        labels: (B, T) int64 POS tags (per-token labels)
        mask:   (B, T) float32 mask (1 for real tokens, 0 for padding)
        epoch:  int, current training epoch (for fix_theta_until)
        """
        B, T = labels.shape
        assert B == self.batch_size, (
            f"Batch size mismatch in dll_update: expected {self.batch_size}, got {B}"
        )
        assert T == self.dll_core.seq_len, (
            f"Sequence length mismatch in dll_update: expected {self.dll_core.seq_len}, got {T}"
        )

        device = self.device
        C = self.dll_core.output_size

        # Create one-hot encoded targets for each timestep
        # target_seq: (T, C, B)
        target_seq = torch.zeros((T, C, B), device=device)
        mask_seq = torch.zeros((T, B), device=device)  # (T, B) mask for masking errors

        for t in range(T):
            # Get labels and mask at timestep t: (B,)
            labels_t = labels[:, t].long()
            mask_t = mask[:, t]  # (B,)

            # Create one-hot: (B, C)
            targets_onehot_t = torch.zeros((B, C), device=device)

            # Only set one-hot for non-padding positions
            valid_mask = (mask_t > 0.5)  # boolean mask for real tokens
            if valid_mask.any():
                valid_indices = torch.arange(B, device=device)[valid_mask]
                valid_labels = labels_t[valid_mask]
                targets_onehot_t[valid_indices, valid_labels] = 1.0

            # Store transposed: (C, B)
            target_seq[t] = targets_onehot_t.T
            mask_seq[t] = mask_t  # Store mask for this timestep

        # Call the DLL update rule (now with optional temporal credit)
        self.dll_core.update_weights(target_seq, epoch, mask_seq=mask_seq)
