"""
Temporal DLL RNN with Recurrent Error Integration

Core insight: Standard DLL lacks temporal credit assignment because a single theta_h
cannot approximate error backprop across time. BPTT requires a global error signal,
which is biologically implausible.

Solution: Introduce RECURRENT ERROR PATHS via theta_h_recurrent (lateral dendrites).
Instead of errors flowing backward globally, they flow locally within the hidden layer
through slow recurrent connections. Each neuron receives:
  1. Output errors (via theta_y) - immediate credit from the task
  2. Recurrent errors from previous timestep (via theta_h_recurrent) - temporal credit

This is biologically plausible because:
- No global backprop signal needed
- All computations are local to each neuron
- Lateral dendrites can integrate signals over time
- Asymmetric forward/backward weights (theta weights as backward dendrites)

The recurrent error path acts like a "temporal eligibility trace" but computed
locally: past errors persist through recurrent connections and modulate weight updates
at the current timestep, allowing credit to flow backward through time without
requiring a global error signal.
"""

import torch
import torch.nn as nn
from types import SimpleNamespace


def tanh(x):
    return torch.tanh(x)

def tanh_deriv(x):
    return 1 - torch.tanh(x) ** 2

def linear(x):
    return x

def linear_deriv(x):
    return torch.ones_like(x)


class TemporalDLL_RNN_Model(object):
    """
    DLL RNN with recurrent error integration for temporal credit assignment.

    Key addition over standard DLL:
    - theta_h_recurrent: Asymmetric weights that carry errors backward through time
      These act like LATERAL DENDRITES that integrate signals from the previous
      timestep's hidden layer error

    The error computation becomes:
      deltah[t] = theta_y.T @ output_error[t]
                + theta_h.T @ hidden_error[t+1]  (standard DLL)
                + theta_h_recurrent.T @ hidden_error[t-1]  (NEW: temporal path)

    This allows errors to flow backward through time locally, without global BPTT.
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
        self.temporal_decay = getattr(args, 'temporal_decay', 0.9)  # Discount older errors

        # Standard forward weights
        self.Wh = torch.empty([self.hidden_size, self.hidden_size]).normal_(
            mean=0.0, std=0.05).to(self.device)
        self.Wx = torch.empty([self.hidden_size, self.input_size]).normal_(
            mean=0.0, std=0.05).to(self.device)
        self.Wy = torch.empty([self.output_size, self.hidden_size]).normal_(
            mean=0.0, std=0.05).to(self.device)
        self.h0 = torch.empty([self.hidden_size, self.batch_size]).normal_(
            mean=0.0, std=0.05).to(self.device)

        # Activation history
        self.hu = torch.empty(
            [self.seq_len+1, self.hidden_size, self.batch_size]
        ).to(self.device)
        self.hx = torch.zeros_like(self.hu).to(self.device)
        self.y = torch.empty(
            [self.seq_len, self.output_size, self.batch_size]
        ).to(self.device)

        # Standard DLL theta weights
        self.theta_h = torch.zeros_like(self.Wh).normal_(
            mean=0.0, std=0.05).to(self.device)
        self.theta_y = torch.zeros_like(self.Wy).normal_(
            mean=0.0, std=0.05).to(self.device)

        # NEW: Recurrent error path weights (lateral dendrites)
        # theta_h_recurrent carries errors backward through time
        self.theta_h_recurrent = torch.zeros_like(self.Wh).normal_(
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
                # Standard RNN: h[t+1] = f(Wh @ h[t] + Wx @ x[t])
                self.hu[i+1] = self.fn(self.Wh @ self.hu[i] + self.Wx @ inp)
                self.hx[i+1] = self.hu[i+1].clone()
                self.y[i] = linear(self.Wy @ self.hu[i+1])
            return self.y

    def update_weights(self, target_seq, epoch_num, mask_seq=None):
        """
        target_seq: (T, output_size, B)
        mask_seq: (T, B) optional mask for padding

        Temporal DLL: errors can flow backward through time via theta_h_recurrent
        """
        with torch.no_grad():
            ehs = torch.zeros_like(self.hu).to(self.device)
            eys = torch.zeros_like(self.y).to(self.device)

            # Backward pass for dendritic errors (now with temporal integration)
            for i, tar in reversed(list(enumerate(target_seq))):
                eys[i] = tar - self.y[i]

                # Apply mask
                if mask_seq is not None:
                    eys[i] = eys[i] * mask_seq[i].unsqueeze(0)

                # Base error from output layer
                deltah = self.theta_y.T @ (
                    eys[i] * linear_deriv(self.Wy @ self.hu[i+1])
                )

                # Standard DLL: error from next hidden state
                if i < len(target_seq)-1:
                    fn_deriv = self.fn_deriv(
                        self.Wh @ self.hu[i+1] + self.Wx @ self.inputs[i]
                    )
                    deltah += self.theta_h.T @ (ehs[i+1] * fn_deriv)

                # NEW: Temporal path - error from PREVIOUS hidden state (backward through time)
                # This is the recurrent error integration: past errors inform current credit
                if i > 0:
                    fn_deriv = self.fn_deriv(
                        self.Wh @ self.hu[i] + self.Wx @ self.inputs[i-1]
                    )
                    # The recurrent path brings error from t-1 to t (backward in time)
                    # Discounted by temporal_decay to weight recent errors more
                    deltah += self.temporal_decay * self.theta_h_recurrent.T @ (ehs[i-1] * fn_deriv)

                ehs[i] = deltah

            if self.noise:
                ehs = ehs * (1 + 2 * self.noise * (torch.rand_like(ehs) - 0.5))

            dWy = torch.zeros_like(self.Wy).to(self.device)
            dWx = torch.zeros_like(self.Wx).to(self.device)
            dWh = torch.zeros_like(self.Wh).to(self.device)
            dtheta_y_T = torch.zeros_like(self.Wy.T).to(self.device)
            dtheta_h_T = torch.zeros_like(self.Wh.T).to(self.device)
            dtheta_h_recurrent_T = torch.zeros_like(self.Wh.T).to(self.device)

            # Forward gradient accumulation (with temporal credit)
            for i, inp in reversed(list(enumerate(self.inputs))):
                fn_deriv = self.fn_deriv(self.Wh @ self.hu[i] + self.Wx @ inp)

                # Standard weight updates
                dWy += (eys[i] * linear_deriv(self.Wy @ self.hu[i+1])) @ self.hu[i+1].T
                dWx += (ehs[i] * fn_deriv) @ inp.T
                dWh += (ehs[i] * fn_deriv) @ self.hu[i].T

                # Standard theta updates
                dtheta_y_T -= ehs[i] @ (
                    eys[i] * linear_deriv(self.Wy @ self.hu[i+1])
                ).T
                if i >= 1:
                    dtheta_h_T -= ehs[i-1] @ (ehs[i] * fn_deriv).T

                # NEW: Recurrent theta update
                # theta_h_recurrent is trained to predict the error that comes from
                # the hidden state at the PREVIOUS timestep
                # This allows it to carry credit backward through time
                if i >= 2:
                    fn_deriv_prev = self.fn_deriv(
                        self.Wh @ self.hu[i-1] + self.Wx @ self.inputs[i-2]
                    )
                    dtheta_h_recurrent_T -= self.temporal_decay * ehs[i-2] @ (ehs[i-1] * fn_deriv_prev).T

            if not self.noclamp:
                dWy = torch.clamp(dWy, -self.clamp_val, self.clamp_val)
                dWx = torch.clamp(dWx, -self.clamp_val, self.clamp_val)
                dWh = torch.clamp(dWh, -self.clamp_val, self.clamp_val)
                dtheta_y_T = torch.clamp(dtheta_y_T, -self.clamp_val, self.clamp_val)
                dtheta_h_T = torch.clamp(dtheta_h_T, -self.clamp_val, self.clamp_val)
                dtheta_h_recurrent_T = torch.clamp(dtheta_h_recurrent_T, -self.clamp_val, self.clamp_val)

            # Apply updates
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
                self.theta_h_recurrent += (
                    self.weight_learning_rate * dtheta_h_recurrent_T.T /
                    self.theta_update_discount
                )


# ======================================================================
#  PyTorch wrapper for sequence tagging with temporal DLL
# ======================================================================

class TemporalDLLRNN(nn.Module):
    """
    Wrapper for Temporal DLL RNN with recurrent error paths.
    Implements biologically plausible temporal credit assignment via lateral dendrite
    integration (theta_h_recurrent).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_classes: int,
        seq_len: int,
        batch_size: int,
        device: str = "cpu",
        weight_lr: float = 1e-3,
        theta_update_discount: float = 10.0,
        fix_theta_until: int = 1,
        noclamp: bool = False,
        noise: float = 0.0,
        temporal_decay: float = 0.9,
    ):
        super().__init__()
        self.device = device
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_classes = num_classes

        # Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Setup args for temporal DLL core
        args = SimpleNamespace(
            seq_len=seq_len,
            hidden_size=hidden_size,
            batch_size=batch_size,
            input_size=embed_dim,
            output_size=num_classes,
            fn=lambda x: torch.tanh(x),
            fn_deriv=lambda x: 1 - torch.tanh(x) ** 2,
            weight_learning_rate=weight_lr,
            theta_update_discount=theta_update_discount,
            fix_theta_until=fix_theta_until,
            noclamp=noclamp,
            noise=noise,
            temporal_decay=temporal_decay,
        )

        # Temporal DLL core
        self.dll_core = TemporalDLL_RNN_Model(args, device=device)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: (B, T)
        returns: (B, T, num_classes) logits
        """
        B, T = token_ids.shape

        # Embed tokens
        embeddings = self.embedding(token_ids)  # (B, T, embed_dim)

        # Transpose to (T, embed_dim, B) for DLL
        inputs_seq = embeddings.permute(1, 2, 0).to(self.device)

        # Forward pass
        outputs = self.dll_core.forward(inputs_seq)  # (T, num_classes, B)

        # Transpose back to (B, T, num_classes)
        outputs = outputs.permute(2, 0, 1)  # (B, T, num_classes)

        return outputs

    def dll_update(self, labels: torch.Tensor, mask: torch.Tensor, epoch: int):
        """
        Perform temporal DLL weight update step.

        labels: (B, T) integer labels
        mask: (B, T) float mask (1.0 for real tokens, 0.0 for padding)
        epoch: current epoch number
        """
        B, T = labels.shape
        num_classes = self.num_classes

        # Build one-hot targets: (T, num_classes, B)
        target_seq = torch.zeros(
            T, num_classes, B,
            device=self.device,
            dtype=torch.float32
        )

        # Transpose mask to (T, B)
        mask_seq = mask.T.float()

        # Fill in one-hot targets for real tokens only
        for t in range(T):
            mask_t = mask[:, t]  # (B,)
            labels_t = labels[:, t]  # (B,)

            # Get valid indices (not padding)
            valid_mask = mask_t.bool()
            valid_indices = torch.where(valid_mask)[0]
            valid_labels = labels_t[valid_mask]

            # Build one-hot
            targets_onehot_t = torch.zeros(num_classes, B, device=self.device)
            targets_onehot_t[valid_labels, valid_indices] = 1.0

            # Store transposed: (num_classes, B)
            target_seq[t] = targets_onehot_t
            mask_seq[t] = mask_t

        # Call temporal DLL update
        self.dll_core.update_weights(target_seq, epoch, mask_seq=mask_seq)
