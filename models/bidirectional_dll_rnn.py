"""
Bidirectional DLL-RNN for Sequence Tagging

Key insight: Standard unidirectional RNNs only see past context. For tasks like POS tagging,
future context is equally important (e.g., "bank" as noun vs verb depends on what comes after).

Solution: Run TWO separate DLL-RNN cores:
  1. Forward RNN: processes left → right (sees past)
  2. Backward RNN: processes right → left (sees future)

At each timestep, concatenate both hidden states: h[t] = [h_fwd[t]; h_bwd[t]]
This gives each position access to full bidirectional context.

Both RNNs learn via DLL independently - no global coordination needed.
Biologically: Like two separate neural pathways that integrate at output layer.
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


class BiDLL_RNN_Model(object):
    """
    Bidirectional DLL-RNN with two independent DLL cores.

    Forward core processes sequences left→right
    Backward core processes sequences right→left
    Both hidden states concatenated before output layer
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

        # ===== FORWARD RNN WEIGHTS =====
        self.Wh_fwd = torch.empty([self.hidden_size, self.hidden_size]).normal_(
            mean=0.0, std=0.05).to(self.device)
        self.Wx_fwd = torch.empty([self.hidden_size, self.input_size]).normal_(
            mean=0.0, std=0.05).to(self.device)
        self.h0_fwd = torch.empty([self.hidden_size, self.batch_size]).normal_(
            mean=0.0, std=0.05).to(self.device)

        # Forward activations
        self.hu_fwd = torch.empty(
            [self.seq_len+1, self.hidden_size, self.batch_size]
        ).to(self.device)
        self.hx_fwd = torch.zeros_like(self.hu_fwd).to(self.device)

        # Forward theta weights
        self.theta_h_fwd = torch.zeros_like(self.Wh_fwd).normal_(
            mean=0.0, std=0.05).to(self.device)

        # ===== BACKWARD RNN WEIGHTS =====
        self.Wh_bwd = torch.empty([self.hidden_size, self.hidden_size]).normal_(
            mean=0.0, std=0.05).to(self.device)
        self.Wx_bwd = torch.empty([self.hidden_size, self.input_size]).normal_(
            mean=0.0, std=0.05).to(self.device)
        self.h0_bwd = torch.empty([self.hidden_size, self.batch_size]).normal_(
            mean=0.0, std=0.05).to(self.device)

        # Backward activations
        self.hu_bwd = torch.empty(
            [self.seq_len+1, self.hidden_size, self.batch_size]
        ).to(self.device)
        self.hx_bwd = torch.zeros_like(self.hu_bwd).to(self.device)

        # Backward theta weights
        self.theta_h_bwd = torch.zeros_like(self.Wh_bwd).normal_(
            mean=0.0, std=0.05).to(self.device)

        # ===== OUTPUT LAYER (operates on concatenated hidden states) =====
        # Input to output layer is 2*hidden_size (fwd + bwd concatenated)
        self.Wy = torch.empty([self.output_size, 2 * self.hidden_size]).normal_(
            mean=0.0, std=0.05).to(self.device)

        # Output predictions
        self.y = torch.empty(
            [self.seq_len, self.output_size, self.batch_size]
        ).to(self.device)

        # Theta for output layer
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

            # ===== FORWARD PASS (left → right) =====
            self.hu_fwd[0] = self.h0_fwd.clone()
            self.hx_fwd[0] = self.h0_fwd.clone()

            for i, inp in enumerate(inputs_seq):
                self.hu_fwd[i+1] = self.fn(self.Wh_fwd @ self.hu_fwd[i] + self.Wx_fwd @ inp)
                self.hx_fwd[i+1] = self.hu_fwd[i+1].clone()

            # ===== BACKWARD PASS (right → left) =====
            # Start from the last timestep and go backward
            self.hu_bwd[self.seq_len] = self.h0_bwd.clone()
            self.hx_bwd[self.seq_len] = self.h0_bwd.clone()

            for i in reversed(range(self.seq_len)):
                self.hu_bwd[i] = self.fn(self.Wh_bwd @ self.hu_bwd[i+1] + self.Wx_bwd @ inputs_seq[i])
                self.hx_bwd[i] = self.hu_bwd[i].clone()

            # ===== OUTPUT LAYER (concatenate fwd + bwd) =====
            for i in range(self.seq_len):
                # Concatenate forward and backward hidden states
                h_concat = torch.cat([self.hu_fwd[i+1], self.hu_bwd[i]], dim=0)  # (2*H, B)
                self.y[i] = linear(self.Wy @ h_concat)

            return self.y

    def update_weights(self, target_seq, epoch_num, mask_seq=None):
        """
        target_seq: (T, output_size, B)
        mask_seq: (T, B) optional mask

        Update both forward and backward RNNs independently using DLL
        """
        with torch.no_grad():
            # Error signals for both directions
            ehs_fwd = torch.zeros_like(self.hu_fwd).to(self.device)
            ehs_bwd = torch.zeros_like(self.hu_bwd).to(self.device)
            eys = torch.zeros_like(self.y).to(self.device)

            # ===== BACKWARD PASS FOR DENDRITIC ERRORS =====
            for i, tar in reversed(list(enumerate(target_seq))):
                eys[i] = tar - self.y[i]

                # Apply mask
                if mask_seq is not None:
                    eys[i] = eys[i] * mask_seq[i].unsqueeze(0)

                # Concatenate hidden states for output layer gradient
                h_concat = torch.cat([self.hu_fwd[i+1], self.hu_bwd[i]], dim=0)

                # Output layer error
                output_error = eys[i] * linear_deriv(self.Wy @ h_concat)

                # Split error back to forward and backward components
                # theta_y has shape (2*H, C), so theta_y.T @ error gives (2*H, B)
                deltah_concat = self.theta_y.T @ output_error
                deltah_fwd_from_output = deltah_concat[:self.hidden_size, :]
                deltah_bwd_from_output = deltah_concat[self.hidden_size:, :]

                # ===== FORWARD RNN ERROR =====
                deltah_fwd = deltah_fwd_from_output.clone()
                if i < len(target_seq)-1:
                    fn_deriv_fwd = self.fn_deriv(
                        self.Wh_fwd @ self.hu_fwd[i+1] + self.Wx_fwd @ self.inputs[i]
                    )
                    deltah_fwd += self.theta_h_fwd.T @ (ehs_fwd[i+1] * fn_deriv_fwd)
                ehs_fwd[i] = deltah_fwd

                # ===== BACKWARD RNN ERROR =====
                deltah_bwd = deltah_bwd_from_output.clone()
                if i > 0:
                    fn_deriv_bwd = self.fn_deriv(
                        self.Wh_bwd @ self.hu_bwd[i] + self.Wx_bwd @ self.inputs[i-1]
                    )
                    deltah_bwd += self.theta_h_bwd.T @ (ehs_bwd[i-1] * fn_deriv_bwd)
                ehs_bwd[i] = deltah_bwd

            if self.noise:
                ehs_fwd = ehs_fwd * (1 + 2 * self.noise * (torch.rand_like(ehs_fwd) - 0.5))
                ehs_bwd = ehs_bwd * (1 + 2 * self.noise * (torch.rand_like(ehs_bwd) - 0.5))

            # ===== WEIGHT GRADIENT ACCUMULATION =====
            dWy = torch.zeros_like(self.Wy).to(self.device)
            dWx_fwd = torch.zeros_like(self.Wx_fwd).to(self.device)
            dWh_fwd = torch.zeros_like(self.Wh_fwd).to(self.device)
            dWx_bwd = torch.zeros_like(self.Wx_bwd).to(self.device)
            dWh_bwd = torch.zeros_like(self.Wh_bwd).to(self.device)

            dtheta_y_T = torch.zeros_like(self.Wy.T).to(self.device)
            dtheta_h_fwd_T = torch.zeros_like(self.Wh_fwd.T).to(self.device)
            dtheta_h_bwd_T = torch.zeros_like(self.Wh_bwd.T).to(self.device)

            for i, inp in reversed(list(enumerate(self.inputs))):
                h_concat = torch.cat([self.hu_fwd[i+1], self.hu_bwd[i]], dim=0)

                # Output layer gradients
                dWy += (eys[i] * linear_deriv(self.Wy @ h_concat)) @ h_concat.T

                # Forward RNN gradients
                fn_deriv_fwd = self.fn_deriv(self.Wh_fwd @ self.hu_fwd[i] + self.Wx_fwd @ inp)
                dWx_fwd += (ehs_fwd[i] * fn_deriv_fwd) @ inp.T
                dWh_fwd += (ehs_fwd[i] * fn_deriv_fwd) @ self.hu_fwd[i].T

                # Backward RNN gradients
                fn_deriv_bwd = self.fn_deriv(self.Wh_bwd @ self.hu_bwd[i+1] + self.Wx_bwd @ inp)
                dWx_bwd += (ehs_bwd[i] * fn_deriv_bwd) @ inp.T
                dWh_bwd += (ehs_bwd[i] * fn_deriv_bwd) @ self.hu_bwd[i+1].T

                # Theta gradients
                dtheta_y_T -= torch.cat([ehs_fwd[i], ehs_bwd[i]], dim=0) @ (
                    eys[i] * linear_deriv(self.Wy @ h_concat)
                ).T

                if i >= 1:
                    dtheta_h_fwd_T -= ehs_fwd[i-1] @ (ehs_fwd[i] * fn_deriv_fwd).T
                    dtheta_h_bwd_T -= ehs_bwd[i+1] @ (ehs_bwd[i] * fn_deriv_bwd).T

            if not self.noclamp:
                dWy = torch.clamp(dWy, -self.clamp_val, self.clamp_val)
                dWx_fwd = torch.clamp(dWx_fwd, -self.clamp_val, self.clamp_val)
                dWh_fwd = torch.clamp(dWh_fwd, -self.clamp_val, self.clamp_val)
                dWx_bwd = torch.clamp(dWx_bwd, -self.clamp_val, self.clamp_val)
                dWh_bwd = torch.clamp(dWh_bwd, -self.clamp_val, self.clamp_val)
                dtheta_y_T = torch.clamp(dtheta_y_T, -self.clamp_val, self.clamp_val)
                dtheta_h_fwd_T = torch.clamp(dtheta_h_fwd_T, -self.clamp_val, self.clamp_val)
                dtheta_h_bwd_T = torch.clamp(dtheta_h_bwd_T, -self.clamp_val, self.clamp_val)

            # ===== APPLY UPDATES =====
            self.Wy += self.weight_learning_rate * dWy
            self.Wx_fwd += self.weight_learning_rate * dWx_fwd
            self.Wh_fwd += self.weight_learning_rate * dWh_fwd
            self.Wx_bwd += self.weight_learning_rate * dWx_bwd
            self.Wh_bwd += self.weight_learning_rate * dWh_bwd

            if epoch_num >= self.fix_theta_until:
                self.theta_y += (
                    self.weight_learning_rate * dtheta_y_T.T /
                    self.theta_update_discount
                )
                self.theta_h_fwd += (
                    self.weight_learning_rate * dtheta_h_fwd_T.T /
                    self.theta_update_discount
                )
                self.theta_h_bwd += (
                    self.weight_learning_rate * dtheta_h_bwd_T.T /
                    self.theta_update_discount
                )


# ======================================================================
#  PyTorch wrapper for sequence tagging with bidirectional DLL
# ======================================================================

class BiDLLRNN(nn.Module):
    """
    Wrapper for Bidirectional DLL-RNN.

    Uses two independent DLL-RNN cores (forward + backward) with concatenated
    hidden states feeding into the output layer.
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
    ):
        super().__init__()
        self.device = device
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_classes = num_classes

        # Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Setup args for bidirectional DLL core
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
        )

        # Bidirectional DLL core
        self.dll_core = BiDLL_RNN_Model(args, device=device)

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
        Perform bidirectional DLL weight update step.

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

        # Call bidirectional DLL update
        self.dll_core.update_weights(target_seq, epoch, mask_seq=mask_seq)
