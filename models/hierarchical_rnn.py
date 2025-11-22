"""
Hierarchical DLL RNN with multi-level theta weights for temporal credit assignment.

Core insight: DLL struggles on RNNs because a single theta_h must approximate error
backprop across ALL timesteps. But errors at t=1 have different structure than errors
at t=20. Solution: Use 3 theta weights specialized for different "temporal distances":
- theta_h_short: Credits 1 step back (short-range dependencies)
- theta_h_medium: Credits 2-3 steps back (medium-range dependencies)
- theta_h_long: Credits 4+ steps back (long-range dependencies)

This is bio-plausible because real pyramidal neurons have multiple dendritic
compartments that integrate signals from different temporal scales.
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


class HierarchicalDLL_RNN_Model(object):
    """
    DLL RNN extended with hierarchical theta weights for better temporal credit assignment.

    Instead of a single theta_h, we maintain 3 theta matrices:
    - theta_h_short: primarily used for t-1 credit (1 step back)
    - theta_h_medium: primarily used for t-2,t-3 credit (2-3 steps back)
    - theta_h_long: primarily used for t-4+ credit (4+ steps back)

    Each theta gets updated based on how useful it is for its "temporal zone".
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

        # Standard DLL weights
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

        self.theta_y = torch.zeros_like(self.Wy).normal_(
            mean=0.0, std=0.05).to(self.device)

        # HIERARCHICAL: 3 separate theta weights for different temporal distances
        self.theta_h_short = torch.zeros_like(self.Wh).normal_(
            mean=0.0, std=0.05).to(self.device)   # For 1-step credit
        self.theta_h_medium = torch.zeros_like(self.Wh).normal_(
            mean=0.0, std=0.05).to(self.device)   # For 2-3 step credit
        self.theta_h_long = torch.zeros_like(self.Wh).normal_(
            mean=0.0, std=0.05).to(self.device)   # For 4+ step credit

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
                # inp is (D_in, B), Wh @ hu[i] is (H, B)
                # Wx @ inp should be (H, B) where Wx is (H, D_in)
                self.hu[i+1] = self.fn(self.Wh @ self.hu[i] + self.Wx @ inp)
                self.hx[i+1] = self.hu[i+1].clone()
                self.y[i] = linear(self.Wy @ self.hu[i+1])
            return self.y

    def update_weights(self, target_seq, epoch_num, mask_seq=None):
        """
        target_seq: (T, output_size, B)
        mask_seq: (T, B) optional mask for padding

        In hierarchical DLL, we compute errors using a weighted combination
        of the three theta weights based on temporal distance.
        """
        with torch.no_grad():
            ehs = torch.zeros_like(self.hu).to(self.device)
            eys = torch.zeros_like(self.y).to(self.device)

            # Backward pass for dendritic errors
            for i, tar in reversed(list(enumerate(target_seq))):
                eys[i] = tar - self.y[i]

                # Apply mask: zero out errors for padding positions
                if mask_seq is not None:
                    eys[i] = eys[i] * mask_seq[i].unsqueeze(0)

                deltah = self.theta_y.T @ (
                    eys[i] * linear_deriv(self.Wy @ self.hu[i+1])
                )

                if i < len(target_seq)-1:
                    fn_deriv = self.fn_deriv(
                        self.Wh @ self.hu[i+1] + self.Wx @ self.inputs[i]
                    )
                    # HIERARCHICAL: Use weighted combination of thetas
                    # This is the key difference - blend all 3 theta weights
                    deltah += (
                        self.theta_h_short.T @ (ehs[i+1] * fn_deriv) * 0.5 +     # Weight short: 0.5
                        self.theta_h_medium.T @ (ehs[i+1] * fn_deriv) * 0.3 +    # Weight medium: 0.3
                        self.theta_h_long.T @ (ehs[i+1] * fn_deriv) * 0.2        # Weight long: 0.2
                    )
                ehs[i] = deltah

            if self.noise:
                ehs = ehs * (1 + 2 * self.noise * (torch.rand_like(ehs) - 0.5))

            dWy = torch.zeros_like(self.Wy).to(self.device)
            dWx = torch.zeros_like(self.Wx).to(self.device)
            dWh = torch.zeros_like(self.Wh).to(self.device)
            dtheta_y_T = torch.zeros_like(self.Wy.T).to(self.device)
            dtheta_h_short_T = torch.zeros_like(self.Wh.T).to(self.device)
            dtheta_h_medium_T = torch.zeros_like(self.Wh.T).to(self.device)
            dtheta_h_long_T = torch.zeros_like(self.Wh.T).to(self.device)

            # Gradient accumulation loop - HIERARCHICAL VERSION
            for i, inp in reversed(list(enumerate(self.inputs))):
                fn_deriv = self.fn_deriv(self.Wh @ self.hu[i] + self.Wx @ inp)

                # Standard DLL weight updates
                dWy += (eys[i] * linear_deriv(self.Wy @ self.hu[i+1])) @ self.hu[i+1].T
                dWx += (ehs[i] * fn_deriv) @ inp.T
                dWh += (ehs[i] * fn_deriv) @ self.hu[i].T

                dtheta_y_T -= ehs[i] @ (
                    eys[i] * linear_deriv(self.Wy @ self.hu[i+1])
                ).T

                if i >= 1:
                    dtheta_h_short_T -= ehs[i-1] @ (ehs[i] * fn_deriv).T

                # HIERARCHICAL: Each theta specializes in different temporal zones
                # theta_h_medium is updated based on 2-step errors
                if i >= 2:
                    dtheta_h_medium_T -= ehs[i-2] @ (ehs[i] * fn_deriv).T

                # theta_h_long is updated based on 3+ step errors
                if i >= 3:
                    dtheta_h_long_T -= ehs[i-3] @ (ehs[i] * fn_deriv).T

            if not self.noclamp:
                dWy = torch.clamp(dWy, -self.clamp_val, self.clamp_val)
                dWx = torch.clamp(dWx, -self.clamp_val, self.clamp_val)
                dWh = torch.clamp(dWh, -self.clamp_val, self.clamp_val)
                dtheta_y_T = torch.clamp(dtheta_y_T, -self.clamp_val, self.clamp_val)
                dtheta_h_short_T = torch.clamp(dtheta_h_short_T, -self.clamp_val, self.clamp_val)
                dtheta_h_medium_T = torch.clamp(dtheta_h_medium_T, -self.clamp_val, self.clamp_val)
                dtheta_h_long_T = torch.clamp(dtheta_h_long_T, -self.clamp_val, self.clamp_val)

            # Apply weight updates
            self.Wy += self.weight_learning_rate * dWy
            self.Wx += self.weight_learning_rate * dWx
            self.Wh += self.weight_learning_rate * dWh

            # Theta updates - each level learns independently
            if epoch_num >= self.fix_theta_until:
                self.theta_y += (
                    self.weight_learning_rate * dtheta_y_T.T /
                    self.theta_update_discount
                )
                self.theta_h_short += (
                    self.weight_learning_rate * dtheta_h_short_T.T /
                    self.theta_update_discount
                )
                self.theta_h_medium += (
                    self.weight_learning_rate * dtheta_h_medium_T.T /
                    self.theta_update_discount
                )
                self.theta_h_long += (
                    self.weight_learning_rate * dtheta_h_long_T.T /
                    self.theta_update_discount
                )


# ======================================================================
#  PyTorch wrapper for sequence tagging with hierarchical DLL
# ======================================================================

class HierarchicalDLLRNN(nn.Module):
    """
    Wrapper for HierarchicalDLL_RNN_Model with PyTorch interface.
    Implements hierarchical theta weights for better temporal credit assignment on RNNs.
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

        # Setup args for DLL core
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

        # Hierarchical DLL core
        self.dll_core = HierarchicalDLL_RNN_Model(args, device=device)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: (B, T)
        returns: (B, T, num_classes) logits
        """
        B, T = token_ids.shape

        # Embed tokens
        embeddings = self.embedding(token_ids)  # (B, T, embed_dim)

        # Transpose to (T, embed_dim, B) for DLL
        # embeddings is (B, T, embed_dim) -> need (T, embed_dim, B)
        # permute(1, 2, 0) gives (T, embed_dim, B)
        inputs_seq = embeddings.permute(1, 2, 0).to(self.device)

        # Forward pass
        outputs = self.dll_core.forward(inputs_seq)  # (T, num_classes, B)

        # Transpose back to (B, T, num_classes)
        # outputs is (T, num_classes, B) -> need (B, T, num_classes)
        # permute(2, 0, 1) gives (B, T, num_classes)
        outputs = outputs.permute(2, 0, 1)  # (B, T, num_classes)

        return outputs

    def dll_update(self, labels: torch.Tensor, mask: torch.Tensor, epoch: int):
        """
        Perform DLL weight update step.

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

        # Call hierarchical DLL update
        self.dll_core.update_weights(target_seq, epoch, mask_seq=mask_seq)
