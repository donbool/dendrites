"""
Predictive Coding DLL for RNNs

Core insight: Standard DLL on RNNs lacks temporal credit assignment because errors
only come from the immediate task loss. But in recurrent systems, a weight's "true"
utility is partially determined by how well it predicts future hidden states.

Solution: Add a parallel predictive objective. The network learns TWO tasks:
  1. Task objective: predict the label at each timestep (standard DLL)
  2. Predictive objective: predict the hidden state at the next timestep

The prediction error (h[t+1] - pred_h[t+1]) creates a temporal learning signal that
implicitly credits weights for maintaining good temporal dynamics. Both errors update
the same forward weights (Wh, Wx) but from different perspectives.

Biological plausibility:
- Basal dendrites integrate task errors (direct feedback)
- Apical dendrites integrate prediction errors (from deeper layers/future)
- Both compartments contribute to the same synaptic weight updates
- This mirrors actual cortical pyramidal neuron architecture
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


class PredictiveCoderRNN_Model(object):
    """
    DLL RNN with auxiliary predictive coding objective.

    Two DLL cores:
    1. Main core: learns task (classification)
    2. Predictor core: learns to predict h[t+1] from h[t]

    Both cores update the shared forward weights (Wh, Wx) based on their respective
    errors, creating a richer learning signal with implicit temporal credit assignment.
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
        self.pred_weight = getattr(args, 'pred_weight', 0.5)  # Balance between task and prediction

        # Shared forward weights (used by both task and prediction)
        self.Wh = torch.empty([self.hidden_size, self.hidden_size]).normal_(
            mean=0.0, std=0.05).to(self.device)
        self.Wx = torch.empty([self.hidden_size, self.input_size]).normal_(
            mean=0.0, std=0.05).to(self.device)

        # Task output layer
        self.Wy = torch.empty([self.output_size, self.hidden_size]).normal_(
            mean=0.0, std=0.05).to(self.device)

        # Prediction output layer: predicts h[t+1] from h[t]
        # This is a dendritic compartment that learns temporal predictions
        self.W_pred = torch.empty([self.hidden_size, self.hidden_size]).normal_(
            mean=0.0, std=0.05).to(self.device)

        self.h0 = torch.empty([self.hidden_size, self.batch_size]).normal_(
            mean=0.0, std=0.05).to(self.device)

        # Activation history
        self.hu = torch.empty(
            [self.seq_len+1, self.hidden_size, self.batch_size]
        ).to(self.device)
        self.hx = torch.zeros_like(self.hu).to(self.device)

        # Task outputs
        self.y = torch.empty(
            [self.seq_len, self.output_size, self.batch_size]
        ).to(self.device)

        # Predicted hidden states
        self.pred_h = torch.empty(
            [self.seq_len, self.hidden_size, self.batch_size]
        ).to(self.device)

        # DLL theta weights for task learning
        self.theta_h = torch.zeros_like(self.Wh).normal_(
            mean=0.0, std=0.05).to(self.device)
        self.theta_y = torch.zeros_like(self.Wy).normal_(
            mean=0.0, std=0.05).to(self.device)

        # DLL theta weights for prediction learning
        # These learn to approximate errors in the hidden state predictions
        self.theta_h_pred = torch.zeros_like(self.W_pred).normal_(
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
                # Forward pass: h[t+1] = f(Wh @ h[t] + Wx @ x[t])
                self.hu[i+1] = self.fn(self.Wh @ self.hu[i] + self.Wx @ inp)
                self.hx[i+1] = self.hu[i+1].clone()

                # Task output: y[t] = Wy @ h[t+1]
                self.y[i] = linear(self.Wy @ self.hu[i+1])

                # Prediction output: predict h[t+1] from h[t]
                # This is computed after we know h[t+1], but during learning
                # the prediction network learns to anticipate it
                self.pred_h[i] = self.fn(self.W_pred @ self.hu[i])

            return self.y

    def update_weights(self, target_seq, epoch_num, mask_seq=None):
        """
        target_seq: (T, output_size, B)
        mask_seq: (T, B) optional mask for padding

        Learning uses TWO error signals:
        1. Task errors: e_y = target - y (standard DLL)
        2. Prediction errors: e_pred = h[t+1] - pred_h[t] (temporal credit)
        """
        with torch.no_grad():
            # Error signals
            ehs_task = torch.zeros_like(self.hu).to(self.device)  # Task-driven hidden errors
            ehs_pred = torch.zeros_like(self.hu).to(self.device)  # Prediction-driven hidden errors
            eys = torch.zeros_like(self.y).to(self.device)  # Output errors

            # Backward pass: compute errors from task objective
            for i, tar in reversed(list(enumerate(target_seq))):
                eys[i] = tar - self.y[i]

                # Apply mask
                if mask_seq is not None:
                    eys[i] = eys[i] * mask_seq[i].unsqueeze(0)

                # Task error propagation (standard DLL)
                deltah_task = self.theta_y.T @ (
                    eys[i] * linear_deriv(self.Wy @ self.hu[i+1])
                )
                if i < len(target_seq)-1:
                    fn_deriv = self.fn_deriv(
                        self.Wh @ self.hu[i+1] + self.Wx @ self.inputs[i]
                    )
                    deltah_task += self.theta_h.T @ (ehs_task[i+1] * fn_deriv)

                ehs_task[i] = deltah_task

            # Backward pass: compute errors from prediction objective
            for i in reversed(range(len(target_seq))):
                # Prediction error: how well did we predict h[t+1]?
                # This error should credit the weights that help with temporal prediction
                pred_error = self.hu[i+1] - self.pred_h[i]

                # Scale prediction error to prevent it from dominating
                # The prediction task is auxiliary, not primary
                pred_error = 0.1 * pred_error

                # Propagate prediction error back through the prediction network
                fn_deriv_pred = self.fn_deriv(self.W_pred @ self.hu[i])

                # The prediction error contributes to hidden state credit
                deltah_pred = self.theta_h_pred.T @ (pred_error * fn_deriv_pred)

                # Prediction errors from future timesteps also matter (but weakly)
                if i < len(target_seq)-1:
                    fn_deriv_pred_next = self.fn_deriv(self.W_pred @ self.hu[i+1])
                    deltah_pred += 0.1 * self.theta_h_pred.T @ (ehs_pred[i+1] * fn_deriv_pred_next)

                ehs_pred[i] = deltah_pred

            # Combine both error signals (weighted by pred_weight)
            # Default: 95% task, 5% prediction (auxiliary)
            ehs = (1.0 - self.pred_weight) * ehs_task + self.pred_weight * ehs_pred

            if self.noise:
                ehs = ehs * (1 + 2 * self.noise * (torch.rand_like(ehs) - 0.5))

            # Weight gradients accumulation
            dWy = torch.zeros_like(self.Wy).to(self.device)
            dWx = torch.zeros_like(self.Wx).to(self.device)
            dWh = torch.zeros_like(self.Wh).to(self.device)
            dW_pred = torch.zeros_like(self.W_pred).to(self.device)

            dtheta_y_T = torch.zeros_like(self.Wy.T).to(self.device)
            dtheta_h_T = torch.zeros_like(self.Wh.T).to(self.device)
            dtheta_h_pred_T = torch.zeros_like(self.W_pred.T).to(self.device)

            # Gradient accumulation
            for i, inp in reversed(list(enumerate(self.inputs))):
                fn_deriv = self.fn_deriv(self.Wh @ self.hu[i] + self.Wx @ inp)
                fn_deriv_pred = self.fn_deriv(self.W_pred @ self.hu[i])

                # Task-driven weight updates (standard DLL)
                dWy += (eys[i] * linear_deriv(self.Wy @ self.hu[i+1])) @ self.hu[i+1].T
                dWx += (ehs[i] * fn_deriv) @ inp.T
                dWh += (ehs[i] * fn_deriv) @ self.hu[i].T

                # Prediction-driven weight updates
                pred_error = self.hu[i+1] - self.pred_h[i]
                dW_pred += (pred_error * fn_deriv_pred) @ self.hu[i].T

                # Theta updates (task objective)
                dtheta_y_T -= ehs[i] @ (
                    eys[i] * linear_deriv(self.Wy @ self.hu[i+1])
                ).T
                if i >= 1:
                    dtheta_h_T -= ehs[i-1] @ (ehs[i] * fn_deriv).T

                # Theta updates (prediction objective)
                if i >= 1:
                    dtheta_h_pred_T -= (ehs_pred[i-1] * fn_deriv_pred) @ (pred_error * fn_deriv_pred).T

            if not self.noclamp:
                dWy = torch.clamp(dWy, -self.clamp_val, self.clamp_val)
                dWx = torch.clamp(dWx, -self.clamp_val, self.clamp_val)
                dWh = torch.clamp(dWh, -self.clamp_val, self.clamp_val)
                dW_pred = torch.clamp(dW_pred, -self.clamp_val, self.clamp_val)
                dtheta_y_T = torch.clamp(dtheta_y_T, -self.clamp_val, self.clamp_val)
                dtheta_h_T = torch.clamp(dtheta_h_T, -self.clamp_val, self.clamp_val)
                dtheta_h_pred_T = torch.clamp(dtheta_h_pred_T, -self.clamp_val, self.clamp_val)

            # Apply updates
            self.Wy += self.weight_learning_rate * dWy
            self.Wx += self.weight_learning_rate * dWx
            self.Wh += self.weight_learning_rate * dWh
            self.W_pred += self.weight_learning_rate * dW_pred

            if epoch_num >= self.fix_theta_until:
                self.theta_y += (
                    self.weight_learning_rate * dtheta_y_T.T /
                    self.theta_update_discount
                )
                self.theta_h += (
                    self.weight_learning_rate * dtheta_h_T.T /
                    self.theta_update_discount
                )
                self.theta_h_pred += (
                    self.weight_learning_rate * dtheta_h_pred_T.T /
                    self.theta_update_discount
                )


# ======================================================================
#  PyTorch wrapper for sequence tagging with predictive coding DLL
# ======================================================================

class PredictiveCoderRNN(nn.Module):
    """
    Wrapper for Predictive Coding DLL RNN with dual objectives.
    Extends DLL with an auxiliary prediction task that provides implicit
    temporal credit assignment.
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
        pred_weight: float = 0.5,
    ):
        super().__init__()
        self.device = device
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_classes = num_classes

        # Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Setup args for predictive coding DLL core
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
            pred_weight=pred_weight,
        )

        # Predictive coding DLL core
        self.dll_core = PredictiveCoderRNN_Model(args, device=device)

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
        Perform predictive coding DLL weight update step.

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

        # Call predictive coding DLL update
        self.dll_core.update_weights(target_seq, epoch, mask_seq=mask_seq)
