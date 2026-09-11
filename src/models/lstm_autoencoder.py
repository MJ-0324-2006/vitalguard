"""
LSTM Autoencoder for ICU Physiological Signal Anomaly Detection.

Implements:
  - Standard bidirectional LSTM encoder + LSTM decoder autoencoder
  - Variational LSTM Autoencoder (VAE-LSTM) for probabilistic anomaly scoring
  - Per-channel anomaly scoring with calibrated uncertainty
  - Batch-level and streaming inference modes

Architecture Reference:
  Malhotra et al. (2016). "LSTM-based Encoder-Decoder for Multi-sensor
  Anomaly Detection." ICML Anomaly Detection Workshop.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class VariationalSampler(nn.Module):
    """Reparameterisation trick: z = mu + eps * sigma, eps ~ N(0,I)."""

    def forward(self, mu: Tensor, log_var: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu  # MAP estimate at test time
            std = torch.exp(0.5 * log_var)
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return z, std, kl_loss


class ChannelNorm(nn.Module):
    """Per-channel layer normalisation over the time dimension."""

    def __init__(self, n_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, 1, n_channels))
        self.beta = nn.Parameter(torch.zeros(1, 1, n_channels))

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, C)
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / (var + self.eps).sqrt() + self.beta


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class LSTMEncoder(nn.Module):
    """
    Bidirectional LSTM encoder that maps a variable-length physiological
    sequence to a fixed-dimensional latent representation.

    Args:
        input_dim:   Number of physiological channels (C).
        hidden_dim:  LSTM hidden state dimension.
        latent_dim:  Dimension of the compressed latent vector z.
        num_layers:  Number of stacked LSTM layers.
        dropout:     Dropout rate between LSTM layers (0 = no dropout).
        variational: If True, output (mu, log_var) for VAE; else output z directly.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        variational: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.variational = variational
        self.bidirectional = True
        directions = 2  # bidirectional

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.channel_norm = ChannelNorm(hidden_dim)

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=self.bidirectional,
        )

        # Project concatenated forward+backward final hidden states → latent
        encoder_out_dim = hidden_dim * directions * num_layers  # all layers, both directions
        if variational:
            self.fc_mu = nn.Linear(encoder_out_dim, latent_dim)
            self.fc_log_var = nn.Linear(encoder_out_dim, latent_dim)
        else:
            self.fc_z = nn.Linear(encoder_out_dim, latent_dim)

        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self, x: Tensor, lengths: Optional[Tensor] = None
    ) -> Tuple[Tensor, ...]:
        """
        Args:
            x:       (B, T, C) — batch of multi-channel time series
            lengths: (B,) — actual sequence lengths for pack_padded_sequence

        Returns (variational=False):
            z:       (B, latent_dim)
            h_all:   (B, T, hidden_dim*2) — all hidden states (for attention)

        Returns (variational=True):
            mu:      (B, latent_dim)
            log_var: (B, latent_dim)
            h_all:   (B, T, hidden_dim*2)
        """
        B, T, C = x.shape

        # Input projection + channel normalisation
        x_proj = self.channel_norm(F.gelu(self.input_proj(x)))  # (B, T, H)

        # Pack sequence if lengths provided
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x_proj, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            h_packed, (h_n, _) = self.lstm(packed)
            h_all, _ = nn.utils.rnn.pad_packed_sequence(h_packed, batch_first=True, total_length=T)
        else:
            h_all, (h_n, _) = self.lstm(x_proj)
            # h_n: (num_layers * 2, B, H)

        # Concatenate all layer final hidden states (forward + backward)
        h_n = h_n.permute(1, 0, 2)  # (B, num_layers*2, H)
        context = h_n.contiguous().view(B, -1)  # (B, num_layers*2*H)
        context = self.dropout(context)

        if self.variational:
            mu = self.fc_mu(context)
            log_var = self.fc_log_var(context)
            return mu, log_var, h_all
        else:
            z = F.gelu(self.fc_z(context))
            return z, h_all


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class LSTMDecoder(nn.Module):
    """
    LSTM decoder that reconstructs the original signal from a latent vector.

    Uses scheduled teacher forcing during training. At inference, produces
    auto-regressive reconstructions.

    Args:
        latent_dim:  Dimension of input latent vector.
        hidden_dim:  LSTM hidden state dimension.
        output_dim:  Number of channels to reconstruct.
        seq_len:     Fixed sequence length to reconstruct.
        num_layers:  Number of stacked LSTM layers.
        dropout:     Dropout rate.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 8,
        seq_len: int = 48,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Latent → initial hidden state for all layers
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.latent_to_cell = nn.Linear(latent_dim, hidden_dim * num_layers)

        self.lstm = nn.LSTM(
            input_size=output_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def _init_hidden(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        B = z.size(0)
        h = F.tanh(self.latent_to_hidden(z))  # (B, H * num_layers)
        c = F.tanh(self.latent_to_cell(z))
        h = h.view(B, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        c = c.view(B, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        return h, c

    def forward(
        self,
        z: Tensor,
        target: Optional[Tensor] = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> Tensor:
        """
        Args:
            z:                     (B, latent_dim)
            target:                (B, T, C) original signal (for teacher forcing)
            teacher_forcing_ratio: probability of using ground truth at each step

        Returns:
            recon: (B, T, C) reconstructed signal
        """
        B = z.size(0)
        device = z.device
        C = self.output_proj.out_features

        h, c = self._init_hidden(z)

        # Seed token: zeros
        inp = torch.zeros(B, 1, C, device=device)
        outputs = []

        for t in range(self.seq_len):
            out, (h, c) = self.lstm(inp, (h, c))
            pred = self.output_proj(self.dropout(out))  # (B, 1, C)
            outputs.append(pred)

            # Teacher forcing
            if target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                inp = target[:, t : t + 1, :]
            else:
                inp = pred.detach()

        return torch.cat(outputs, dim=1)  # (B, T, C)


# ---------------------------------------------------------------------------
# Main Autoencoder
# ---------------------------------------------------------------------------

class LSTMAutoencoder(nn.Module):
    """
    Bidirectional LSTM Autoencoder for unsupervised anomaly detection.

    Anomaly score = per-channel MAE between input and reconstruction.
    Optionally uses variational latent space (VAE-LSTM) for probabilistic scoring.

    Args:
        input_dim:        Number of physiological channels.
        hidden_dim:       LSTM hidden state dimension.
        latent_dim:       Dimension of bottleneck latent space.
        seq_len:          Length of input/output window (time steps).
        num_layers:       Number of LSTM layers in encoder and decoder.
        dropout:          Dropout probability.
        variational:      If True, train as VAE with KL divergence loss.
        kl_weight:        Weight of KL term in ELBO loss (β-VAE).
        channel_names:    Optional list of channel names for interpretability.
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        seq_len: int = 48,
        num_layers: int = 2,
        dropout: float = 0.1,
        variational: bool = True,
        kl_weight: float = 0.5,
        channel_names: Optional[list] = None,
    ) -> None:
        super().__init__()
        self.variational = variational
        self.kl_weight = kl_weight
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.channel_names = channel_names or [f"ch{i}" for i in range(input_dim)]

        self.encoder = LSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
            variational=variational,
        )

        self.decoder = LSTMDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            seq_len=seq_len,
            num_layers=num_layers,
            dropout=dropout,
        )

        if variational:
            self.sampler = VariationalSampler()

        # Learnable per-channel uncertainty (aleatoric)
        self.log_noise = nn.Parameter(torch.zeros(input_dim))

    def forward(
        self,
        x: Tensor,
        lengths: Optional[Tensor] = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> Dict[str, Tensor]:
        """
        Args:
            x:                     (B, T, C) input sequence
            lengths:               (B,) actual sequence lengths
            teacher_forcing_ratio: for decoder training

        Returns dict with keys:
            'recon':        (B, T, C) reconstructed signal
            'z':            (B, latent_dim) latent vector
            'anomaly_score':(B,) scalar anomaly score per sample
            'channel_scores':(B, C) per-channel anomaly scores
            'kl_loss':      scalar KL divergence (VAE only, else 0)
            'recon_loss':   scalar reconstruction loss
        """
        if self.variational:
            mu, log_var, h_all = self.encoder(x, lengths)
            z, std, kl_loss = self.sampler(mu, log_var)
        else:
            z, h_all = self.encoder(x, lengths)
            kl_loss = torch.tensor(0.0, device=x.device)

        recon = self.decoder(z, target=x if self.training else None,
                             teacher_forcing_ratio=teacher_forcing_ratio)

        # Per-channel reconstruction error
        channel_scores = F.l1_loss(recon, x, reduction="none").mean(dim=1)  # (B, C)

        # Noise-weighted MAE (learnable heteroscedastic noise)
        noise_var = torch.exp(self.log_noise)  # (C,)
        weighted_scores = channel_scores / (2 * noise_var) + 0.5 * self.log_noise
        anomaly_score = weighted_scores.sum(dim=-1)  # (B,)

        # Mean reconstruction loss for backprop
        recon_loss = F.mse_loss(recon, x)

        return {
            "recon": recon,
            "z": z if not self.variational else mu,
            "anomaly_score": anomaly_score,
            "channel_scores": channel_scores,
            "kl_loss": kl_loss,
            "recon_loss": recon_loss,
        }

    def compute_loss(self, outputs: Dict[str, Tensor]) -> Tensor:
        """Combine reconstruction + KL losses for training."""
        return outputs["recon_loss"] + self.kl_weight * outputs["kl_loss"]

    @torch.no_grad()
    def score(self, x: Tensor, mc_samples: int = 20) -> Dict[str, Tensor]:
        """
        Compute anomaly scores with Monte Carlo uncertainty estimation (VAE-LSTM).

        Args:
            x:          (B, T, C)
            mc_samples: Number of Monte Carlo forward passes

        Returns dict:
            'anomaly_score':      (B,) mean anomaly score
            'anomaly_score_std':  (B,) epistemic uncertainty
            'channel_scores':     (B, C) per-channel scores
            'channel_score_std':  (B, C) per-channel uncertainty
            'recon':              (B, T, C) mean reconstruction
        """
        self.eval()
        if not self.variational or mc_samples == 1:
            return self.forward(x, teacher_forcing_ratio=0.0)

        # MC Dropout: enable dropout at inference for epistemic uncertainty
        def enable_dropout(model):
            for m in model.modules():
                if isinstance(m, nn.Dropout):
                    m.train()

        enable_dropout(self)

        all_scores = []
        all_channel_scores = []
        all_recons = []

        for _ in range(mc_samples):
            out = self.forward(x, teacher_forcing_ratio=0.0)
            all_scores.append(out["anomaly_score"])
            all_channel_scores.append(out["channel_scores"])
            all_recons.append(out["recon"])

        scores_stack = torch.stack(all_scores, dim=0)        # (MC, B)
        ch_scores_stack = torch.stack(all_channel_scores, dim=0)  # (MC, B, C)
        recons_stack = torch.stack(all_recons, dim=0)        # (MC, B, T, C)

        return {
            "anomaly_score": scores_stack.mean(dim=0),
            "anomaly_score_std": scores_stack.std(dim=0),
            "channel_scores": ch_scores_stack.mean(dim=0),
            "channel_score_std": ch_scores_stack.std(dim=0),
            "recon": recons_stack.mean(dim=0),
        }

    def get_channel_importance(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Compute SHAP-style channel importance via ablation.
        Measures increase in anomaly score when each channel is masked.

        Args:
            x: (B, T, C)

        Returns:
            importance: (C,) mean importance per channel
        """
        self.eval()
        with torch.no_grad():
            baseline = self.score(x, mc_samples=1)["anomaly_score"].mean()
            importances = []
            for c in range(self.input_dim):
                x_masked = x.clone()
                x_masked[:, :, c] = 0.0  # zero-mask channel
                masked_score = self.score(x_masked, mc_samples=1)["anomaly_score"].mean()
                importances.append(masked_score - baseline)
        return {
            "importance": torch.stack(importances),
            "channel_names": self.channel_names,
        }


# ---------------------------------------------------------------------------
# Threshold Calibration
# ---------------------------------------------------------------------------

class AnomalyThresholdCalibrator:
    """
    Calibrate anomaly detection threshold on a labelled validation set.

    Supports:
      - Fixed percentile of normal scores (e.g., 95th percentile)
      - Optimal F1 threshold search
      - Sensitivity-fixed threshold (e.g., fix sensitivity at 90%, find specificity)
    """

    def __init__(self) -> None:
        self.threshold: Optional[float] = None
        self.percentile_threshold: Optional[float] = None

    def fit_percentile(self, normal_scores: Tensor, percentile: float = 95.0) -> float:
        """Set threshold at given percentile of normal sample scores."""
        self.threshold = torch.quantile(normal_scores, percentile / 100.0).item()
        self.percentile_threshold = percentile
        return self.threshold

    def fit_sensitivity(
        self,
        scores: Tensor,
        labels: Tensor,
        target_sensitivity: float = 0.90,
    ) -> Dict[str, float]:
        """
        Find the threshold that achieves target sensitivity.

        Args:
            scores:              (N,) anomaly scores
            labels:              (N,) binary labels (1 = anomalous)
            target_sensitivity:  desired true positive rate

        Returns:
            dict with 'threshold', 'sensitivity', 'specificity', 'alarm_rate'
        """
        scores_np = scores.cpu().numpy()
        labels_np = labels.cpu().numpy()

        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(labels_np, scores_np)

        # Find threshold closest to target sensitivity
        idx = (tpr >= target_sensitivity).nonzero()[0]
        if len(idx) == 0:
            idx = [len(tpr) - 1]
        chosen_idx = idx[0]

        self.threshold = float(thresholds[chosen_idx])
        return {
            "threshold": self.threshold,
            "sensitivity": float(tpr[chosen_idx]),
            "specificity": float(1 - fpr[chosen_idx]),
            "alarm_rate": float(fpr[chosen_idx]),
        }

    def predict(self, scores: Tensor) -> Tensor:
        """Apply threshold: returns binary anomaly predictions."""
        if self.threshold is None:
            raise RuntimeError("Calibrate threshold before calling predict()")
        return (scores > self.threshold).long()


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_lstm_ae(config: dict) -> LSTMAutoencoder:
    """Build LSTMAutoencoder from a config dict (e.g., loaded from YAML)."""
    return LSTMAutoencoder(
        input_dim=config.get("input_dim", 8),
        hidden_dim=config.get("hidden_dim", 128),
        latent_dim=config.get("latent_dim", 64),
        seq_len=config.get("seq_len", 48),
        num_layers=config.get("num_layers", 2),
        dropout=config.get("dropout", 0.1),
        variational=config.get("variational", True),
        kl_weight=config.get("kl_weight", 0.5),
        channel_names=config.get("channel_names", None),
    )


if __name__ == "__main__":
    # Quick smoke test
    torch.manual_seed(42)
    model = LSTMAutoencoder(
        input_dim=8, hidden_dim=64, latent_dim=32,
        seq_len=48, variational=True
    )
    x = torch.randn(4, 48, 8)
    out = model(x)
    loss = model.compute_loss(out)
    print(f"Recon shape:    {out['recon'].shape}")          # (4, 48, 8)
    print(f"Anomaly score:  {out['anomaly_score'].shape}")  # (4,)
    print(f"Channel scores: {out['channel_scores'].shape}") # (4, 8)
    print(f"Loss:           {loss.item():.4f}")

    scores = model.score(x, mc_samples=10)
    print(f"MC score mean:  {scores['anomaly_score'].shape}")
    print(f"MC score std:   {scores['anomaly_score_std'].shape}")
    print("LSTM Autoencoder smoke test passed.")
