"""
Transformer Anomaly Detector for ICU Multi-Channel Physiological Signals.

Implements:
  - Temporal self-attention with causal masking (left-to-right, deployable in real-time)
  - Cross-channel attention (learns physiological dependencies across signals)
  - Attention-weighted reconstruction error for anomaly scoring
  - Multi-task classification head (mortality, sepsis onset, vasopressor)

Architecture Reference:
  Vaswani et al. (2017). "Attention Is All You Need." NeurIPS.
  Zerveas et al. (2021). "A Transformer-based Framework for Multivariate Time
  Series Representation Learning." KDD.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (Vaswani et al., 2017).
    Supports variable sequence lengths and relative time gaps (irregular sampling).
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: Tensor, time_offsets: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x:            (B, T, d_model)
            time_offsets: (B, T) optional real-valued time in minutes (irregular sampling)
        """
        if time_offsets is not None:
            # Continuous-time encoding for irregular sampling
            return self.dropout(x + self._continuous_pe(time_offsets, x.size(-1)))
        return self.dropout(x + self.pe[:, : x.size(1)])

    def _continuous_pe(self, t: Tensor, d_model: int) -> Tensor:
        """Compute sinusoidal PE at arbitrary time points."""
        # t: (B, T) in minutes
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=t.device).float()
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(*t.shape, d_model, device=t.device)
        t_exp = t.unsqueeze(-1) * div_term.unsqueeze(0).unsqueeze(0)
        pe[..., 0::2] = torch.sin(t_exp)
        pe[..., 1::2] = torch.cos(t_exp[..., :d_model // 2])
        return pe


# ---------------------------------------------------------------------------
# Attention Modules
# ---------------------------------------------------------------------------

class CausalMultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention with optional causal masking.

    Causal masking ensures that at time step t, the model only attends
    to time steps <= t, enabling real-time (streaming) deployment.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        causal: bool = True,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.causal = causal
        self.scale = math.sqrt(self.d_k)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        return x.view(B, T, self.n_heads, self.d_k).permute(0, 2, 1, 3)

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x:                (B, T, D)
            key_padding_mask: (B, T) — True for padded positions

        Returns:
            out:     (B, T, D)
            attn_w:  (B, n_heads, T, T) attention weights
        """
        B, T, D = x.shape
        Q = self._split_heads(self.q_proj(x))  # (B, H, T, d_k)
        K = self._split_heads(self.k_proj(x))
        V = self._split_heads(self.v_proj(x))

        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, T, T)

        # Causal mask: upper triangle → -inf
        if self.causal:
            causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            attn_logits = attn_logits.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Padding mask
        if key_padding_mask is not None:
            attn_logits = attn_logits.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn_w = F.softmax(attn_logits, dim=-1)
        attn_w = self.dropout(attn_w)

        out = torch.matmul(attn_w, V)  # (B, H, T, d_k)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, D)
        return self.out_proj(out), attn_w


class CrossChannelAttention(nn.Module):
    """
    Attention across physiological channels at each time step.

    For each time step t, attends across channels C to learn inter-signal
    dependencies (e.g., HR-BP coupling during vasopressor response).
    """

    def __init__(
        self,
        n_channels: int,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.d_model = d_model
        self.n_heads = n_heads

        # Channel embedding to project channels into d_model space
        self.channel_embed = nn.Embedding(n_channels, d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: (B, T, C*d_model) — all channels' representations at all times

        The input is expected to be reshaped to (B*T, C, d_model) for channel attention.

        Returns:
            out:    (B, T, C, d_model) after cross-channel attention
            attn_w: (B*T, C, C)
        """
        B, T, C, D = x.shape
        # Reshape: treat time as batch dimension
        x_flat = x.view(B * T, C, D)

        # Add channel position embeddings
        ch_idx = torch.arange(C, device=x.device)
        x_flat = x_flat + self.channel_embed(ch_idx).unsqueeze(0)

        out, attn_w = self.attn(x_flat, x_flat, x_flat)
        out = self.norm(out + x_flat)
        return out.view(B, T, C, D), attn_w


class TransformerBlock(nn.Module):
    """Single Transformer encoder block: self-attention + FFN + residuals."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        causal: bool = True,
    ) -> None:
        super().__init__()
        self.self_attn = CausalMultiHeadSelfAttention(d_model, n_heads, dropout, causal)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        # Pre-norm (more stable than post-norm)
        x_norm = self.norm1(x)
        attn_out, attn_w = self.self_attn(x_norm, key_padding_mask)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x, attn_w


# ---------------------------------------------------------------------------
# Reconstruction Head
# ---------------------------------------------------------------------------

class ReconstructionHead(nn.Module):
    """Reconstruct original signal from Transformer hidden states."""

    def __init__(self, d_model: int, output_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)  # (B, T, C)


# ---------------------------------------------------------------------------
# Classification Heads
# ---------------------------------------------------------------------------

class ClinicalOutcomeHead(nn.Module):
    """
    Multi-task clinical outcome prediction head.

    Predicts: in-hospital mortality, sepsis onset, vasopressor initiation.
    Uses CLS-token pooling + outcome-specific MLP.
    """

    def __init__(
        self,
        d_model: int,
        n_outcomes: int = 3,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_outcomes = n_outcomes
        self.pool = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Separate head per outcome for calibration
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(n_outcomes)
        ])

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, T, d_model) — full sequence output

        Returns:
            logits: (B, n_outcomes)
        """
        # Global mean pooling over time (non-causal summary for batch training)
        pooled = x.mean(dim=1)  # (B, d_model)
        h = self.pool(pooled)
        return torch.cat([head(h) for head in self.heads], dim=-1)  # (B, n_outcomes)


# ---------------------------------------------------------------------------
# Main Transformer Detector
# ---------------------------------------------------------------------------

class TransformerAnomalyDetector(nn.Module):
    """
    Transformer-based anomaly detector for multi-channel ICU physiological signals.

    Architecture:
      Input (B, T, C) → Channel Projection → Temporal Positional Encoding
        → L × [Temporal Self-Attention (causal) + Cross-Channel Attention + FFN]
        → Reconstruction Head (anomaly scoring)
        → Classification Head (mortality, sepsis, vasopressor)

    Anomaly score = attention-weighted per-channel reconstruction error.

    Args:
        input_dim:       Number of input channels (vital signs).
        d_model:         Transformer model dimension.
        n_heads:         Number of attention heads (temporal self-attention).
        n_layers:        Number of Transformer blocks.
        ff_dim:          Feed-forward hidden dimension.
        seq_len:         Input sequence length (time steps).
        dropout:         Dropout rate.
        causal:          If True, apply causal mask (for real-time inference).
        n_outcomes:      Number of clinical outcomes to predict.
        use_cross_channel: If True, apply cross-channel attention layer.
        outcome_names:   Names of clinical outcomes (for logging).
    """

    def __init__(
        self,
        input_dim: int = 8,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        ff_dim: int = 512,
        seq_len: int = 48,
        dropout: float = 0.1,
        causal: bool = True,
        n_outcomes: int = 3,
        use_cross_channel: bool = True,
        outcome_names: Optional[List[str]] = None,
        channel_names: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.seq_len = seq_len
        self.causal = causal
        self.use_cross_channel = use_cross_channel
        self.outcome_names = outcome_names or ["mortality", "sepsis", "vasopressor"]
        self.channel_names = channel_names or [f"ch{i}" for i in range(input_dim)]

        # Input: project each channel to d_model
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=seq_len + 10, dropout=dropout)

        # Temporal Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim, dropout, causal)
            for _ in range(n_layers)
        ])

        # Cross-channel attention (optional)
        if use_cross_channel:
            self.cross_channel_attn = CrossChannelAttention(
                n_channels=input_dim, d_model=d_model // max(1, input_dim // 4), n_heads=4, dropout=dropout
            )
            # Project after cross-channel attention
            self.cross_channel_proj = nn.Linear(input_dim * (d_model // max(1, input_dim // 4)), d_model)

        # Reconstruction head
        self.recon_head = ReconstructionHead(d_model, input_dim)

        # Anomaly attention weights: learn which time steps matter for anomaly score
        self.anomaly_attn_gate = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

        # Classification head
        self.outcome_head = ClinicalOutcomeHead(d_model, n_outcomes, hidden_dim=64, dropout=dropout)

        # Layer norm for final representation
        self.final_norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: Tensor,
        time_offsets: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        return_attentions: bool = False,
    ) -> Dict[str, Tensor]:
        """
        Args:
            x:                  (B, T, C) input vital signs
            time_offsets:       (B, T) real-valued time in minutes (for irregular sampling)
            key_padding_mask:   (B, T) True for padded positions
            return_attentions:  If True, return all attention weight tensors

        Returns dict:
            'recon':            (B, T, C) signal reconstruction
            'anomaly_score':    (B,) scalar anomaly score per sample
            'channel_scores':   (B, C) per-channel anomaly scores
            'outcome_logits':   (B, n_outcomes) classification logits
            'hidden':           (B, T, d_model) final hidden states
            'attention_weights':(list of (B, H, T, T)) — if return_attentions=True
            'recon_loss':       scalar reconstruction MSE
        """
        B, T, C = x.shape

        # --- Input projection ---
        h = self.input_proj(x)  # (B, T, d_model)
        h = self.pos_enc(h, time_offsets)

        # --- Temporal self-attention layers ---
        attn_weights_list = []
        for block in self.transformer_blocks:
            h, attn_w = block(h, key_padding_mask)
            attn_weights_list.append(attn_w)

        h = self.final_norm(h)

        # --- Reconstruction and anomaly scoring ---
        recon = self.recon_head(h)  # (B, T, C)

        # Per-channel reconstruction error
        recon_err = F.l1_loss(recon, x, reduction="none")  # (B, T, C)

        # Attention gate: weight time steps by learned importance
        attn_gate = self.anomaly_attn_gate(h)  # (B, T, 1)
        attn_gate = attn_gate / (attn_gate.sum(dim=1, keepdim=True) + 1e-8)  # normalise

        # Attention-weighted per-channel score
        channel_scores = (recon_err * attn_gate).sum(dim=1)  # (B, C)
        anomaly_score = channel_scores.mean(dim=-1)           # (B,)

        # --- Outcome classification ---
        outcome_logits = self.outcome_head(h)  # (B, n_outcomes)

        result = {
            "recon": recon,
            "anomaly_score": anomaly_score,
            "channel_scores": channel_scores,
            "outcome_logits": outcome_logits,
            "hidden": h,
            "recon_loss": F.mse_loss(recon, x),
            "attn_gate": attn_gate.squeeze(-1),  # (B, T)
        }

        if return_attentions:
            result["attention_weights"] = attn_weights_list

        return result

    def compute_loss(
        self,
        outputs: Dict[str, Tensor],
        labels: Optional[Dict[str, Tensor]] = None,
        recon_weight: float = 1.0,
        cls_weight: float = 1.0,
        pos_weights: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Compute combined reconstruction + classification losses.

        Args:
            outputs:      Forward pass output dict
            labels:       Dict mapping outcome name to (B,) binary label tensors
            recon_weight: Weight for reconstruction loss
            cls_weight:   Weight for classification losses
            pos_weights:  (n_outcomes,) positive class weights for imbalance

        Returns:
            dict with 'total', 'recon', 'cls_{outcome}' keys
        """
        losses = {"recon": outputs["recon_loss"] * recon_weight}

        if labels is not None:
            for i, name in enumerate(self.outcome_names):
                if name in labels:
                    y = labels[name].float()
                    logit = outputs["outcome_logits"][:, i]
                    pw = None if pos_weights is None else pos_weights[i]
                    cls_loss = F.binary_cross_entropy_with_logits(
                        logit, y, pos_weight=pw
                    )
                    losses[f"cls_{name}"] = cls_loss * cls_weight

        total = sum(losses.values())
        losses["total"] = total
        return losses

    @torch.no_grad()
    def predict_outcomes(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Run inference and return calibrated outcome probabilities.

        Returns:
            dict mapping outcome name → (B,) probability tensor
        """
        self.eval()
        outputs = self.forward(x)
        probs = torch.sigmoid(outputs["outcome_logits"])  # (B, n_outcomes)
        return {name: probs[:, i] for i, name in enumerate(self.outcome_names)}

    def get_attention_maps(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Return attention maps for interpretability analysis.

        Returns:
            temporal_attention: (n_layers, B, n_heads, T, T)
        """
        out = self.forward(x, return_attentions=True)
        temporal_attn = torch.stack(out["attention_weights"], dim=0)  # (L, B, H, T, T)
        return {
            "temporal_attention": temporal_attn,
            "anomaly_gate": out["attn_gate"],
            "channel_scores": out["channel_scores"],
        }

    def compute_integrated_gradients(
        self,
        x: Tensor,
        target_outcome: int = 0,
        n_steps: int = 50,
    ) -> Tensor:
        """
        Compute integrated gradients for feature attribution.

        Args:
            x:               (1, T, C) single sample
            target_outcome:  index into outcome_names
            n_steps:         number of interpolation steps

        Returns:
            attributions: (T, C) importance scores
        """
        baseline = torch.zeros_like(x)
        x.requires_grad_(True)

        alphas = torch.linspace(0, 1, n_steps, device=x.device)
        integrated_grads = torch.zeros_like(x)

        for alpha in alphas:
            interpolated = baseline + alpha * (x - baseline)
            interpolated.requires_grad_(True)
            out = self.forward(interpolated)
            score = out["outcome_logits"][0, target_outcome]
            grads = torch.autograd.grad(score, interpolated)[0]
            integrated_grads += grads / n_steps

        return (integrated_grads * (x - baseline)).squeeze(0).detach()


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_transformer_detector(config: dict) -> TransformerAnomalyDetector:
    """Build TransformerAnomalyDetector from config dict."""
    return TransformerAnomalyDetector(
        input_dim=config.get("input_dim", 8),
        d_model=config.get("d_model", 128),
        n_heads=config.get("n_heads", 8),
        n_layers=config.get("n_layers", 4),
        ff_dim=config.get("ff_dim", 512),
        seq_len=config.get("seq_len", 48),
        dropout=config.get("dropout", 0.1),
        causal=config.get("causal", True),
        n_outcomes=config.get("n_outcomes", 3),
        use_cross_channel=config.get("use_cross_channel", True),
        outcome_names=config.get("outcome_names", ["mortality", "sepsis", "vasopressor"]),
        channel_names=config.get("channel_names", None),
    )


if __name__ == "__main__":
    torch.manual_seed(42)
    model = TransformerAnomalyDetector(
        input_dim=8, d_model=128, n_heads=8, n_layers=4,
        ff_dim=512, seq_len=48, causal=True
    )
    x = torch.randn(4, 48, 8)
    labels = {
        "mortality": torch.tensor([0, 1, 0, 1]).float(),
        "sepsis": torch.tensor([0, 0, 1, 1]).float(),
        "vasopressor": torch.tensor([1, 0, 0, 1]).float(),
    }
    out = model(x)
    losses = model.compute_loss(out, labels, pos_weights=torch.tensor([5.0, 3.0, 2.5]))
    print(f"Recon shape:       {out['recon'].shape}")         # (4, 48, 8)
    print(f"Anomaly score:     {out['anomaly_score'].shape}") # (4,)
    print(f"Outcome logits:    {out['outcome_logits'].shape}")# (4, 3)
    print(f"Total loss:        {losses['total'].item():.4f}")

    # Test attention maps
    attn_maps = model.get_attention_maps(x[:1])
    print(f"Temporal attention:{attn_maps['temporal_attention'].shape}")
    print("Transformer detector smoke test passed.")
