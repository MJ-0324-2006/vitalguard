"""
Temporal Convolutional Network (TCN) for Vital Sign Forecasting.

Implements:
  - Dilated causal convolutions with residual connections
  - Multi-horizon prediction (15, 30, 60 minutes ahead)
  - Prediction error as anomaly signal
  - Channel-specific anomaly scores
  - Cross-channel fusion
  - Monte Carlo Dropout uncertainty estimation
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ============================================================================
# CAUSAL CONVOLUTION
# ============================================================================

class CausalConv1d(nn.Module):
    """
    1D causal convolution.

    Padding is added only to the left, so output at time t
    never uses future information.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()

        self.padding = (kernel_size - 1) * dilation

        # IMPORTANT:
        # The actual Conv1d is stored as self.conv.
        # Therefore weight_norm must be applied to self.conv,
        # NOT to this wrapper module.
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )

        # Apply weight normalization to the real Conv1d.
        self.conv = nn.utils.weight_norm(self.conv)

    def forward(self, x: Tensor) -> Tensor:
        """
        x shape:
            (B, C, T)
        """

        x = F.pad(x, (self.padding, 0))

        return self.conv(x)


# ============================================================================
# DILATED RESIDUAL BLOCK
# ============================================================================

class DilatedResidualBlock(nn.Module):
    """
    Two dilated causal convolutions with residual connection.
    """

    def __init__(
        self,
        n_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.conv1 = CausalConv1d(
            n_channels,
            n_channels,
            kernel_size,
            dilation,
        )

        self.conv2 = CausalConv1d(
            n_channels,
            n_channels,
            kernel_size,
            dilation,
        )

        # GroupNorm is stable for small batches.
        n_groups = min(8, n_channels)

        # Make sure number of channels is divisible by groups.
        while n_groups > 1 and n_channels % n_groups != 0:
            n_groups -= 1

        self.norm1 = nn.GroupNorm(
            num_groups=n_groups,
            num_channels=n_channels,
        )

        self.norm2 = nn.GroupNorm(
            num_groups=n_groups,
            num_channels=n_channels,
        )

        self.activation = nn.GELU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        x:
            (B, C, T)
        """

        residual = x

        h = self.conv1(x)
        h = self.norm1(h)
        h = self.activation(h)
        h = self.dropout(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.activation(h)
        h = self.dropout(h)

        return h + residual


# ============================================================================
# TCN BACKBONE
# ============================================================================

class TCNBackbone(nn.Module):
    """
    Stack of dilated residual blocks.

    Dilation:
        1, 2, 4, 8, 16, ...

    For kernel_size=3 and 8 levels:
        receptive field = 1021 time steps
    """

    def __init__(
        self,
        n_hidden: int,
        kernel_size: int = 3,
        n_levels: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.n_hidden = n_hidden
        self.kernel_size = kernel_size
        self.n_levels = n_levels

        self.blocks = nn.ModuleList()

        for i in range(n_levels):

            dilation = 2 ** i

            self.blocks.append(
                DilatedResidualBlock(
                    n_channels=n_hidden,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )

    def forward(self, x: Tensor) -> Tensor:

        for block in self.blocks:
            x = block(x)

        return x

    @property
    def receptive_field(self) -> int:
        """
        Calculate total receptive field.
        """

        rf = 1

        for i in range(self.n_levels):

            dilation = 2 ** i

            # Two convolutions per residual block.
            rf += 2 * (self.kernel_size - 1) * dilation

        return rf


# ============================================================================
# MULTI-HORIZON HEAD
# ============================================================================

class MultiHorizonHead(nn.Module):
    """
    Predict values at multiple future horizons.

    Each horizon produces:

        mean
        log_variance
    """

    def __init__(
        self,
        n_hidden: int,
        output_dim: int,
        horizons: List[int],
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.horizons = horizons
        self.output_dim = output_dim

        hidden_half = max(1, n_hidden // 2)

        self.heads = nn.ModuleDict()

        for horizon in horizons:

            self.heads[f"h{horizon}"] = nn.Sequential(
                nn.Linear(
                    n_hidden,
                    hidden_half,
                ),

                nn.GELU(),

                nn.Dropout(dropout),

                nn.Linear(
                    hidden_half,
                    output_dim * 2,
                ),
            )

    def forward(self, h: Tensor) -> Dict[str, Dict[str, Tensor]]:
        """
        h:
            (B, hidden)

        Returns:
            {
                "h15": {
                    "mean": ...,
                    "log_var": ...
                },
                ...
            }
        """

        results = {}

        for horizon in self.horizons:

            output = self.heads[f"h{horizon}"](h)

            mean, log_var = output.chunk(2, dim=-1)

            results[f"h{horizon}"] = {
                "mean": mean,
                "log_var": log_var,
            }

        return results


# ============================================================================
# MAIN TCN MODEL
# ============================================================================

class TCNPredictor(nn.Module):
    """
    Temporal Convolutional Network for ICU vital-sign forecasting.
    """

    def __init__(
        self,
        input_dim: int = 8,
        n_hidden: int = 128,
        kernel_size: int = 3,
        n_levels: int = 8,
        horizons: Optional[List[int]] = None,
        dropout: float = 0.1,
        channel_names: Optional[List[str]] = None,
    ) -> None:

        super().__init__()

        self.input_dim = input_dim
        self.n_hidden = n_hidden

        self.horizons = horizons or [15, 30, 60]

        self.channel_names = (
            channel_names
            if channel_names is not None
            else [f"ch{i}" for i in range(input_dim)]
        )

        # ------------------------------------------------------------------
        # Input projection
        # ------------------------------------------------------------------

        self.input_proj = nn.Conv1d(
            input_dim,
            n_hidden,
            kernel_size=1,
        )

        # ------------------------------------------------------------------
        # TCN
        # ------------------------------------------------------------------

        self.backbone = TCNBackbone(
            n_hidden=n_hidden,
            kernel_size=kernel_size,
            n_levels=n_levels,
            dropout=dropout,
        )

        # ------------------------------------------------------------------
        # Prediction heads
        # ------------------------------------------------------------------

        self.prediction_head = MultiHorizonHead(
            n_hidden=n_hidden,
            output_dim=input_dim,
            horizons=self.horizons,
            dropout=dropout,
        )

        # ------------------------------------------------------------------
        # Anomaly head
        # ------------------------------------------------------------------

        self.anomaly_head = nn.Sequential(
            nn.Linear(
                n_hidden,
                max(1, n_hidden // 2),
            ),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(
                max(1, n_hidden // 2),
                input_dim,
            ),
        )

        # ------------------------------------------------------------------
        # Cross-channel fusion
        # ------------------------------------------------------------------

        self.cross_channel_fusion = nn.Sequential(
            nn.Linear(
                n_hidden,
                n_hidden,
            ),

            nn.GELU(),

            nn.Linear(
                n_hidden,
                n_hidden,
            ),
        )

        self.fusion_gate = nn.Sequential(
            nn.Linear(
                n_hidden * 2,
                n_hidden,
            ),

            nn.Sigmoid(),
        )

    # ========================================================================
    # FORWARD
    # ========================================================================

    def forward(
        self,
        x: Tensor,
        return_all_timesteps: bool = False,
    ) -> Dict[str, Tensor]:

        # Expected:
        #
        # x = (B, T, C)

        if x.ndim != 3:
            raise ValueError(
                f"Expected input shape (B, T, C), got {tuple(x.shape)}"
            )

        B, T, C = x.shape

        if C != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} input channels, "
                f"got {C}"
            )

        # ------------------------------------------------------------------
        # Convert:
        #
        # (B, T, C)
        #
        # -> (B, C, T)
        # ------------------------------------------------------------------

        x_t = x.permute(0, 2, 1)

        # ------------------------------------------------------------------
        # Input projection
        # ------------------------------------------------------------------

        h = self.input_proj(x_t)

        # ------------------------------------------------------------------
        # TCN
        # ------------------------------------------------------------------

        h = self.backbone(h)

        # ------------------------------------------------------------------
        # Convert back:
        #
        # (B, hidden, T)
        #
        # -> (B, T, hidden)
        # ------------------------------------------------------------------

        h_time = h.permute(0, 2, 1)

        # ------------------------------------------------------------------
        # Cross-channel fusion
        # ------------------------------------------------------------------

        h_fused = self.cross_channel_fusion(h_time)

        gate_input = torch.cat(
            [
                h_time,
                h_fused,
            ],
            dim=-1,
        )

        gate = self.fusion_gate(gate_input)

        h_combined = (
            gate * h_fused
            + (1.0 - gate) * h_time
        )

        # ------------------------------------------------------------------
        # Last timestep
        # ------------------------------------------------------------------

        h_last = h_combined[:, -1, :]

        # ------------------------------------------------------------------
        # Forecast predictions
        # ------------------------------------------------------------------

        predictions = self.prediction_head(h_last)

        # ------------------------------------------------------------------
        # Anomaly score
        # ------------------------------------------------------------------

        channel_anomaly = torch.sigmoid(
            self.anomaly_head(h_last)
        )

        anomaly_score = channel_anomaly.mean(dim=-1)

        # ------------------------------------------------------------------
        # Output
        # ------------------------------------------------------------------

        return {
            "predictions": predictions,

            "hidden": (
                h_combined
                if return_all_timesteps
                else h_last
            ),

            "channel_anomaly": channel_anomaly,

            "anomaly_score": anomaly_score,
        }

    # ========================================================================
    # FORECAST LOSS
    # ========================================================================

    def compute_forecast_loss(
        self,
        outputs: Dict,
        targets: Dict,
        loss_type: str = "nll",
    ) -> Dict[str, Tensor]:
        """
        Compute forecasting loss.

        Supports target formats such as:

            {
                "h15": tensor,
                "h30": tensor,
                "h60": tensor
            }

        and nested formats such as:

            {
                "targets": {
                    "h15": tensor,
                    ...
                }
            }

        Also supports target keys such as:

            "15"
            "30"
            "60"

        if necessary.
        """

        predictions = outputs.get("predictions", {})

        # ------------------------------------------------------------------
        # Some datasets/trainers may wrap the horizon targets.
        # ------------------------------------------------------------------

        if isinstance(targets, dict):

            if "targets" in targets and isinstance(
                targets["targets"],
                dict,
            ):
                targets = targets["targets"]

            elif "forecast" in targets and isinstance(
                targets["forecast"],
                dict,
            ):
                targets = targets["forecast"]

        losses = {}

        # ------------------------------------------------------------------
        # Find each horizon
        # ------------------------------------------------------------------

        for horizon in self.horizons:

            pred_key = f"h{horizon}"

            target = None

            possible_keys = [
                f"h{horizon}",
                str(horizon),
                horizon,
            ]

            for key in possible_keys:

                if key in targets:
                    target = targets[key]
                    break

            if target is None:
                continue

            pred = predictions.get(pred_key)

            if pred is None:
                continue

            # --------------------------------------------------------------
            # Target may itself be a dict.
            # --------------------------------------------------------------

            if isinstance(target, dict):

                if "target" in target:
                    target = target["target"]

                elif "value" in target:
                    target = target["value"]

                elif "mean" in target:
                    target = target["mean"]

            # --------------------------------------------------------------
            # Ensure tensor
            # --------------------------------------------------------------

            if not torch.is_tensor(target):

                target = torch.as_tensor(
                    target,
                    dtype=pred["mean"].dtype,
                    device=pred["mean"].device,
                )

            else:

                target = target.to(
                    device=pred["mean"].device,
                    dtype=pred["mean"].dtype,
                )

            # --------------------------------------------------------------
            # Match dimensions
            # --------------------------------------------------------------

            if target.ndim == 3:

                # Possible shape:
                #
                # (B, 1, C)
                #
                # or
                #
                # (B, T, C)
                #
                # Use the last timestep.

                target = target[:, -1, :]

            elif target.ndim == 1:

                target = target.unsqueeze(-1)

            # --------------------------------------------------------------
            # Match channel count
            # --------------------------------------------------------------

            if target.shape[-1] != pred["mean"].shape[-1]:

                raise ValueError(
                    f"Horizon {horizon}: target has "
                    f"{target.shape[-1]} channels but prediction has "
                    f"{pred['mean'].shape[-1]} channels."
                )

            # --------------------------------------------------------------
            # Loss
            # --------------------------------------------------------------

            if loss_type.lower() == "nll":

                mu = pred["mean"]

                log_var = pred["log_var"].clamp(
                    min=-6.0,
                    max=6.0,
                )

                variance = torch.exp(log_var)

                nll = 0.5 * (
                    log_var
                    + (target - mu).pow(2) / variance
                )

                loss = nll.mean()

            elif loss_type.lower() == "mae":

                loss = F.l1_loss(
                    pred["mean"],
                    target,
                )

            else:

                loss = F.mse_loss(
                    pred["mean"],
                    target,
                )

            losses[pred_key] = loss

        # ------------------------------------------------------------------
        # IMPORTANT FIX
        #
        # Your trainer calls:
        #
        #     loss_components = model.compute_forecast_loss(...)
        #
        # If no horizons match, the old implementation returned a loss
        # that had no gradient or raised an exception.
        #
        # Here we fail clearly if the target format is genuinely wrong.
        # ------------------------------------------------------------------

        if not losses:

            available_targets = (
                list(targets.keys())
                if isinstance(targets, dict)
                else []
            )

            available_predictions = list(
                predictions.keys()
            )

            raise ValueError(
                "No matching horizons found between predictions and targets. "
                f"Predictions: {available_predictions}; "
                f"Targets: {available_targets}. "
                "Expected target keys such as h15, h30, h60."
            )

        # ------------------------------------------------------------------
        # Total loss
        # ------------------------------------------------------------------

        total = torch.stack(
            list(losses.values())
        ).sum()

        losses["total"] = total

        return losses

    # ========================================================================
    # PREDICTION ERROR
    # ========================================================================

    def compute_prediction_error(
        self,
        predictions: Dict,
        observations: Dict,
    ) -> Dict:

        errors = {}

        for horizon in self.horizons:

            key = f"h{horizon}"

            if key not in predictions:
                continue

            if key not in observations:
                continue

            pred = predictions[key]

            mu = pred["mean"]

            log_var = pred["log_var"].clamp(
                min=-6.0,
                max=6.0,
            )

            std = torch.exp(
                0.5 * log_var
            ).clamp(min=1e-3)

            y = observations[key]

            if not torch.is_tensor(y):

                y = torch.as_tensor(
                    y,
                    dtype=mu.dtype,
                    device=mu.device,
                )

            else:

                y = y.to(
                    dtype=mu.dtype,
                    device=mu.device,
                )

            mae = (
                y - mu
            ).abs()

            z_score = mae / std

            # Gaussian two-sided tail probability.
            normal = torch.distributions.Normal(
                torch.tensor(
                    0.0,
                    device=mu.device,
                ),
                torch.tensor(
                    1.0,
                    device=mu.device,
                ),
            )

            anomaly_prob = (
                2.0
                * (
                    1.0
                    - normal.cdf(
                        z_score.clamp(max=10.0)
                    )
                )
            )

            errors[key] = {
                "mae": mae,
                "z_score": z_score,
                "anomaly_prob": anomaly_prob,
                "std": std,
            }

        return errors

    # ========================================================================
    # MC DROPOUT
    # ========================================================================

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        x: Tensor,
        mc_samples: int = 30,
    ) -> Dict:

        if mc_samples < 1:
            raise ValueError(
                "mc_samples must be >= 1"
            )

        # Save original training state.
        was_training = self.training

        # Evaluation mode.
        self.eval()

        # Enable dropout only.
        for module in self.modules():

            if isinstance(
                module,
                nn.Dropout,
            ):
                module.train()

        all_preds = {
            f"h{h}": []
            for h in self.horizons
        }

        for _ in range(mc_samples):

            output = self.forward(x)

            for horizon in self.horizons:

                all_preds[
                    f"h{horizon}"
                ].append(
                    output[
                        "predictions"
                    ][
                        f"h{horizon}"
                    ]["mean"]
                )

        results = {}

        for horizon in self.horizons:

            preds = torch.stack(
                all_preds[f"h{horizon}"],
                dim=0,
            )

            results[f"h{horizon}"] = {
                "mean": preds.mean(dim=0),

                "std": preds.std(
                    dim=0,
                    unbiased=False,
                ),

                "lower_90": torch.quantile(
                    preds,
                    0.05,
                    dim=0,
                ),

                "upper_90": torch.quantile(
                    preds,
                    0.95,
                    dim=0,
                ),
            }

        # Restore original state.
        self.train(was_training)

        return results

    # ========================================================================
    # RECEPTIVE FIELD
    # ========================================================================

    def get_channel_receptive_field(self) -> Dict[str, float]:

        rf = self.backbone.receptive_field

        return {
            "receptive_field_steps": rf,
            "receptive_field_minutes": rf,
            "receptive_field_hours": rf / 60.0,
        }


# ============================================================================
# FACTORY
# ============================================================================

def build_tcn_predictor(config: dict) -> TCNPredictor:

    return TCNPredictor(
        input_dim=config.get(
            "input_dim",
            8,
        ),

        n_hidden=config.get(
            "n_hidden",
            128,
        ),

        kernel_size=config.get(
            "kernel_size",
            3,
        ),

        n_levels=config.get(
            "n_levels",
            8,
        ),

        horizons=config.get(
            "horizons",
            [15, 30, 60],
        ),

        dropout=config.get(
            "dropout",
            0.1,
        ),

        channel_names=config.get(
            "channel_names",
            None,
        ),
    )


# ============================================================================
# SMOKE TEST
# ============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TCN Predictor Smoke Test")
    print("=" * 60)

    torch.manual_seed(42)

    model = TCNPredictor(
        input_dim=8,
        n_hidden=128,
        kernel_size=3,
        n_levels=8,
        horizons=[15, 30, 60],
        dropout=0.1,
    )

    # ------------------------------------------------------------------------
    # Receptive field
    # ------------------------------------------------------------------------

    rf_info = model.get_channel_receptive_field()

    print(
        f"Receptive field: "
        f"{rf_info['receptive_field_steps']} steps"
    )

    print(
        f"Receptive field: "
        f"{rf_info['receptive_field_hours']:.1f} hours"
    )

    # ------------------------------------------------------------------------
    # Dummy input
    # ------------------------------------------------------------------------

    x = torch.randn(
        4,
        60,
        8,
    )

    # ------------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------------

    out = model(x)

    print(
        f"Anomaly score:  "
        f"{out['anomaly_score'].shape}"
    )

    print(
        f"Channel anomaly: "
        f"{out['channel_anomaly'].shape}"
    )

    print(
        f"H15 prediction:  "
        f"{out['predictions']['h15']['mean'].shape}"
    )

    # ------------------------------------------------------------------------
    # Forecast targets
    # ------------------------------------------------------------------------

    targets = {

        "h15":
            x[:, -1, :]
            + torch.randn(4, 8) * 0.1,

        "h30":
            x[:, -1, :]
            + torch.randn(4, 8) * 0.2,

        "h60":
            x[:, -1, :]
            + torch.randn(4, 8) * 0.3,
    }

    # ------------------------------------------------------------------------
    # Forecast loss
    # ------------------------------------------------------------------------

    losses = model.compute_forecast_loss(
        out,
        targets,
        loss_type="nll",
    )

    print(
        f"Forecast loss:   "
        f"{losses['total'].item():.4f}"
    )

    # ------------------------------------------------------------------------
    # BACKWARD TEST
    # ------------------------------------------------------------------------

    model.zero_grad()

    losses["total"].backward()

    has_gradient = any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Gradient test:   "
        f"{'PASSED' if has_gradient else 'FAILED'}"
    )

    # ------------------------------------------------------------------------
    # MC Dropout
    # ------------------------------------------------------------------------

    mc_out = model.predict_with_uncertainty(
        x[:1],
        mc_samples=5,
    )

    print(
        f"MC mean shape:   "
        f"{mc_out['h15']['mean'].shape}"
    )

    ci_width = (
        mc_out["h15"]["upper_90"]
        - mc_out["h15"]["lower_90"]
    ).mean().item()

    print(
        f"MC 90% CI width: "
        f"{ci_width:.4f}"
    )

    print("=" * 60)
    print("TCN predictor smoke test passed.")
    print("=" * 60)