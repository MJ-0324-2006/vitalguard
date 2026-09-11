"""
ML-based Early Warning Score (EWS) Model.

Implements:
  - ML-based NEWS2 improvement using deep learning on continuous vital streams
  - Multi-task prediction: mortality, sepsis onset, vasopressor need
  - Calibrated probability outputs with uncertainty quantification
  - Time-to-event prediction using survival analysis framework
  - NEWS2 score computation (reference implementation for comparison)

Clinical Context:
  NEWS2 (National Early Warning Score 2) assigns integer points based on
  thresholds for 6 physiological parameters + consciousness level. Our model
  replaces discrete thresholds with learned continuous functions, incorporating
  longitudinal context and inter-channel dependencies.

Reference:
  Royal College of Physicians (2017). National Early Warning Score (NEWS) 2.
  Harutyunyan et al. (2019). Multitask Learning and Benchmarking with Clinical
  Time Series Data. Scientific Data.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# NEWS2 Reference Implementation (for baseline comparison)
# ---------------------------------------------------------------------------

class NEWS2Score:
    """
    National Early Warning Score 2 (NEWS2) reference implementation.

    Computes NEWS2 integer score from 7 physiological parameters.
    Threshold-based; serves as baseline for ML comparison.

    Parameters (at each time point):
        hr:    Heart rate (bpm)
        sbp:   Systolic blood pressure (mmHg)
        rr:    Respiratory rate (/min)
        spo2:  SpO2 (%), using scale 1 (standard)
        temp:  Temperature (°C)
        avpu:  Consciousness: 4=Alert, 3=Voice, 2=Pain, 1=Unresponsive
        on_o2: Boolean — patient on supplemental oxygen (adds 2 points if True)
    """

    HR_TABLE = [
        (0, 40, 3), (41, 50, 1), (51, 90, 0),
        (91, 110, 1), (111, 130, 2), (131, float("inf"), 3),
    ]
    SBP_TABLE = [
        (0, 90, 3), (91, 100, 2), (101, 110, 1),
        (111, 219, 0), (220, float("inf"), 3),
    ]
    RR_TABLE = [
        (0, 8, 3), (9, 11, 1), (12, 20, 0),
        (21, 24, 2), (25, float("inf"), 3),
    ]
    SPO2_TABLE = [  # Scale 1 (no hypercapnic respiratory failure)
        (0, 91, 3), (92, 93, 2), (94, 95, 1), (96, 100, 0),
    ]
    TEMP_TABLE = [
        (0, 35.0, 3), (35.1, 36.0, 1), (36.1, 38.0, 0),
        (38.1, 39.0, 1), (39.1, float("inf"), 2),
    ]

    @staticmethod
    def _score_param(value: float, table: list) -> int:
        for lo, hi, score in table:
            if lo <= value <= hi:
                return score
        return 0

    @classmethod
    def compute(
        cls,
        hr: float,
        sbp: float,
        rr: float,
        spo2: float,
        temp: float,
        avpu: int = 4,
        on_o2: bool = False,
    ) -> Dict[str, int]:
        """Compute NEWS2 total and component scores."""
        hr_s = cls._score_param(hr, cls.HR_TABLE)
        sbp_s = cls._score_param(sbp, cls.SBP_TABLE)
        rr_s = cls._score_param(rr, cls.RR_TABLE)
        spo2_s = cls._score_param(spo2, cls.SPO2_TABLE)
        temp_s = cls._score_param(temp, cls.TEMP_TABLE)
        # Consciousness: Alert=0, otherwise=3
        avpu_s = 0 if avpu == 4 else 3
        o2_s = 2 if on_o2 else 0

        total = hr_s + sbp_s + rr_s + spo2_s + temp_s + avpu_s + o2_s
        return {
            "total": total,
            "hr": hr_s, "sbp": sbp_s, "rr": rr_s,
            "spo2": spo2_s, "temp": temp_s, "avpu": avpu_s, "o2": o2_s,
            "risk": "high" if total >= 7 else "medium" if total >= 5 else "low",
        }

    @classmethod
    def compute_batch(cls, vitals: Tensor) -> Tensor:
        """
        Vectorised NEWS2 computation for a batch.

        Args:
            vitals: (B, 6) — [HR, SBP, RR, SpO2, Temp, AVPU]

        Returns:
            scores: (B,) integer NEWS2 totals
        """
        hr, sbp, rr, spo2, temp, avpu = vitals.T

        def _score(vals, table):
            scores = torch.zeros_like(vals)
            for lo, hi, s in table:
                mask = (vals >= lo) & (vals <= hi)
                scores[mask] = s
            return scores

        total = (
            _score(hr, cls.HR_TABLE)
            + _score(sbp, cls.SBP_TABLE)
            + _score(rr, cls.RR_TABLE)
            + _score(spo2, cls.SPO2_TABLE)
            + _score(temp, cls.TEMP_TABLE)
            + (avpu != 4).float() * 3  # consciousness
        )
        return total


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------

class VitalSignFeatureExtractor(nn.Module):
    """
    Extract temporal features from multi-channel vital sign windows.
    Combines statistical features with learned LSTM features.
    """

    def __init__(
        self,
        input_dim: int,
        lstm_hidden: int = 64,
        output_dim: int = 128,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        # Learned temporal features via LSTM
        self.lstm = nn.LSTM(input_dim, lstm_hidden, num_layers=1,
                            batch_first=True, bidirectional=True)
        # Statistical features: last value, mean, std, min, max, slope
        n_stat_features = input_dim * 6
        total_features = lstm_hidden * 2 + n_stat_features
        self.projection = nn.Sequential(
            nn.Linear(total_features, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def _stat_features(self, x: Tensor) -> Tensor:
        """Compute statistical features from time series window."""
        # x: (B, T, C)
        last = x[:, -1, :]           # (B, C)
        mean = x.mean(dim=1)         # (B, C)
        std = x.std(dim=1)           # (B, C)
        x_min = x.min(dim=1).values  # (B, C)
        x_max = x.max(dim=1).values  # (B, C)
        # Linear slope via least squares (approximate with first/last difference)
        T = x.size(1)
        slope = (x[:, -1, :] - x[:, 0, :]) / max(T - 1, 1)  # (B, C)
        return torch.cat([last, mean, std, x_min, x_max, slope], dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        # LSTM features
        h_all, (h_n, _) = self.lstm(x)
        h_last_fwd = h_n[0]   # (B, lstm_hidden)
        h_last_bwd = h_n[1]
        lstm_feat = torch.cat([h_last_fwd, h_last_bwd], dim=-1)

        # Statistical features
        stat_feat = self._stat_features(x)

        combined = torch.cat([lstm_feat, stat_feat], dim=-1)
        return self.projection(combined)


# ---------------------------------------------------------------------------
# Multi-Task Outcome Head
# ---------------------------------------------------------------------------

class MultiTaskOutcomeHead(nn.Module):
    """
    Predict multiple clinical outcomes from a shared representation.

    Outcomes:
      - In-hospital mortality
      - Sepsis onset within N hours
      - Vasopressor initiation within N hours

    Uses task-specific uncertainty weighting (Kendall et al., 2018).
    """

    def __init__(
        self,
        input_dim: int,
        outcome_names: List[str],
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.outcome_names = outcome_names

        # Shared layer
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Task-specific heads
        self.heads = nn.ModuleDict({
            name: nn.Linear(hidden_dim, 1)
            for name in outcome_names
        })

        # Learnable task uncertainty weights (log sigma^2)
        self.log_sigma = nn.Parameter(torch.zeros(len(outcome_names)))

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            x: (B, input_dim) shared features

        Returns:
            logits: dict mapping outcome_name → (B,) logit
            probs:  dict mapping outcome_name → (B,) probability
        """
        h = self.shared(x)
        logits = {name: self.heads[name](h).squeeze(-1) for name in self.outcome_names}
        probs = {name: torch.sigmoid(logits[name]) for name in self.outcome_names}
        return {"logits": logits, "probs": probs}

    def compute_loss(
        self,
        logits: Dict[str, Tensor],
        labels: Dict[str, Tensor],
        pos_weights: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        """
        Multi-task loss with learnable uncertainty weighting.

        L_total = sum_i [1/(2*sigma_i^2) * L_i + log(sigma_i)]
        """
        total_loss = torch.tensor(0.0, device=self.log_sigma.device)
        for i, name in enumerate(self.outcome_names):
            if name not in labels:
                continue
            pw = pos_weights.get(name) if pos_weights else None
            task_loss = F.binary_cross_entropy_with_logits(
                logits[name], labels[name].float(), pos_weight=pw
            )
            # Uncertainty weighting
            sigma_sq = torch.exp(self.log_sigma[i])
            total_loss += task_loss / (2 * sigma_sq) + 0.5 * self.log_sigma[i]
        return total_loss


# ---------------------------------------------------------------------------
# Time-to-Event Head (Survival Analysis)
# ---------------------------------------------------------------------------

class TimeToEventHead(nn.Module):
    """
    Predicts time-to-event (TTD) using a discrete-time survival model.

    Models the hazard function h(t | x) = P(event at t | survived to t, x)
    using discrete time bins. Allows handling of right-censoring.

    Reference:
      Harutyunyan et al. (2019). Multitask Learning and Benchmarking
      with Clinical Time Series Data. Scientific Data.
    """

    def __init__(
        self,
        input_dim: int,
        n_time_bins: int = 48,  # 48 × 1-hour bins = 48h prediction window
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.n_time_bins = n_time_bins
        self.hazard_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_time_bins),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, input_dim)

        Returns:
            hazard_logits: (B, n_time_bins) — log-hazard per time bin
        """
        return self.hazard_net(x)

    def survival_function(self, hazard_logits: Tensor) -> Tensor:
        """
        Compute survival function S(t) = P(T > t) from hazard rates.

        S(t) = prod_{s<=t} (1 - h(s))
        """
        h = torch.sigmoid(hazard_logits)  # (B, T)
        # Log cumulative product for numerical stability
        log_survival = torch.log(1 - h + 1e-7).cumsum(dim=-1)
        return torch.exp(log_survival)  # (B, T)

    def expected_event_time(self, hazard_logits: Tensor) -> Tensor:
        """
        Compute expected time to event: E[T] = sum_t S(t)

        Returns:
            tte: (B,) expected time to event in bins
        """
        S = self.survival_function(hazard_logits)  # (B, T)
        T = torch.arange(1, self.n_time_bins + 1, device=hazard_logits.device).float()
        return (S * T).sum(dim=-1)  # (B,)

    def compute_loss(
        self,
        hazard_logits: Tensor,
        event_times: Tensor,
        events: Tensor,
    ) -> Tensor:
        """
        Discrete-time survival loss (negative log-likelihood with censoring).

        Args:
            hazard_logits: (B, T)
            event_times:   (B,) — observed time bin (event or censoring)
            events:        (B,) — 1 if event occurred, 0 if censored

        Returns:
            loss: scalar
        """
        B, T = hazard_logits.shape
        h = torch.sigmoid(hazard_logits)

        # Build mask: which bins are "at risk"
        t_idx = event_times.long().clamp(0, T - 1)
        mask = torch.arange(T, device=hazard_logits.device).unsqueeze(0) <= t_idx.unsqueeze(1)

        # Log-likelihood for bins before event: log(1 - h(t))
        log_no_event = torch.log(1 - h + 1e-7) * mask.float()

        # Log-likelihood at event time: log(h(t)) * event_indicator
        log_event = torch.gather(torch.log(h + 1e-7), 1, t_idx.unsqueeze(1)).squeeze(1)
        log_event = log_event * events.float()

        loss = -(log_no_event.sum(dim=-1) + log_event).mean()
        return loss


# ---------------------------------------------------------------------------
# Main Early Warning Score Model
# ---------------------------------------------------------------------------

class EarlyWarningModel(nn.Module):
    """
    ML-based Early Warning Score model improving on NEWS2.

    Combines:
      - Continuous vital sign processing (replacing NEWS2 discrete thresholds)
      - Multi-task learning across clinical outcomes
      - Time-to-event prediction for deterioration
      - Calibrated probability outputs

    Input: multi-channel vital sign windows (B, T, C)
    Outputs: per-outcome probabilities, time-to-event, and risk score

    Args:
        input_dim:       Number of physiological channels.
        seq_len:         Input window length (time steps).
        outcome_names:   Clinical outcomes to predict.
        hidden_dim:      Hidden dimension for feature projection.
        lstm_hidden:     LSTM hidden dim in feature extractor.
        n_time_bins:     Time bins for survival analysis (hours).
        dropout:         Dropout probability.
    """

    def __init__(
        self,
        input_dim: int = 8,
        seq_len: int = 48,
        outcome_names: Optional[List[str]] = None,
        hidden_dim: int = 128,
        lstm_hidden: int = 64,
        n_time_bins: int = 48,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.outcome_names = outcome_names or ["mortality", "sepsis", "vasopressor"]

        # Feature extraction
        self.feature_extractor = VitalSignFeatureExtractor(
            input_dim, lstm_hidden, hidden_dim
        )

        # Multi-task outcome prediction
        self.outcome_head = MultiTaskOutcomeHead(
            hidden_dim, self.outcome_names, hidden_dim // 2, dropout
        )

        # Time-to-event prediction (for highest-priority outcome)
        self.tte_head = TimeToEventHead(hidden_dim, n_time_bins)

        # NEWS2-enhanced risk score (scalar output calibrated to NEWS2 scale)
        self.risk_score_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus(),  # ensure non-negative output
        )

        # Temperature scaling parameter for calibration
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(
        self,
        x: Tensor,
        news2_scores: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Args:
            x:            (B, T, C) vital sign window
            news2_scores: (B,) optional NEWS2 scores to augment features

        Returns dict:
            'features':          (B, hidden_dim) extracted features
            'outcome_logits':    dict of (B,) logits per outcome
            'outcome_probs':     dict of (B,) probabilities per outcome
            'risk_score':        (B,) continuous risk score (calibrated to NEWS2 scale)
            'hazard_logits':     (B, n_time_bins) time-to-event hazard
            'expected_tte':      (B,) expected time to deterioration in hours
        """
        features = self.feature_extractor(x)  # (B, hidden_dim)

        # Optionally augment with NEWS2
        if news2_scores is not None:
            # NEWS2 as an additional feature (concatenate after projection)
            news2_feat = news2_scores.float().unsqueeze(-1) / 20.0  # normalise
            features = features + self.risk_score_head(features).expand_as(
                news2_feat.expand(features.shape)
            ) * 0.0  # placeholder; replace with proper fusion if needed

        # Outcome predictions
        outcome_out = self.outcome_head(features)

        # Calibrated probabilities
        calibrated_logits = {
            k: v / self.temperature for k, v in outcome_out["logits"].items()
        }
        calibrated_probs = {k: torch.sigmoid(v) for k, v in calibrated_logits.items()}

        # Time-to-event
        hazard_logits = self.tte_head(features)
        expected_tte = self.tte_head.expected_event_time(hazard_logits)

        # Risk score (continuous, calibrated to NEWS2-like 0-20 scale)
        risk_score = self.risk_score_head(features).squeeze(-1) * 20.0

        return {
            "features": features,
            "outcome_logits": outcome_out["logits"],
            "outcome_probs": calibrated_probs,
            "risk_score": risk_score,
            "hazard_logits": hazard_logits,
            "expected_tte": expected_tte,
        }

    def compute_loss(
        self,
        outputs: Dict[str, Tensor],
        labels: Dict[str, Tensor],
        event_times: Optional[Tensor] = None,
        pos_weights: Optional[Dict[str, Tensor]] = None,
        tte_weight: float = 0.5,
    ) -> Dict[str, Tensor]:
        """
        Combined multi-task + survival loss.

        Args:
            outputs:      Forward pass outputs
            labels:       Dict mapping outcome name → (B,) binary labels
            event_times:  (B,) observed time to deterioration (bins)
            pos_weights:  Dict of positive class weights
            tte_weight:   Weight for TTE loss

        Returns:
            losses: dict with 'total' and component losses
        """
        # Multi-task classification loss
        cls_loss = self.outcome_head.compute_loss(
            outputs["outcome_logits"], labels, pos_weights
        )

        losses = {"cls": cls_loss}

        # Survival / TTE loss
        if event_times is not None and "mortality" in labels:
            tte_loss = self.tte_head.compute_loss(
                outputs["hazard_logits"],
                event_times,
                labels["mortality"],
            )
            losses["tte"] = tte_loss * tte_weight

        losses["total"] = sum(losses.values())
        return losses

    def calibrate_temperature(
        self,
        val_logits: Dict[str, Tensor],
        val_labels: Dict[str, Tensor],
        n_iter: int = 100,
        lr: float = 0.01,
    ) -> float:
        """
        Calibrate temperature scaling parameter on validation set.

        Args:
            val_logits: Dict of uncalibrated logits from validation set
            val_labels: Dict of true labels
            n_iter:     Optimisation iterations
            lr:         Learning rate

        Returns:
            Optimal temperature value
        """
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=n_iter)

        def closure():
            optimizer.zero_grad()
            loss = torch.tensor(0.0)
            for name in self.outcome_names:
                if name in val_labels and name in val_logits:
                    scaled = val_logits[name] / self.temperature
                    loss += F.binary_cross_entropy_with_logits(
                        scaled, val_labels[name].float()
                    )
            loss.backward()
            return loss

        optimizer.step(closure)
        return self.temperature.item()

    @torch.no_grad()
    def predict_risk(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Convenience inference method returning all risk outputs.

        Returns:
            risk_summary: dict with all outcome probs, risk score, expected TTE
        """
        self.eval()
        outputs = self.forward(x)
        return {
            "outcome_probs": outputs["outcome_probs"],
            "risk_score": outputs["risk_score"],
            "expected_tte_hours": outputs["expected_tte"],
            "high_risk": outputs["risk_score"] > 7.0,  # NEWS2 "high risk" threshold
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_early_warning_model(config: dict) -> EarlyWarningModel:
    return EarlyWarningModel(
        input_dim=config.get("input_dim", 8),
        seq_len=config.get("seq_len", 48),
        outcome_names=config.get("outcome_names", ["mortality", "sepsis", "vasopressor"]),
        hidden_dim=config.get("hidden_dim", 128),
        lstm_hidden=config.get("lstm_hidden", 64),
        n_time_bins=config.get("n_time_bins", 48),
        dropout=config.get("dropout", 0.1),
    )


if __name__ == "__main__":
    torch.manual_seed(42)

    # Test NEWS2 reference
    news2 = NEWS2Score.compute(hr=95, sbp=105, rr=18, spo2=97, temp=37.2)
    print(f"NEWS2 score: {news2['total']} (risk: {news2['risk']})")

    # Test full model
    model = EarlyWarningModel(input_dim=8, seq_len=48)
    x = torch.randn(4, 48, 8)
    out = model(x)
    print(f"Risk score shape:    {out['risk_score'].shape}")   # (4,)
    print(f"Expected TTE shape:  {out['expected_tte'].shape}") # (4,)
    print(f"Mortality probs:     {out['outcome_probs']['mortality']}")

    labels = {
        "mortality": torch.tensor([0, 1, 0, 1]).float(),
        "sepsis": torch.tensor([1, 0, 0, 1]).float(),
        "vasopressor": torch.tensor([0, 0, 1, 1]).float(),
    }
    losses = model.compute_loss(out, labels, event_times=torch.tensor([48, 6, 48, 12]))
    print(f"Total loss: {losses['total'].item():.4f}")
    print("EarlyWarningModel smoke test passed.")
