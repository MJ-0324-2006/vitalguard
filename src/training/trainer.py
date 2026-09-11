"""
Multi-Task Training Pipeline for ICU Anomaly Detection Models.

Handles:
  - Reconstruction loss (LSTM-AE, TCN)
  - Classification loss (event prediction)
  - Multi-task loss weighting
  - Class imbalance (mortality ~10%, sepsis ~15%)
  - Mixed precision (AMP) training
  - Gradient clipping, LR scheduling
  - Early stopping with patience
  - Checkpoint saving/loading
  - TensorBoard / W&B logging
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainerConfig:
    """All hyperparameters for training."""

    # General
    model_type: str = "transformer"         # 'transformer', 'lstm_ae', 'tcn', 'early_warning'
    device: str = "auto"                    # 'cuda', 'cpu', 'auto'
    seed: int = 42

    # Optimiser
    lr: float = 3e-4
    weight_decay: float = 1e-5
    optimizer: str = "adamw"               # 'adam', 'adamw', 'sgd'
    gradient_clip_norm: float = 1.0

    # Scheduler
    scheduler: str = "cosine"             # 'cosine', 'plateau', 'warmup_cosine', 'none'
    warmup_steps: int = 200
    min_lr: float = 1e-6
    patience_scheduler: int = 5           # for ReduceLROnPlateau

    # Training loop
    epochs: int = 100
    batch_size: int = 64
    accumulation_steps: int = 1           # gradient accumulation
    mixed_precision: bool = True

    # Loss weights
    recon_weight: float = 1.0
    cls_weight: float = 1.0
    kl_weight: float = 0.5
    tte_weight: float = 0.5

    # Class imbalance
    use_weighted_sampler: bool = True
    outcome: str = "mortality"            # outcome to use for sampling weights

    # Early stopping
    patience: int = 15
    min_delta: float = 1e-4
    monitor: str = "auto"                 # 'auto', 'val_auroc', or 'val_loss'

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_best_only: bool = True
    save_every_n_epochs: int = 10

    # Logging
    log_interval: int = 50               # steps
    use_wandb: bool = False
    project_name: str = "icu-anomaly"
    run_name: str = "run_v1"

    # Data
    num_workers: int = 4
    pin_memory: bool = True


# ---------------------------------------------------------------------------
# Metrics tracking
# ---------------------------------------------------------------------------

class MetricsTracker:
    """Accumulate and compute running metrics during training."""

    def __init__(self) -> None:
        self._sums: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}

    def update(self, metrics: Dict[str, float], n: int = 1) -> None:
        for k, v in metrics.items():
            if isinstance(v, Tensor):
                v = v.item()
            self._sums[k] = self._sums.get(k, 0.0) + v * n
            self._counts[k] = self._counts.get(k, 0) + n

    def compute(self) -> Dict[str, float]:
        return {
            k: self._sums[k] / self._counts[k]
            for k in self._sums
            if self._counts[k] > 0
        }

    def reset(self) -> None:
        self._sums.clear()
        self._counts.clear()


# ---------------------------------------------------------------------------
# Learning Rate Schedulers
# ---------------------------------------------------------------------------

class WarmupCosineScheduler:
    """Linear warmup followed by cosine annealing."""

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self._step = 0
        self._base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self) -> None:
        self._step += 1
        for i, pg in enumerate(self.optimizer.param_groups):
            if self._step < self.warmup_steps:
                lr = self._base_lrs[i] * (self._step / max(self.warmup_steps, 1))
            else:
                progress = (self._step - self.warmup_steps) / max(
                    self.total_steps - self.warmup_steps, 1
                )
                lr = self.min_lr + 0.5 * (self._base_lrs[i] - self.min_lr) * (
                    1 + np.cos(np.pi * progress)
                )
            pg["lr"] = max(lr, self.min_lr)

    @property
    def last_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Stop training when monitored metric stops improving."""

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 1e-4,
        mode: str = "max",  # 'max' for AUROC, 'min' for loss
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Universal trainer for all ICU anomaly detection models.

    Supports:
      - LSTMAutoencoder (unsupervised / semi-supervised)
      - TransformerAnomalyDetector (supervised multi-task)
      - TCNPredictor (forecasting)
      - EarlyWarningModel (multi-task classification + survival)

    Usage:
        trainer = Trainer(model, config)
        trainer.fit(train_loader, val_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[TrainerConfig] = None,
    ) -> None:
        self.config = config or TrainerConfig()
        self.model = model

        # Device setup
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        logger.info(f"Training on device: {self.device}")
        self.model = self.model.to(self.device)

        # Reproducibility
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        # Optimiser
        self.optimizer = self._build_optimizer()

        # Mixed precision scaler
        self.scaler = GradScaler(enabled=self.config.mixed_precision and self.device.type == "cuda")

        # Metrics and state
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()

        # Determine monitor metric dynamically
        model_type = type(self.model).__name__
        if hasattr(self.model, "outcome_names") and model_type != "LSTMAutoencoder":
            self.monitor = "val_auroc"
        else:
            self.monitor = "val_loss"

        # Update the config to reflect the chosen monitor
        self.config.monitor = self.monitor

        self.early_stopping = EarlyStopping(
            patience=self.config.patience,
            min_delta=self.config.min_delta,
            mode="max" if "auroc" in self.monitor else "min",
        )
        self.history: List[Dict] = []
        self.best_epoch: int = 0
        self.global_step: int = 0

        # Checkpoint dir
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # W&B
        if self.config.use_wandb:
            self._init_wandb()

    def _build_optimizer(self) -> optim.Optimizer:
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.config.optimizer == "adamw":
            return optim.AdamW(params, lr=self.config.lr, weight_decay=self.config.weight_decay)
        elif self.config.optimizer == "adam":
            return optim.Adam(params, lr=self.config.lr, weight_decay=self.config.weight_decay)
        elif self.config.optimizer == "sgd":
            return optim.SGD(params, lr=self.config.lr, momentum=0.9,
                             weight_decay=self.config.weight_decay)
        raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _build_scheduler(self, total_steps: int):
        if self.config.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_steps, eta_min=self.config.min_lr
            )
        elif self.config.scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", patience=self.config.patience_scheduler,
                min_lr=self.config.min_lr
            )
        elif self.config.scheduler == "warmup_cosine":
            return WarmupCosineScheduler(
                self.optimizer, self.config.warmup_steps,
                total_steps, self.config.min_lr
            )
        return None

    def _init_wandb(self) -> None:
        try:
            import wandb
            wandb.init(
                project=self.config.project_name,
                name=self.config.run_name,
                config=self.config.__dict__,
            )
            self._wandb = wandb
        except ImportError:
            logger.warning("wandb not installed; disabling W&B logging")
            self.config.use_wandb = False

    def _compute_pos_weights(self, train_loader: DataLoader) -> Dict[str, Tensor]:
        """Compute positive class weights from training set label distribution."""
        outcome_names = getattr(self.model, "outcome_names",
                                getattr(self.model, "outcome_head",
                                        None) and getattr(self.model.outcome_head, "outcome_names", []))
        if not outcome_names:
            return {}

        label_counts = {name: {"pos": 0, "total": 0} for name in outcome_names}
        for batch in train_loader:
            for name in outcome_names:
                if name in batch:
                    y = batch[name]
                    valid = y[~torch.isnan(y)]
                    label_counts[name]["pos"] += int(valid.sum().item())
                    label_counts[name]["total"] += int(len(valid))
            break  # one pass to estimate

        pos_weights = {}
        for name, counts in label_counts.items():
            n_pos = max(counts["pos"], 1)
            n_neg = max(counts["total"] - n_pos, 1)
            pos_weights[name] = torch.tensor(n_neg / n_pos).to(self.device)
        return pos_weights

    def _validate_config(self, batch: Dict[str, Tensor]) -> None:
        """Validate config against model expectations using a sample batch."""
        x = batch["x"]
        B, T, C = x.shape
        model_type = type(self.model).__name__

        if model_type == "TCNPredictor" and hasattr(self.model, "horizons"):
            invalid_horizons = [h for h in self.model.horizons if h >= T]
            if invalid_horizons:
                raise ValueError(f"Config mismatch: TCN horizons {invalid_horizons} "
                                 f"are >= input sequence length ({T}). Targets would be empty.")

    def _build_targets(self, batch: Dict[str, Tensor], device: torch.device) -> Dict[str, Tensor]:
        """Model-agnostic target builder based on model capabilities."""
        x = batch["x"].to(device)
        B, T, C = x.shape
        targets = {}

        # 1. Forecasting targets
        if hasattr(self.model, "horizons"):
            for h in self.model.horizons:
                if h < T:
                    targets[f"h{h}"] = x[:, h:, :].mean(dim=1)

        # 2. Classification outcome targets
        if hasattr(self.model, "outcome_names"):
            _primary = self.model.outcome_names[0]  # e.g. 'mortality'
            for name in self.model.outcome_names:
                if name in batch:
                    targets[name] = batch[name].to(device).float()
                elif "label" in batch and name == _primary:
                    # Dataset stores primary outcome under generic 'label' key.
                    targets[name] = batch["label"].to(device).float()

        # 3. Survival/Time-to-event targets
        if "event_times" in batch:
            targets["event_times"] = batch["event_times"].to(device)

        return targets

    def _forward_and_loss(
        self,
        batch: Dict[str, Tensor],
        pos_weights: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Forward pass + loss computation for any model type.

        Returns:
            total_loss: scalar
            loss_dict:  component losses for logging
        """
        x = batch["x"].to(self.device)
        targets = self._build_targets(batch, self.device)

        with autocast(enabled=self.config.mixed_precision and self.device.type == "cuda"):
            outputs = self.model(x)
            model_type = type(self.model).__name__

            if model_type == "LSTMAutoencoder":
                total_loss = self.model.compute_loss(outputs)
                loss_dict = {
                    "recon": outputs["recon_loss"].item(),
                    "kl": outputs["kl_loss"].item() if "kl_loss" in outputs else 0,
                    "total": total_loss.item(),
                }

            elif model_type == "TransformerAnomalyDetector":
                pw = pos_weights if pos_weights else None
                pw_tensor = None
                if pw:
                    pw_tensor = torch.stack([
                        pw.get(name, torch.tensor(1.0, device=self.device))
                        for name in getattr(self.model, "outcome_names", [])
                    ])

                loss_components = self.model.compute_loss(
                    outputs, targets,
                    recon_weight=self.config.recon_weight,
                    cls_weight=self.config.cls_weight,
                    pos_weights=pw_tensor,
                )
                total_loss = loss_components["total"]
                loss_dict = {k: v.item() if isinstance(v, Tensor) else v
                             for k, v in loss_components.items()}

            elif model_type == "TCNPredictor":
                loss_components = self.model.compute_forecast_loss(outputs, targets, "nll")
                total_loss = loss_components["total"]
                loss_dict = {k: v.item() if isinstance(v, Tensor) else v
                             for k, v in loss_components.items()}

            elif model_type == "EarlyWarningModel":
                event_times = targets.get("event_times")
                cls_labels = {k: v for k, v in targets.items() if k != "event_times"}
                loss_components = self.model.compute_loss(
                    outputs, cls_labels, event_times,
                    tte_weight=self.config.tte_weight,
                )
                total_loss = loss_components["total"]
                loss_dict = {k: v.item() if isinstance(v, Tensor) else v
                             for k, v in loss_components.items()}
            else:
                raise ValueError(f"Unknown model type: {model_type}")

        return total_loss, loss_dict

    def _train_epoch(
        self,
        train_loader: DataLoader,
        scheduler,
        pos_weights: Optional[Dict[str, Tensor]] = None,
        epoch: int = 0,
    ) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        self.train_metrics.reset()
        self.optimizer.zero_grad()
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            with autocast(enabled=self.config.mixed_precision and self.device.type == "cuda"):
                loss, loss_dict = self._forward_and_loss(batch, pos_weights)
                loss = loss / self.config.accumulation_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.config.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                if scheduler is not None and self.config.scheduler == "warmup_cosine":
                    scheduler.step()

                self.global_step += 1

            self.train_metrics.update(loss_dict)

            if (step + 1) % self.config.log_interval == 0:
                metrics = self.train_metrics.compute()
                elapsed = time.time() - t0
                logger.info(
                    f"Epoch {epoch+1} | Step {step+1}/{len(train_loader)} | "
                    f"Loss: {metrics.get('total', 0):.4f} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | "
                    f"Time: {elapsed:.1f}s"
                )

        return self.train_metrics.compute()

    @torch.no_grad()
    def _validate(
        self,
        val_loader: DataLoader,
        pos_weights: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, float]:
        """Run validation epoch and compute AUROC."""
        from sklearn.metrics import roc_auc_score

        self.model.eval()
        self.val_metrics.reset()

        all_scores: Dict[str, List] = {}
        all_labels: Dict[str, List] = {}

        for batch in val_loader:
            x = batch["x"].to(self.device)
            loss, loss_dict = self._forward_and_loss(batch, pos_weights)
            self.val_metrics.update(loss_dict)

            # Collect predictions for AUROC
            outputs = self.model(x)
            model_type = type(self.model).__name__

            if model_type == "TransformerAnomalyDetector":
                probs = torch.sigmoid(outputs["outcome_logits"])
                # Dataset stores primary outcome as generic 'label' key.
                # Map it to the first outcome name (mortality).
                _primary_outcome = self.model.outcome_names[0]  # 'mortality'
                for i, name in enumerate(self.model.outcome_names):
                    if name not in all_scores:
                        all_scores[name] = []
                        all_labels[name] = []
                    all_scores[name].extend(probs[:, i].cpu().numpy().tolist())
                    if name in batch:
                        # exact key match (future-proof for multi-label datasets)
                        all_labels[name].extend(batch[name].cpu().numpy().tolist())
                    elif "label" in batch and name == _primary_outcome:
                        # fallback: dataset uses generic 'label' for primary outcome
                        all_labels[name].extend(batch["label"].cpu().numpy().tolist())

            elif model_type in ("EarlyWarningModel",):
                for name, probs in outputs["outcome_probs"].items():
                    if name not in all_scores:
                        all_scores[name] = []
                        all_labels[name] = []
                    all_scores[name].extend(probs.cpu().numpy().tolist())
                    if name in batch:
                        all_labels[name].extend(batch[name].numpy().tolist())
                    elif "label" in batch and name == list(outputs["outcome_probs"].keys())[0]:
                        all_labels[name].extend(batch["label"].cpu().numpy().tolist())

            else:
                # Unsupervised: collect anomaly scores
                if "anomaly_score" in outputs:
                    all_scores.setdefault("anomaly", []).extend(
                        outputs["anomaly_score"].cpu().numpy().tolist()
                    )

        val_result = self.val_metrics.compute()

        # Compute AUROC where possible
        for name in all_scores:
            if name in all_labels and len(all_labels[name]) > 0:
                y = np.array(all_labels[name])
                s = np.array(all_scores[name])
                valid = ~np.isnan(y)
                if valid.sum() > 0 and len(np.unique(y[valid])) > 1:
                    try:
                        auroc = roc_auc_score(y[valid].astype(int), s[valid])
                        val_result[f"auroc_{name}"] = auroc
                    except Exception:
                        pass

        # Primary AUROC for early stopping
        auroc_vals = [v for k, v in val_result.items() if k.startswith("auroc_")]
        if auroc_vals:
            val_result["val_auroc"] = np.mean(auroc_vals)

        return val_result

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        resume_from: Optional[str] = None,
    ) -> Dict[str, List]:
        """
        Main training loop.

        Args:
            train_loader: DataLoader for training set
            val_loader:   DataLoader for validation set
            resume_from:  Path to checkpoint to resume from

        Returns:
            history: dict of metric lists per epoch
        """
        # Validate config against a sample batch
        sample_batch = next(iter(train_loader))
        self._validate_config(sample_batch)

        if resume_from:
            self.load_checkpoint(resume_from)

        # Compute positive class weights
        pos_weights = self._compute_pos_weights(train_loader)
        if pos_weights:
            logger.info(f"Positive class weights: {pos_weights}")

        # Build scheduler
        total_steps = len(train_loader) * self.config.epochs
        scheduler = self._build_scheduler(total_steps)

        history: Dict[str, List] = {"train_loss": [], "val_loss": [], "val_auroc": []}

        logger.info(
            f"Starting training: {self.config.epochs} epochs | "
            f"batch={self.config.batch_size} | lr={self.config.lr}"
        )

        for epoch in range(self.config.epochs):
            # Train
            train_metrics = self._train_epoch(
                train_loader, scheduler, pos_weights, epoch
            )

            # Validate
            val_metrics = self._validate(val_loader, pos_weights)

            # Scheduler step (epoch-based)
            if scheduler is not None and self.config.scheduler in ("cosine", "plateau"):
                if self.config.scheduler == "plateau":
                    scheduler.step(val_metrics.get("val_auroc", val_metrics.get("total", 0)))
                else:
                    scheduler.step()

            # Logging
            log_str = (
                f"Epoch {epoch+1:3d}/{self.config.epochs} | "
                f"Train loss: {train_metrics.get('total', 0):.4f} | "
                f"Val loss: {val_metrics.get('total', 0):.4f} | "
                f"Val AUROC: {val_metrics.get('val_auroc', 0):.4f}"
            )
            logger.info(log_str)

            epoch_record = {
                "epoch": epoch + 1,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            self.history.append(epoch_record)
            history["train_loss"].append(train_metrics.get("total", 0))
            history["val_loss"].append(val_metrics.get("total", 0))
            history["val_auroc"].append(val_metrics.get("val_auroc", 0))

            if self.config.use_wandb:
                self._wandb.log(epoch_record, step=epoch + 1)

            # Checkpoint
            monitor_val = val_metrics.get(self.monitor, val_metrics.get("total", 0))
            if self.config.save_best_only:
                is_improved = False
                if self.early_stopping.best_score is None:
                    is_improved = True
                else:
                    if self.early_stopping.mode == "max":
                        is_improved = monitor_val > self.early_stopping.best_score
                    else:
                        is_improved = monitor_val < self.early_stopping.best_score

                if is_improved:
                    self.save_checkpoint(epoch, val_metrics, is_best=True)
                    self.best_epoch = epoch + 1
            elif (epoch + 1) % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(epoch, val_metrics)

            # Early stopping
            if self.early_stopping(monitor_val):
                logger.info(
                    f"Early stopping at epoch {epoch+1} "
                    f"(best at epoch {self.best_epoch})"
                )
                break

        logger.info(f"Training complete. Best epoch: {self.best_epoch}")
        return history

    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
    ) -> str:
        """Save model checkpoint."""
        fname = "best_model.pt" if is_best else f"checkpoint_epoch_{epoch+1:04d}.pt"
        path = self.checkpoint_dir / fname
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config.__dict__,
            "history": self.history,
        }, path)
        logger.debug(f"Saved checkpoint: {path}")
        return str(path)

    def load_checkpoint(self, path: str) -> Dict:
        """Load model and optimizer state from checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.history = ckpt.get("history", [])
        logger.info(f"Loaded checkpoint from {path} (epoch {ckpt['epoch']+1})")
        return ckpt

    @classmethod
    def load_for_inference(cls, path: str, model: nn.Module) -> nn.Module:
        """Load model weights from checkpoint for inference only."""
        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    config: TrainerConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders with optional weighted sampling."""
    train_sampler = None
    if config.use_weighted_sampler and hasattr(train_dataset, "sample_weights"):
        weights = train_dataset.sample_weights
        train_sampler = WeightedRandomSampler(
            weights=torch.from_numpy(weights).float(),
            num_samples=len(weights),
            replacement=True,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    from src.models.transformer_detector import TransformerAnomalyDetector
    from src.data.preprocessing import ICUTimeSeriesDataset

    torch.manual_seed(42)

    # Tiny synthetic dataset
    windows = []
    for i in range(200):
        windows.append({
            "x": np.random.randn(48, 8).astype(np.float32),
            "label": int(np.random.rand() < 0.15),
            "mortality": int(np.random.rand() < 0.10),
            "sepsis": int(np.random.rand() < 0.15),
            "vasopressor": int(np.random.rand() < 0.18),
            "missing_mask": np.zeros((48, 8), dtype=bool),
        })

    dataset = ICUTimeSeriesDataset(windows, outcome="label")
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=16, shuffle=False)

    model = TransformerAnomalyDetector(input_dim=8, d_model=64, n_heads=4,
                                        n_layers=2, ff_dim=128, seq_len=48)
    config = TrainerConfig(epochs=3, lr=1e-3, use_wandb=False, mixed_precision=False)
    trainer = Trainer(model, config)
    history = trainer.fit(train_loader, val_loader)
    print(f"Training history epochs: {len(history['train_loss'])}")
    print("Trainer smoke test passed.")