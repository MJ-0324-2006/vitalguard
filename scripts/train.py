"""
Model Training Script.

Usage:
    # Train Transformer (recommended)
    python scripts/train.py --config configs/mimic_config.yaml --model transformer

    # Train LSTM autoencoder (unsupervised)
    python scripts/train.py --config configs/mimic_config.yaml --model lstm_ae

    # Train TCN
    python scripts/train.py --config configs/mimic_config.yaml --model tcn

    # Train with W&B logging
    python scripts/train.py --config configs/mimic_config.yaml --model transformer --wandb
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import (
    ICUTimeSeriesDataset,
    PreprocessingPipeline,
)
from src.models.lstm_autoencoder import build_lstm_ae
from src.models.transformer_detector import build_transformer_detector
from src.models.tcn_predictor import build_tcn_predictor
from src.models.early_warning import build_early_warning_model
from src.training.trainer import Trainer, TrainerConfig, build_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train VitalGuard deterioration detection model")
    parser.add_argument("--config", type=str, default="configs/mimic_config.yaml")
    parser.add_argument("--model", type=str, default="transformer",
                        choices=["transformer", "lstm_ae", "tcn", "early_warning"])
    parser.add_argument("--data-dir", type=str, default="data/processed/")
    parser.add_argument("--output-dir", type=str, default="experiments/")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--debug", action="store_true",
                        help="Use tiny synthetic dataset for debugging")
    return parser.parse_args()


def load_datasets(data_dir: str, cfg: dict, outcome: str = "mortality"):
    """Load train/val/test datasets from processed data."""
    import pandas as pd
    from pathlib import Path

    data_dir = Path(data_dir)
    vitals_dir = data_dir / "vitals"
    labels_dir = data_dir / "labels"

    train_ids = pd.read_csv(data_dir / "train_ids.csv")["subject_id"].tolist()
    val_ids = pd.read_csv(data_dir / "val_ids.csv")["subject_id"].tolist()
    test_ids = pd.read_csv(data_dir / "test_ids.csv")["subject_id"].tolist()
    cohort = pd.read_csv(data_dir / "cohort.csv")

    # Map subject_id to icustay_id
    subj_to_stay = dict(zip(cohort["subject_id"], cohort["icustay_id"]))

    pp_cfg = cfg["preprocessing"]
    pipeline = PreprocessingPipeline(
        window_size=pp_cfg["window_size"],
        horizon=pp_cfg["horizon"],
        step_size=pp_cfg["step_size"],
        imputation_strategy=pp_cfg["imputation_strategy"],
        normalization_strategy=pp_cfg["normalization_strategy"],
        add_missing_indicator=pp_cfg["add_missing_indicator"],
        min_completeness=pp_cfg["min_completeness"],
    )

    def load_split(subject_ids, split_name):
        all_windows = []
        for subj_id in subject_ids:
            stay_id = subj_to_stay.get(subj_id)
            if stay_id is None:
                continue
            vitals_path = vitals_dir / f"{stay_id}.parquet"
            labels_path = labels_dir / f"{stay_id}_labels.parquet"
            if not vitals_path.exists():
                continue
            vitals_df = pd.read_parquet(vitals_path)
            labels_df = pd.read_parquet(labels_path) if labels_path.exists() else None
            labels_series = labels_df[outcome] if labels_df is not None and outcome in labels_df else None

            windows = pipeline.transform(
                vitals_df, labels_series,
                metadata={"icustay_id": stay_id, "subject_id": subj_id}
            )
            all_windows.extend(windows)

        logger.info(f"{split_name}: {len(all_windows)} windows")
        return all_windows

    # Fit preprocessing pipeline on training set
    train_vitals = []
    for subj_id in train_ids[:100]:  # fit on first 100 stays for speed
        stay_id = subj_to_stay.get(subj_id)
        if stay_id and (vitals_dir / f"{stay_id}.parquet").exists():
            train_vitals.append(pd.read_parquet(vitals_dir / f"{stay_id}.parquet"))
    if train_vitals:
        pipeline.fit(train_vitals)
        pipeline.save(data_dir)

    train_windows = load_split(train_ids, "train")
    val_windows = load_split(val_ids, "val")
    test_windows = load_split(test_ids, "test")

    return (
        ICUTimeSeriesDataset(train_windows, outcome="label"),
        ICUTimeSeriesDataset(val_windows, outcome="label"),
        ICUTimeSeriesDataset(test_windows, outcome="label"),
    )


def make_debug_datasets(cfg: dict, n_train: int = 500, n_val: int = 100, n_test: int = 100):
    """Create tiny synthetic datasets for debugging."""
    np.random.seed(42)
    channel_names = cfg["channels"]["vital_signs"]

    def make_windows(n, prevalence=0.12):
        windows = []
        for _ in range(n):
            w_size = cfg["preprocessing"]["window_size"]
            n_ch = len(channel_names)
            windows.append({
                "x": np.random.randn(w_size, n_ch).astype(np.float32),
                "label": int(np.random.rand() < prevalence),
                "mortality": int(np.random.rand() < 0.10),
                "sepsis": int(np.random.rand() < 0.15),
                "vasopressor": int(np.random.rand() < 0.18),
                "missing_mask": np.zeros((w_size, n_ch), dtype=bool),
            })
        return windows

    return (
        ICUTimeSeriesDataset(make_windows(n_train), outcome="label"),
        ICUTimeSeriesDataset(make_windows(n_val), outcome="label"),
        ICUTimeSeriesDataset(make_windows(n_test), outcome="label"),
    )


def build_model(model_type: str, cfg: dict):
    """Instantiate model from config."""
    model_cfg = cfg["models"][model_type]
    if model_type == "transformer":
        return build_transformer_detector(model_cfg)
    elif model_type == "lstm_ae":
        return build_lstm_ae(model_cfg)
    elif model_type == "tcn":
        return build_tcn_predictor(model_cfg)
    elif model_type == "early_warning":
        return build_early_warning_model(model_cfg)
    raise ValueError(f"Unknown model type: {model_type}")


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Override config from CLI
    train_cfg = cfg["training"]
    if args.epochs:
        train_cfg["epochs"] = args.epochs
    if args.lr:
        train_cfg["lr"] = args.lr
    if args.batch_size:
        train_cfg["batch_size"] = args.batch_size
    if args.wandb:
        train_cfg["use_wandb"] = True

    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    train_cfg["checkpoint_dir"] = str(output_dir / "checkpoints")

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    if args.debug:
        logger.info("Debug mode: using synthetic data")
        train_ds, val_ds, test_ds = make_debug_datasets(cfg)
    else:
        logger.info(f"Loading processed data from {args.data_dir}")
        train_ds, val_ds, test_ds = load_datasets(args.data_dir, cfg)

    logger.info(
        f"Dataset sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
    )
    n_neg, n_pos = train_ds.class_counts
    logger.info(f"Train class balance: {n_neg} neg, {n_pos} pos "
                f"(prevalence={n_pos/(n_neg+n_pos)*100:.1f}%)")

    # -------------------------------------------------------------------------
    # Build model
    # -------------------------------------------------------------------------
    logger.info(f"Building model: {args.model}")
    model = build_model(args.model, cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    # -------------------------------------------------------------------------
    # Build trainer
    # -------------------------------------------------------------------------
    trainer_config = TrainerConfig(
        model_type=args.model,
        device=train_cfg.get("device", "auto"),
        seed=train_cfg.get("seed", 42),
        lr=train_cfg.get("lr", 3e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-5),
        optimizer=train_cfg.get("optimizer", "adamw"),
        gradient_clip_norm=train_cfg.get("gradient_clip_norm", 1.0),
        scheduler=train_cfg.get("scheduler", "warmup_cosine"),
        warmup_steps=train_cfg.get("warmup_steps", 200),
        min_lr=train_cfg.get("min_lr", 1e-6),
        epochs=train_cfg.get("epochs", 100),
        batch_size=train_cfg.get("batch_size", 64),
        accumulation_steps=train_cfg.get("accumulation_steps", 1),
        mixed_precision=train_cfg.get("mixed_precision", True),
        recon_weight=train_cfg.get("recon_weight", 1.0),
        cls_weight=train_cfg.get("cls_weight", 1.0),
        kl_weight=train_cfg.get("kl_weight", 0.5),
        use_weighted_sampler=train_cfg.get("use_weighted_sampler", True),
        patience=train_cfg.get("patience", 15),
        monitor=train_cfg.get("monitor", "auto"),
        checkpoint_dir=train_cfg["checkpoint_dir"],
        save_best_only=train_cfg.get("save_best_only", True),
        log_interval=train_cfg.get("log_interval", 50),
        use_wandb=train_cfg.get("use_wandb", False),
        project_name=train_cfg.get("project_name", "vitalguard"),
        run_name=f"{args.model}_{Path(args.config).stem}",
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=train_cfg.get("pin_memory", True),
    )

    train_loader, val_loader, test_loader = build_dataloaders(
        train_ds, val_ds, test_ds, trainer_config
    )

    trainer = Trainer(model, trainer_config)

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------
    logger.info(f"Starting training: {trainer_config.epochs} epochs")
    history = trainer.fit(train_loader, val_loader, resume_from=args.checkpoint)

    # -------------------------------------------------------------------------
    # Save history
    # -------------------------------------------------------------------------
    import json
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f, indent=2)
    logger.info(f"Training history saved to {history_path}")

    best_val_auroc = max(history.get("val_auroc", [0]))
    logger.info(f"Best validation AUROC: {best_val_auroc:.4f}")
    logger.info(f"Model saved to: {trainer_config.checkpoint_dir}/best_model.pt")


if __name__ == "__main__":
    main()

