"""
Synthetic Data Generator for VitalGuard pipeline.
Generates realistic synthetic ICU vital signs and labels in the exact format
expected by scripts/train.py — no MIMIC-III access required.

Usage:
    python scripts/generate_synthetic_data.py
    python scripts/generate_synthetic_data.py --n-patients 200
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Normal physiological ranges
VITALS_CONFIG = {
    "heart_rate":  {"mean": 80,   "std": 12,  "clip_min": 30,  "clip_max": 200},
    "sbp":         {"mean": 120,  "std": 15,  "clip_min": 60,  "clip_max": 220},
    "dbp":         {"mean": 80,   "std": 10,  "clip_min": 30,  "clip_max": 140},
    "map":         {"mean": 93,   "std": 11,  "clip_min": 40,  "clip_max": 160},
    "spo2":        {"mean": 97,   "std": 1.5, "clip_min": 70,  "clip_max": 100},
    "resp_rate":   {"mean": 16,   "std": 3,   "clip_min": 4,   "clip_max": 40},
    "temperature": {"mean": 37.0, "std": 0.4, "clip_min": 34,  "clip_max": 41},
    "gcs":         {"mean": 14.0, "std": 1.2, "clip_min": 3,   "clip_max": 15},
}

# What happens when a patient deteriorates
DETERIORATION_DELTAS = {
    "heart_rate":  +25,    # tachycardia
    "sbp":         -30,    # hypotension
    "dbp":         -20,
    "map":         -25,    # MAP drops (renamed from mbp)
    "spo2":        -5,     # desaturation
    "resp_rate":   +10,    # tachypnea
    "temperature": +1.5,   # fever
    "gcs":         -4,     # neurological decline
}


def generate_patient_vitals(icustay_id, n_hours, freq="1h",
                             deteriorating=False, deterioration_onset_frac=0.6,
                             seed=None):
    """Generate a time-indexed vitals DataFrame for one ICU stay."""
    rng = np.random.default_rng(seed)
    start_time = pd.Timestamp("2020-01-01") + pd.Timedelta(hours=icustay_id * 7)
    index = pd.date_range(start=start_time, periods=n_hours, freq=freq)

    data = {}
    for vital, cfg in VITALS_CONFIG.items():
        baseline = rng.normal(cfg["mean"], cfg["std"] * 0.3)
        noise = rng.normal(0, cfg["std"] * 0.15, size=n_hours)
        trend = np.cumsum(rng.normal(0, cfg["std"] * 0.02, size=n_hours))
        series = baseline + noise + trend

        if deteriorating:
            onset = int(n_hours * deterioration_onset_frac)
            ramp = np.zeros(n_hours)
            ramp[onset:] = np.linspace(0, DETERIORATION_DELTAS[vital], n_hours - onset)
            series += ramp

        # Clip to physiologically plausible range
        series = np.clip(series, cfg["clip_min"], cfg["clip_max"])
        data[vital] = series

    df = pd.DataFrame(data, index=index)

    # Add ~15% missing at random (realistic for ICU charts)
    mask = rng.random(df.shape) < 0.15
    df[mask] = np.nan

    return df


def generate_labels(icustay_id, n_hours, freq="1h",
                    deteriorating=False, deterioration_onset_frac=0.6,
                    start_time=None):
    """Generate outcome labels for one ICU stay."""
    if start_time is None:
        start_time = pd.Timestamp("2020-01-01") + pd.Timedelta(hours=icustay_id * 7)
    index = pd.date_range(start=start_time, periods=n_hours, freq=freq)
    onset = int(n_hours * deterioration_onset_frac)

    mortality     = np.ones(n_hours, dtype=int)  if deteriorating else np.zeros(n_hours, dtype=int)
    sepsis        = np.zeros(n_hours, dtype=int)
    deterioration = np.zeros(n_hours, dtype=int)

    if deteriorating:
        sepsis[onset:]        = 1
        deterioration[onset:] = 1

    return pd.DataFrame(
        {"mortality": mortality, "sepsis": sepsis, "deterioration": deterioration},
        index=index,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic ICU data")
    parser.add_argument("--n-patients",   type=int,   default=150)
    parser.add_argument("--output-dir",   type=str,   default="data/processed/")
    parser.add_argument("--pos-fraction", type=float, default=0.30,
                        help="Fraction of deteriorating patients (default 30%%)")
    parser.add_argument("--min-hours",    type=int,   default=24)
    parser.add_argument("--max-hours",    type=int,   default=72)
    parser.add_argument("--freq",         type=str,   default="1h")
    parser.add_argument("--seed",         type=int,   default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    output_dir = Path(args.output_dir)
    vitals_dir = output_dir / "vitals"
    labels_dir = output_dir / "labels"
    for d in [output_dir, vitals_dir, labels_dir]:
        d.mkdir(parents=True, exist_ok=True)

    n     = args.n_patients
    n_pos = int(n * args.pos_fraction)
    n_neg = n - n_pos

    logger.info(f"Generating {n} patients ({n_pos} deteriorating, {n_neg} stable)...")

    subject_ids        = list(range(1000, 1000 + n))
    icustay_ids        = list(range(200000, 200000 + n))
    deteriorating_flags = [True] * n_pos + [False] * n_neg
    rng.shuffle(deteriorating_flags)

    icu_intimes = [
        pd.Timestamp("2020-01-01") + pd.Timedelta(hours=i * 7) for i in range(n)
    ]
    stay_hours  = rng.integers(args.min_hours, args.max_hours + 1, size=n)
    icu_outtimes = [
        icu_intimes[i] + pd.Timedelta(hours=int(stay_hours[i])) for i in range(n)
    ]

    # ── Cohort CSV ──────────────────────────────────────────────────────────
    cohort = pd.DataFrame({
        "subject_id":           subject_ids,
        "icustay_id":           icustay_ids,
        "hadm_id":              [300000 + i for i in range(n)],
        "icu_intime":           icu_intimes,
        "icu_outtime":          icu_outtimes,
        "los_hours":            stay_hours,
        "age":                  rng.integers(18, 90, size=n),
        "gender":               rng.choice(["M", "F"], size=n),
        "icu_type":             rng.choice(["MICU", "SICU", "CCU"], size=n),
        "hospital_expire_flag": [int(d) for d in deteriorating_flags],
    })
    cohort.to_csv(output_dir / "cohort.csv", index=False)
    logger.info("cohort.csv saved")

    # ── Train / Val / Test split (70 / 15 / 15) ─────────────────────────────
    idx = np.arange(n)
    rng.shuffle(idx)
    t_end = int(n * 0.70)
    v_end = int(n * 0.85)
    train_idx, val_idx, test_idx = idx[:t_end], idx[t_end:v_end], idx[v_end:]

    pd.DataFrame({"subject_id": [subject_ids[i] for i in train_idx]}).to_csv(
        output_dir / "train_ids.csv", index=False)
    pd.DataFrame({"subject_id": [subject_ids[i] for i in val_idx]}).to_csv(
        output_dir / "val_ids.csv", index=False)
    pd.DataFrame({"subject_id": [subject_ids[i] for i in test_idx]}).to_csv(
        output_dir / "test_ids.csv", index=False)
    logger.info(f"Splits: {len(train_idx)} train | {len(val_idx)} val | {len(test_idx)} test")

    # ── Per-patient vitals + labels ──────────────────────────────────────────
    for i, (subj_id, stay_id, det, n_hrs) in enumerate(
        zip(subject_ids, icustay_ids, deteriorating_flags, stay_hours)
    ):
        vitals_df = generate_patient_vitals(
            icustay_id=stay_id,
            n_hours=int(n_hrs),
            freq=args.freq,
            deteriorating=det,
            seed=args.seed + i,
        )
        vitals_df.to_parquet(vitals_dir / f"{stay_id}.parquet")

        labels_df = generate_labels(
            icustay_id=stay_id,
            n_hours=int(n_hrs),
            freq=args.freq,
            deteriorating=det,
            start_time=vitals_df.index[0],
        )
        labels_df.to_parquet(labels_dir / f"{stay_id}_labels.parquet")

        if (i + 1) % 30 == 0:
            logger.info(f"  {i+1}/{n} patients done...")

    logger.info("=" * 55)
    logger.info("Synthetic data generation complete!")
    logger.info(f"  cohort.csv          — {n} patients")
    logger.info(f"  train_ids.csv       — {len(train_idx)} patients")
    logger.info(f"  val_ids.csv         — {len(val_idx)} patients")
    logger.info(f"  test_ids.csv        — {len(test_idx)} patients")
    logger.info(f"  vitals/             — {n} .parquet files")
    logger.info(f"  labels/             — {n} .parquet files")
    logger.info(f"  Positive rate       — {n_pos}/{n} = {n_pos/n*100:.0f}%")
    logger.info("=" * 55)
    logger.info("Now run: python scripts/train.py --config configs/mimic_config.yaml")


if __name__ == "__main__":
    main()
