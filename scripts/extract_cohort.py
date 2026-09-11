"""
MIMIC ICU Cohort Extraction Script.

Usage:
    python scripts/extract_cohort.py \
        --config configs/mimic_config.yaml \
        --output-dir data/processed/ \
        --mimic-version 3

Requires MIMIC-III CSV files or a running MIMIC-III PostgreSQL database.
See https://mimic.mit.edu/docs/gettingstarted/ for access instructions.
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.mimic_loader import (
    ICUCohortExtractor,
    MIMICConnector,
    MIMICDatasetBuilder,
    VitalSignExtractor,
    LabValueExtractor,
)
from src.data.sepsis_labeler import SepsisLabeler
from src.data.outcome_labeler import OutcomeLabeler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("extract_cohort.log")],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract MIMIC ICU cohort")
    parser.add_argument("--config", type=str, default="configs/mimic_config.yaml")
    parser.add_argument("--output-dir", type=str, default="data/processed/")
    parser.add_argument("--mimic-version", type=int, default=3, choices=[3, 4])
    parser.add_argument("--backend", type=str, default=None,
                        help="Override config backend (csv or postgresql)")
    parser.add_argument("--csv-dir", type=str, default=None,
                        help="Override config MIMIC CSV directory")
    parser.add_argument("--n-stays", type=int, default=None,
                        help="Limit to first N ICU stays (for testing)")
    parser.add_argument("--resample-freq", type=str, default="1h")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run without saving (test data access only)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    cohort_cfg = cfg["cohort"]

    # Build connector
    backend = args.backend or data_cfg["backend"]
    csv_dir = args.csv_dir or data_cfg["csv_dir"]
    schema = "mimiciii" if args.mimic_version == 3 else "mimiciv"

    connector = MIMICConnector(
        backend=backend,
        db_uri=data_cfg.get("db_uri", ""),
        csv_dir=csv_dir,
        schema=schema,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Extract cohort
    # -------------------------------------------------------------------------
    logger.info("Step 1: Extracting ICU cohort...")
    cohort_extractor = ICUCohortExtractor(connector)
    cohort = cohort_extractor.extract(
        min_age=cohort_cfg["min_age"],
        min_los_hours=cohort_cfg["min_los_hours"],
        exclude_csru=cohort_cfg["exclude_csru"],
        first_admission_only=cohort_cfg["first_admission_only"],
    )

    if args.n_stays:
        cohort = cohort.head(args.n_stays)
        logger.info(f"Limited to first {args.n_stays} stays for testing")

    logger.info(f"Cohort size: {len(cohort)} ICU stays")
    logger.info(f"Hospital mortality: {cohort['hospital_expire_flag'].mean()*100:.1f}%")
    logger.info(f"ICU type distribution:\n{cohort['icu_type'].value_counts()}")

    # Train/val/test split
    train, val, test = cohort_extractor.train_val_test_split(
        cohort,
        train_frac=cohort_cfg["train_fraction"],
        val_frac=cohort_cfg["val_fraction"],
        seed=cohort_cfg["seed"],
    )

    if args.dry_run:
        logger.info("Dry run complete. Exiting without saving.")
        return

    # Save cohort
    cohort.to_csv(output_dir / "cohort.csv", index=False)
    train[["subject_id"]].to_csv(output_dir / "train_ids.csv", index=False)
    val[["subject_id"]].to_csv(output_dir / "val_ids.csv", index=False)
    test[["subject_id"]].to_csv(output_dir / "test_ids.csv", index=False)
    logger.info(f"Cohort saved: {len(train)} train | {len(val)} val | {len(test)} test")

    # -------------------------------------------------------------------------
    # Step 2: Extract vital signs
    # -------------------------------------------------------------------------
    logger.info("Step 2: Extracting vital signs...")
    vital_extractor = VitalSignExtractor(connector)
    vitals_dir = output_dir / "vitals"
    vitals_dir.mkdir(exist_ok=True)

    for _, row in cohort.iterrows():
        try:
            vitals_df = vital_extractor.extract_for_stay(
                icustay_id=row["icustay_id"],
                icu_intime=row["icu_intime"],
                icu_outtime=row["icu_outtime"],
                resample_freq=args.resample_freq,
            )
            if not vitals_df.empty:
                vitals_df.to_parquet(vitals_dir / f"{row['icustay_id']}.parquet")
        except Exception as e:
            logger.error(f"Error extracting vitals for stay {row['icustay_id']}: {e}")

    logger.info(f"Vitals extracted: {len(list(vitals_dir.glob('*.parquet')))} stays")

    # -------------------------------------------------------------------------
    # Step 3: Extract lab values
    # -------------------------------------------------------------------------
    logger.info("Step 3: Extracting lab values...")
    lab_extractor = LabValueExtractor(connector)
    labs_dir = output_dir / "labs"
    labs_dir.mkdir(exist_ok=True)

    for _, row in cohort.iterrows():
        try:
            labs_df = lab_extractor.extract_for_stay(
                hadm_id=row["hadm_id"],
                icu_intime=row["icu_intime"],
                icu_outtime=row["icu_outtime"],
                resample_freq=args.resample_freq,
            )
            if not labs_df.empty:
                labs_df.to_parquet(labs_dir / f"{row['icustay_id']}.parquet")
        except Exception as e:
            logger.error(f"Error extracting labs for stay {row['icustay_id']}: {e}")

    # -------------------------------------------------------------------------
    # Step 4: Generate labels
    # -------------------------------------------------------------------------
    logger.info("Step 4: Generating clinical outcome labels...")
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(exist_ok=True)

    # Load vitals for labeling
    vitals_dict = {}
    for f in vitals_dir.glob("*.parquet"):
        stay_id = int(f.stem)
        import pandas as pd
        vitals_dict[stay_id] = pd.read_parquet(f)

    # Sepsis labels
    logger.info("Generating Sepsis-3 labels...")
    labs_dict = {}
    for f in labs_dir.glob("*.parquet"):
        import pandas as pd
        labs_dict[int(f.stem)] = pd.read_parquet(f)

    sepsis_labeler = SepsisLabeler(connector=connector)
    sepsis_df = sepsis_labeler.process_cohort(cohort, vitals_dict, labs_dict)
    sepsis_df.to_csv(output_dir / "sepsis_labels.csv", index=False)
    logger.info(
        f"Sepsis labels: {sepsis_df['sepsis'].sum()} positive "
        f"({sepsis_df['sepsis'].mean()*100:.1f}%)"
    )

    # Other outcome labels
    outcome_labeler = OutcomeLabeler(connector=connector)
    all_outcome_labels = outcome_labeler.process_cohort(
        cohort, vitals_dict, save_dir=str(labels_dir)
    )

    logger.info(f"Outcome labels generated for {len(all_outcome_labels)} stays")
    logger.info("Cohort extraction complete!")
    logger.info(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
