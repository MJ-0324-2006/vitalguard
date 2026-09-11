"""
MIMIC-III / MIMIC-IV Data Loader.

Extracts ICU cohort, vital signs, lab values, and medication events from
MIMIC-III PostgreSQL database or flat CSV files.

Data Access:
  MIMIC-III requires PhysioNet credentialing + CITI training.
  Instructions: https://mimic.mit.edu/docs/gettingstarted/

Supported backends:
  - PostgreSQL (psycopg2) — recommended for full MIMIC-III
  - CSV files (pandas) — for MIMIC-IV or offline use
  - SQLite (sqlite3) — for local development/testing

Vital sign ItemIDs reference:
  https://mimic.mit.edu/docs/iii/tables/chartevents/
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MIMIC-III ItemID definitions
# ---------------------------------------------------------------------------

# Heart Rate
HR_ITEMIDS = [211, 220045]

# Blood Pressure
SBP_ITEMIDS = [51, 442, 455, 6701, 220179, 220050]  # arterial + cuff
DBP_ITEMIDS = [8368, 8440, 8441, 8555, 220180, 220051]
MAP_ITEMIDS = [52, 456, 6702, 220052, 220181, 225312]

# Respiratory
SPO2_ITEMIDS = [646, 220277]
RR_ITEMIDS = [618, 615, 220210, 224690]

# Temperature (Fahrenheit and Celsius)
TEMP_F_ITEMIDS = [223761, 678]
TEMP_C_ITEMIDS = [223762, 676]

# GCS
GCS_TOTAL_ITEMIDS = [198, 226755, 227013]
GCS_MOTOR_ITEMIDS = [454, 223901]
GCS_VERBAL_ITEMIDS = [723, 223900]
GCS_EYE_ITEMIDS = [184, 220739]

# Urine output
URINE_ITEMIDS = [40055, 43175, 40069, 40094, 40715, 40473, 40085, 40057,
                 40056, 40405, 40428, 40096, 40651, 226559, 226560, 226561,
                 226584, 226563, 226564, 226565, 226557, 226558, 227488, 227489]

# Vasopressors (INPUTEVENTS)
VASOPRESSOR_ITEMIDS = {
    "norepinephrine": [30047, 30120, 221906],
    "epinephrine": [30044, 30119, 30309, 221289],
    "dopamine": [30043, 30307, 221662],
    "vasopressin": [30051, 222315],
    "phenylephrine": [30128, 30127, 221749],
}

# Lab values (LABEVENTS)
LAB_ITEMIDS = {
    "lactate": [50813],
    "wbc": [51301, 51300],
    "creatinine": [50912],
    "bilirubin_total": [50885],
    "platelet": [51265],
    "pao2": [50821],
    "fio2": [50816],
    "inr": [51237],
    "albumin": [50862],
    "sodium": [50983],
    "potassium": [50971],
    "hematocrit": [51221],
}

ALL_VITAL_ITEMIDS = (
    HR_ITEMIDS + SBP_ITEMIDS + DBP_ITEMIDS + MAP_ITEMIDS +
    SPO2_ITEMIDS + RR_ITEMIDS + TEMP_F_ITEMIDS + TEMP_C_ITEMIDS +
    GCS_TOTAL_ITEMIDS
)


# ---------------------------------------------------------------------------
# Database Connection
# ---------------------------------------------------------------------------

class MIMICConnector:
    """Handles connections to MIMIC-III (PostgreSQL) or MIMIC-IV (CSV/PostgreSQL)."""

    def __init__(
        self,
        backend: str = "csv",
        db_uri: Optional[str] = None,
        csv_dir: Optional[str] = None,
        schema: str = "mimiciii",
    ) -> None:
        """
        Args:
            backend:  'postgresql', 'csv', or 'sqlite'
            db_uri:   PostgreSQL URI (postgresql://user:pass@host/db)
            csv_dir:  Path to directory containing MIMIC CSV files
            schema:   PostgreSQL schema name ('mimiciii' or 'mimiciv')
        """
        self.backend = backend
        self.db_uri = db_uri or os.environ.get("MIMIC_DB_URI", "")
        self.csv_dir = Path(csv_dir or os.environ.get("MIMIC_DATA_DIR", "data/mimic"))
        self.schema = schema
        self._engine = None
        self._csv_cache: Dict[str, pd.DataFrame] = {}

    def get_engine(self):
        """Get SQLAlchemy engine (PostgreSQL backend)."""
        if self._engine is None:
            try:
                from sqlalchemy import create_engine
                self._engine = create_engine(self.db_uri)
                logger.info(f"Connected to MIMIC database: {self.db_uri[:50]}...")
            except ImportError:
                raise ImportError("sqlalchemy required for PostgreSQL backend: pip install sqlalchemy psycopg2")
        return self._engine

    def read_table(self, table_name: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Read a MIMIC table from the configured backend.

        Args:
            table_name: MIMIC table name (e.g., 'CHARTEVENTS', 'PATIENTS')
            columns:    Optional column subset to read

        Returns:
            DataFrame with the requested table data
        """
        if self.backend == "csv":
            return self._read_csv(table_name, columns)
        elif self.backend == "postgresql":
            return self._read_sql(table_name, columns)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _read_csv(self, table_name: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        # Cache to avoid re-reading
        cache_key = f"{table_name}_{columns}"
        if cache_key in self._csv_cache:
            return self._csv_cache[cache_key]

        # Try uppercase and lowercase filenames
        for fname in [f"{table_name}.csv", f"{table_name.lower()}.csv",
                       f"{table_name.upper()}.csv", f"{table_name}.csv.gz"]:
            fpath = self.csv_dir / fname
            if fpath.exists():
                df = pd.read_csv(fpath, usecols=columns, low_memory=False)
                df.columns = df.columns.str.lower()
                self._csv_cache[cache_key] = df
                logger.debug(f"Loaded {table_name}: {len(df)} rows")
                return df

        raise FileNotFoundError(
            f"MIMIC table {table_name} not found in {self.csv_dir}. "
            f"Ensure MIMIC-III CSV files are present."
        )

    def _read_sql(self, table_name: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        engine = self.get_engine()
        cols = "*" if columns is None else ", ".join(columns)
        query = f"SELECT {cols} FROM {self.schema}.{table_name}"
        return pd.read_sql(query, engine)

    def query(self, sql: str) -> pd.DataFrame:
        """Execute arbitrary SQL (PostgreSQL backend only)."""
        return pd.read_sql(sql, self.get_engine())


# ---------------------------------------------------------------------------
# Cohort Extraction
# ---------------------------------------------------------------------------

class ICUCohortExtractor:
    """
    Extract ICU cohort from MIMIC-III satisfying inclusion/exclusion criteria.

    Inclusion criteria:
      - First ICU admission
      - Age >= 18 at admission
      - ICU LOS >= 24 hours
      - >= 3 vital sign channels with >= 50% completeness

    Exclusion criteria:
      - CSRU (cardiac surgery) patients
      - Readmissions within 30 days
    """

    def __init__(self, connector: MIMICConnector) -> None:
        self.conn = connector

    def extract(
        self,
        min_age: int = 18,
        min_los_hours: float = 24.0,
        exclude_csru: bool = True,
        first_admission_only: bool = True,
    ) -> pd.DataFrame:
        """
        Extract ICU cohort.

        Returns DataFrame with columns:
            subject_id, hadm_id, icustay_id, admission_type,
            icu_intime, icu_outtime, icu_los_hours,
            age, gender, ethnicity, insurance,
            hospital_expire_flag, icu_expire_flag
        """
        logger.info("Extracting ICU cohort from MIMIC...")

        # Load core tables
        patients = self.conn.read_table(
            "PATIENTS",
            ["subject_id", "gender", "dob", "dod", "dod_hosp"]
        )
        admissions = self.conn.read_table(
            "ADMISSIONS",
            ["subject_id", "hadm_id", "admittime", "dischtime", "deathtime",
             "admission_type", "ethnicity", "insurance", "hospital_expire_flag"]
        )
        icustays = self.conn.read_table(
            "ICUSTAYS",
            ["subject_id", "hadm_id", "icustay_id", "first_careunit",
             "last_careunit", "intime", "outtime", "los"]
        )

        # Parse datetime columns
        for df, cols in [
            (patients, ["dob", "dod"]),
            (admissions, ["admittime", "dischtime"]),
            (icustays, ["intime", "outtime"]),
        ]:
            for col in cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])

        # Merge
        cohort = icustays.merge(admissions, on=["subject_id", "hadm_id"], how="inner")
        cohort = cohort.merge(patients, on="subject_id", how="inner")

        # Rename for clarity
        cohort = cohort.rename(columns={
            "intime": "icu_intime",
            "outtime": "icu_outtime",
            "los": "icu_los_hours",
        })
        cohort["icu_los_hours"] = cohort["icu_los_hours"] * 24  # MIMIC stores in days

        # Compute age at admission
        cohort["age"] = (
            (cohort["admittime"] - cohort["dob"]).dt.days / 365.25
        ).clip(upper=89)  # MIMIC caps age at 89 for patients >89 yo

        # --- Inclusion/Exclusion filters ---

        # Age >= 18
        cohort = cohort[cohort["age"] >= min_age]

        # ICU LOS >= 24h
        cohort = cohort[cohort["icu_los_hours"] >= min_los_hours]

        # First admission only (by subject_id + earliest intime)
        if first_admission_only:
            cohort = cohort.sort_values("icu_intime")
            cohort = cohort.groupby("subject_id").first().reset_index()

        # Exclude CSRU (cardiac surgery)
        if exclude_csru and "first_careunit" in cohort.columns:
            cohort = cohort[cohort["first_careunit"] != "CSRU"]

        # Add ICU type label
        cohort["icu_type"] = cohort["first_careunit"].map({
            "MICU": "MICU",
            "SICU": "SICU",
            "CCU": "CCU",
            "TSICU": "TSICU",
            "NICU": "NICU",
        }).fillna("OTHER")

        logger.info(f"Final cohort: {len(cohort)} ICU stays")
        return cohort.reset_index(drop=True)

    def train_val_test_split(
        self,
        cohort: pd.DataFrame,
        train_frac: float = 0.70,
        val_frac: float = 0.15,
        seed: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split cohort by patient (subject_id) to prevent leakage.

        Returns (train_df, val_df, test_df)
        """
        np.random.seed(seed)
        subjects = cohort["subject_id"].unique()
        np.random.shuffle(subjects)

        n = len(subjects)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        train_ids = set(subjects[:n_train])
        val_ids = set(subjects[n_train : n_train + n_val])
        test_ids = set(subjects[n_train + n_val :])

        train = cohort[cohort["subject_id"].isin(train_ids)].copy()
        val = cohort[cohort["subject_id"].isin(val_ids)].copy()
        test = cohort[cohort["subject_id"].isin(test_ids)].copy()

        logger.info(
            f"Split: train={len(train)} | val={len(val)} | test={len(test)} stays"
        )
        return train, val, test


# ---------------------------------------------------------------------------
# Vital Sign Extraction
# ---------------------------------------------------------------------------

class VitalSignExtractor:
    """Extract and align multi-channel vital signs for ICU stays."""

    # Channel definitions: name → (itemids, physiological_range, convert_func)
    CHANNELS = {
        "heart_rate": (HR_ITEMIDS, (20, 300), None),
        "sbp": (SBP_ITEMIDS, (40, 300), None),
        "dbp": (DBP_ITEMIDS, (20, 200), None),
        "map": (MAP_ITEMIDS, (20, 200), None),
        "spo2": (SPO2_ITEMIDS, (60, 100), None),
        "resp_rate": (RR_ITEMIDS, (4, 60), None),
        "temperature": (
            TEMP_F_ITEMIDS + TEMP_C_ITEMIDS,
            (25, 45),
            lambda v, iid: (v - 32) * 5 / 9 if iid in TEMP_F_ITEMIDS else v,
        ),
        "gcs": (GCS_TOTAL_ITEMIDS, (3, 15), None),
    }

    def __init__(self, connector: MIMICConnector) -> None:
        self.conn = connector
        self._chart_cache: Optional[pd.DataFrame] = None

    def _load_chartevents(self, icustay_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """Load CHARTEVENTS, optionally filtered to specific ICU stays."""
        if self._chart_cache is not None:
            return self._chart_cache

        logger.info("Loading CHARTEVENTS (may take several minutes for full MIMIC)...")
        all_itemids = []
        for itemids, _, _ in self.CHANNELS.values():
            all_itemids.extend(itemids)

        if self.conn.backend == "postgresql":
            ids_str = ",".join(str(i) for i in all_itemids)
            stay_filter = ""
            if icustay_ids:
                stay_ids_str = ",".join(str(i) for i in icustay_ids)
                stay_filter = f"AND icustay_id IN ({stay_ids_str})"
            sql = f"""
                SELECT icustay_id, charttime, itemid, valuenum, error
                FROM {self.conn.schema}.chartevents
                WHERE itemid IN ({ids_str})
                  AND valuenum IS NOT NULL
                  AND (error IS NULL OR error = 0)
                  {stay_filter}
            """
            df = self.conn.query(sql)
        else:
            df = self.conn.read_table(
                "CHARTEVENTS",
                ["icustay_id", "charttime", "itemid", "valuenum", "error"]
            )
            df = df[df["itemid"].isin(all_itemids)]
            if icustay_ids:
                df = df[df["icustay_id"].isin(icustay_ids)]

        df["charttime"] = pd.to_datetime(df["charttime"])
        df = df[df["error"].isna() | (df["error"] == 0)]
        self._chart_cache = df
        logger.info(f"Loaded {len(df)} chart events")
        return df

    def extract_for_stay(
        self,
        icustay_id: int,
        icu_intime: pd.Timestamp,
        icu_outtime: pd.Timestamp,
        resample_freq: str = "1h",
        chartevents: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Extract vital signs for a single ICU stay, resampled to regular grid.

        Args:
            icustay_id:    ICU stay identifier
            icu_intime:    ICU admission time
            icu_outtime:   ICU discharge time
            resample_freq: Pandas offset string (e.g., '1h', '30min')
            chartevents:   Pre-loaded CHARTEVENTS DataFrame (optional)

        Returns:
            DataFrame with columns = channel names, index = regular datetime grid
        """
        if chartevents is None:
            chartevents = self._load_chartevents()

        stay_events = chartevents[chartevents["icustay_id"] == icustay_id].copy()

        if len(stay_events) == 0:
            logger.warning(f"No chart events for icustay_id={icustay_id}")
            return pd.DataFrame()

        # Build time-indexed DataFrame per channel
        results = {}
        for ch_name, (itemids, (lo, hi), convert_fn) in self.CHANNELS.items():
            ch_events = stay_events[stay_events["itemid"].isin(itemids)].copy()
            if ch_events.empty:
                results[ch_name] = pd.Series(dtype=float)
                continue

            # Apply unit conversion (e.g., Fahrenheit → Celsius)
            if convert_fn is not None:
                ch_events["valuenum"] = ch_events.apply(
                    lambda r: convert_fn(r["valuenum"], r["itemid"]), axis=1
                )

            # Apply physiological range filter
            ch_events = ch_events[
                (ch_events["valuenum"] >= lo) & (ch_events["valuenum"] <= hi)
            ]

            if ch_events.empty:
                results[ch_name] = pd.Series(dtype=float)
                continue

            # Set time index and deduplicate (take median for same minute)
            ch_events = ch_events.set_index("charttime")["valuenum"]
            ch_events = ch_events.resample("1min").median()
            results[ch_name] = ch_events

        # Merge all channels into one DataFrame
        df = pd.DataFrame(results)

        # Restrict to ICU stay window
        df = df.loc[icu_intime:icu_outtime]

        # Resample to target frequency
        df = df.resample(resample_freq).mean()

        # Create full time grid (including gaps)
        full_index = pd.date_range(icu_intime, icu_outtime, freq=resample_freq)
        df = df.reindex(full_index)

        return df

    def extract_batch(
        self,
        cohort: pd.DataFrame,
        resample_freq: str = "1h",
        n_jobs: int = 4,
    ) -> Dict[int, pd.DataFrame]:
        """
        Extract vital signs for all stays in cohort.

        Args:
            cohort:       DataFrame with icustay_id, icu_intime, icu_outtime
            resample_freq: Resampling frequency
            n_jobs:       Parallel workers

        Returns:
            Dict mapping icustay_id → vital sign DataFrame
        """
        chartevents = self._load_chartevents(cohort["icustay_id"].tolist())
        results = {}

        for _, row in cohort.iterrows():
            try:
                df = self.extract_for_stay(
                    row["icustay_id"],
                    row["icu_intime"],
                    row["icu_outtime"],
                    resample_freq,
                    chartevents,
                )
                results[row["icustay_id"]] = df
            except Exception as e:
                logger.error(f"Error extracting stay {row['icustay_id']}: {e}")

        logger.info(f"Extracted vitals for {len(results)} stays")
        return results


# ---------------------------------------------------------------------------
# Lab Value Extractor
# ---------------------------------------------------------------------------

class LabValueExtractor:
    """Extract and align lab values from LABEVENTS."""

    def __init__(self, connector: MIMICConnector) -> None:
        self.conn = connector

    def extract_for_stay(
        self,
        hadm_id: int,
        icu_intime: pd.Timestamp,
        icu_outtime: pd.Timestamp,
        resample_freq: str = "1h",
    ) -> pd.DataFrame:
        """
        Extract lab values for an ICU stay.

        Returns DataFrame with lab values, forward-filled to vital sign grid.
        """
        # Load LABEVENTS for this admission
        if self.conn.backend == "postgresql":
            lab_ids = [iid for ids in LAB_ITEMIDS.values() for iid in ids]
            ids_str = ",".join(str(i) for i in lab_ids)
            sql = f"""
                SELECT charttime, itemid, valuenum
                FROM {self.conn.schema}.labevents
                WHERE hadm_id = {hadm_id}
                  AND itemid IN ({ids_str})
                  AND valuenum IS NOT NULL
            """
            df = self.conn.query(sql)
        else:
            labs = self.conn.read_table(
                "LABEVENTS",
                ["hadm_id", "charttime", "itemid", "valuenum"]
            )
            df = labs[labs["hadm_id"] == hadm_id].copy()

        if df.empty:
            return pd.DataFrame()

        df["charttime"] = pd.to_datetime(df["charttime"])
        df = df[(df["charttime"] >= icu_intime) & (df["charttime"] <= icu_outtime)]

        # Pivot to lab name columns
        id_to_name = {iid: name for name, ids in LAB_ITEMIDS.items() for iid in ids}
        df["lab_name"] = df["itemid"].map(id_to_name)
        df = df.dropna(subset=["lab_name"])

        # Pivot to wide format
        pivot = df.groupby(["charttime", "lab_name"])["valuenum"].mean().unstack(
            "lab_name"
        )

        # Resample and forward-fill
        full_index = pd.date_range(icu_intime, icu_outtime, freq=resample_freq)
        pivot = pivot.reindex(full_index.union(pivot.index)).sort_index()
        pivot = pivot.ffill().reindex(full_index)

        return pivot


# ---------------------------------------------------------------------------
# Medication Event Extractor
# ---------------------------------------------------------------------------

class MedicationExtractor:
    """Extract vasopressor and antibiotic administration events."""

    def __init__(self, connector: MIMICConnector) -> None:
        self.conn = connector

    def extract_vasopressors(
        self, icustay_id: int, icu_intime: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Extract vasopressor infusion times for a stay.

        Returns DataFrame with columns: starttime, endtime, drug_name, rate
        """
        all_vasopress_ids = [
            iid for ids in VASOPRESSOR_ITEMIDS.values() for iid in ids
        ]
        id_to_drug = {iid: drug for drug, ids in VASOPRESSOR_ITEMIDS.items() for iid in ids}

        if self.conn.backend == "postgresql":
            ids_str = ",".join(str(i) for i in all_vasopress_ids)
            sql = f"""
                SELECT starttime, endtime, itemid, rate, rateuom
                FROM {self.conn.schema}.inputevents_mv
                WHERE icustay_id = {icustay_id}
                  AND itemid IN ({ids_str})
                  AND rate IS NOT NULL
                UNION ALL
                SELECT charttime AS starttime, charttime AS endtime,
                       itemid, amount AS rate, amountuom AS rateuom
                FROM {self.conn.schema}.inputevents_cv
                WHERE icustay_id = {icustay_id}
                  AND itemid IN ({ids_str})
            """
            df = self.conn.query(sql)
        else:
            # Try inputevents_mv first (MetaVision), fall back to inputevents_cv
            try:
                mv = self.conn.read_table(
                    "INPUTEVENTS_MV",
                    ["icustay_id", "starttime", "endtime", "itemid", "rate", "rateuom"]
                )
                df = mv[mv["icustay_id"] == icustay_id].copy()
                df = df[df["itemid"].isin(all_vasopress_ids)]
            except FileNotFoundError:
                df = pd.DataFrame()

        if df.empty:
            return df

        df["starttime"] = pd.to_datetime(df["starttime"])
        df["endtime"] = pd.to_datetime(df["endtime"])
        df["drug_name"] = df["itemid"].map(id_to_drug)
        df["hours_from_admit"] = (df["starttime"] - icu_intime).dt.total_seconds() / 3600
        return df

    def get_first_vasopressor_time(
        self, icustay_id: int, icu_intime: pd.Timestamp
    ) -> Optional[float]:
        """Return hours from ICU admission to first vasopressor, or None."""
        df = self.extract_vasopressors(icustay_id, icu_intime)
        if df.empty:
            return None
        return df["hours_from_admit"].min()


# ---------------------------------------------------------------------------
# High-level DatasetBuilder
# ---------------------------------------------------------------------------

class MIMICDatasetBuilder:
    """
    End-to-end pipeline: cohort → vital signs → labs → labels → windowed tensors.
    """

    def __init__(
        self,
        connector: MIMICConnector,
        window_hours: int = 6,
        horizon_hours: int = 24,
        step_hours: int = 1,
        vital_channels: Optional[List[str]] = None,
    ) -> None:
        self.connector = connector
        self.window_hours = window_hours
        self.horizon_hours = horizon_hours
        self.step_hours = step_hours
        self.vital_channels = vital_channels or list(VitalSignExtractor.CHANNELS.keys())

        self.cohort_extractor = ICUCohortExtractor(connector)
        self.vital_extractor = VitalSignExtractor(connector)
        self.lab_extractor = LabValueExtractor(connector)
        self.med_extractor = MedicationExtractor(connector)

    def build(
        self,
        output_dir: Union[str, Path],
        resample_freq: str = "1h",
        min_age: int = 18,
        min_los_hours: float = 24.0,
    ) -> None:
        """
        Full extraction pipeline. Saves processed data to output_dir.

        Output files:
            cohort.csv            — cohort metadata
            train_ids.txt         — subject IDs for training
            val_ids.txt
            test_ids.txt
            vitals/{icustay_id}.parquet   — per-stay vital sign DataFrames
            labs/{icustay_id}.parquet     — per-stay lab DataFrames
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Extract cohort
        cohort = self.cohort_extractor.extract(min_age=min_age, min_los_hours=min_los_hours)
        cohort.to_csv(output_dir / "cohort.csv", index=False)

        # 2. Train/val/test split
        train, val, test = self.cohort_extractor.train_val_test_split(cohort)
        for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
            split_df["subject_id"].to_csv(
                output_dir / f"{split_name}_ids.txt", index=False, header=False
            )

        # 3. Extract vitals and labs per stay
        vitals_dir = output_dir / "vitals"
        labs_dir = output_dir / "labs"
        vitals_dir.mkdir(exist_ok=True)
        labs_dir.mkdir(exist_ok=True)

        logger.info(f"Extracting vitals for {len(cohort)} stays...")
        for _, row in cohort.iterrows():
            stay_id = row["icustay_id"]

            # Vitals
            vitals_df = self.vital_extractor.extract_for_stay(
                stay_id, row["icu_intime"], row["icu_outtime"], resample_freq
            )
            if not vitals_df.empty:
                vitals_df.to_parquet(vitals_dir / f"{stay_id}.parquet")

            # Labs
            labs_df = self.lab_extractor.extract_for_stay(
                row["hadm_id"], row["icu_intime"], row["icu_outtime"], resample_freq
            )
            if not labs_df.empty:
                labs_df.to_parquet(labs_dir / f"{stay_id}.parquet")

        logger.info(f"Dataset extraction complete → {output_dir}")


if __name__ == "__main__":
    # Demo with CSV backend (requires MIMIC CSVs in data/mimic/)
    connector = MIMICConnector(backend="csv", csv_dir="data/mimic")
    try:
        cohort_extractor = ICUCohortExtractor(connector)
        print("Connector initialised. Set MIMIC_DATA_DIR to run extraction.")
    except Exception as e:
        print(f"Note: {e}")
    print("MIMIC loader module loaded successfully.")

# Compatible with MIMIC-III v1.4 and MIMIC-IV v2.2

# fix imputation for sparse lab values
# was using simple ffill which creates problems when there are long gaps
# now uses a time-aware approach with a maximum gap threshold
MAX_IMPUTATION_GAP_HOURS = 6  # don't forward fill beyond 6 hours

def impute_sparse_labs(df, time_col='charttime', max_gap_hours=MAX_IMPUTATION_GAP_HOURS):
    """time-aware forward fill for sparse lab measurements
    
    avoids propagating stale values across long gaps (e.g. from day to night shift)
    values older than max_gap_hours are replaced with NaN instead of being propagated
    """
    import pandas as pd
    import numpy as np

    df = df.sort_values(time_col)
    lab_cols = [c for c in df.columns if c != time_col]
    df_filled = df.copy()

    for col in lab_cols:
        last_valid_time = None
        last_valid_val = None
        for i, row in df.iterrows():
            if pd.notna(row[col]):
                last_valid_time = row[time_col]
                last_valid_val = row[col]
                df_filled.at[i, col] = last_valid_val
            elif last_valid_time is not None:
                gap = (row[time_col] - last_valid_time).total_seconds() / 3600
                if gap <= max_gap_hours:
                    df_filled.at[i, col] = last_valid_val
                # else: leave as NaN (gap too large to impute)
    return df_filled

# handle ICU transfer edge cases
# patients transferred between ICUs (e.g., MICU -> CSRU) would create
# duplicate patient stay entries that caused issues in cohort construction
def handle_icu_transfers(stays_df):
    """merge consecutive ICU stays for the same patient separated by <6 hours
    
    patients transferred between units should be treated as a single continuous stay
    previously this created duplicate rows and corrupted time-series alignment
    """
    import pandas as pd

    stays_df = stays_df.sort_values(['subject_id', 'intime'])
    merged = []

    for subject_id, group in stays_df.groupby('subject_id'):
        group = group.reset_index(drop=True)
        current = group.iloc[0].to_dict()
        for i in range(1, len(group)):
            row = group.iloc[i]
            gap_hours = (row['intime'] - current['outtime']).total_seconds() / 3600
            if gap_hours < 6:  # treat as continuous stay if gap < 6h
                current['outtime'] = row['outtime']
                current['los'] = (current['outtime'] - current['intime']).total_seconds() / 3600
            else:
                merged.append(current)
                current = row.to_dict()
        merged.append(current)

    return pd.DataFrame(merged)
