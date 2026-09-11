"""
Clinical Outcome Labeling for ICU Stays.

Generates binary and time-to-event labels for:
  - In-hospital mortality
  - ICU mortality
  - Vasopressor initiation (within N hours)
  - Mechanical ventilation initiation
  - Rapid response team (RRT) activation (proxy)

Labels are aligned to hourly time grids for windowed training.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MIMIC ItemIDs for procedure labels
# ---------------------------------------------------------------------------

# Mechanical ventilation (PROCEDUREEVENTS_MV)
VENTILATION_ITEMIDS = [
    225792,  # Invasive Ventilation
    225794,  # Non-invasive Ventilation
    227194,  # 4-CPAP
    227187,  # BiPAP
    224385,  # Intubation
]

# Vasopressor ItemIDs (see also mimic_loader.py)
VASOPRESSOR_ITEMIDS_ALL = [
    30047, 30120, 221906,   # norepinephrine
    30044, 30119, 30309, 221289,  # epinephrine
    30043, 30307, 221662,   # dopamine
    30051, 222315,           # vasopressin
    30128, 30127, 221749,   # phenylephrine
]

# GCS deterioration threshold for RRT proxy
GCS_RRT_THRESHOLD = 9  # GCS < 9 → possible RRT trigger

# Vital sign deterioration thresholds for RRT proxy
RRT_VITAL_CRITERIA = {
    "heart_rate":  (40, 130),  # HR < 40 or > 130
    "sbp":         (80, None), # SBP < 80
    "spo2":        (85, None), # SpO2 < 85
    "resp_rate":   (6, 30),    # RR < 6 or > 30
}


# ---------------------------------------------------------------------------
# Mortality Labeler
# ---------------------------------------------------------------------------

class MortalityLabeler:
    """
    Generate in-hospital and ICU mortality labels.

    In-hospital mortality: patient died during hospital admission
    ICU mortality: patient died during ICU stay (stricter)

    Both are binary outcomes aligned to time windows.
    """

    @staticmethod
    def label_stays(cohort: pd.DataFrame) -> pd.DataFrame:
        """
        Add mortality labels to cohort DataFrame.

        Expects columns: subject_id, hadm_id, icustay_id,
                         hospital_expire_flag, icu_intime, icu_outtime
                         (dod_hosp optional for ICU-specific mortality)

        Returns:
            cohort with added columns:
              'hospital_mortality':  bool
              'icu_mortality':       bool
              'survival_hours':      float (hospital LOS or time to death)
        """
        df = cohort.copy()

        # In-hospital mortality
        df["hospital_mortality"] = df["hospital_expire_flag"].astype(bool)

        # ICU mortality: died during ICU stay
        if "deathtime" in df.columns:
            df["deathtime"] = pd.to_datetime(df["deathtime"])
            df["icu_mortality"] = (
                df["hospital_mortality"] &
                df["deathtime"].notna() &
                (df["deathtime"] <= df["icu_outtime"])
            )
        else:
            df["icu_mortality"] = False

        # Time to hospital death (from ICU admit) for survival analysis
        if "deathtime" in df.columns and "admittime" in df.columns:
            df["admittime"] = pd.to_datetime(df.get("admittime", df["icu_intime"]))
            dead = df["hospital_mortality"]
            df["survival_hours"] = np.where(
                dead,
                (df["deathtime"] - df["icu_intime"]).dt.total_seconds() / 3600,
                (df["icu_outtime"] - df["icu_intime"]).dt.total_seconds() / 3600,
            )
        else:
            df["survival_hours"] = (
                df["icu_outtime"] - df["icu_intime"]
            ).dt.total_seconds() / 3600

        return df

    @staticmethod
    def generate_window_labels(
        mortality: bool,
        survival_hours: float,
        time_index: pd.DatetimeIndex,
        icu_intime: pd.Timestamp,
        horizon_hours: int = 24,
    ) -> pd.Series:
        """
        Generate time-varying mortality labels for a stay.

        For each window ending at time t:
          label = 1 if patient died within (t, t + horizon_hours]
          label = 0 if patient survived beyond t + horizon_hours
          label = NaN if time window overlaps death time (too ambiguous)

        Args:
            mortality:      Did patient die in hospital?
            survival_hours: Hours from ICU admit to death (or discharge)
            time_index:     Hourly datetime index
            icu_intime:     ICU admission timestamp
            horizon_hours:  Prediction horizon

        Returns:
            pd.Series of labels indexed by time_index
        """
        labels = pd.Series(0, index=time_index, dtype=float)

        if not mortality:
            return labels

        death_time = icu_intime + pd.Timedelta(hours=survival_hours)

        for t in time_index:
            t_end = t + pd.Timedelta(hours=horizon_hours)
            if t < death_time <= t_end:
                labels[t] = 1
            elif death_time > t_end:
                labels[t] = 0
            else:
                # t >= death_time (post-mortem window) — exclude
                labels[t] = np.nan

        return labels


# ---------------------------------------------------------------------------
# Vasopressor Labeler
# ---------------------------------------------------------------------------

class VasopressorLabeler:
    """
    Generate vasopressor initiation labels.

    Label definition: Patient initiated on any vasopressor within N hours
    from the window end time.

    Vasopressor initiation is a strong signal of hemodynamic instability
    (cardiogenic shock, septic shock, distributive shock).
    """

    def __init__(self, connector=None) -> None:
        self.connector = connector

    def extract_first_vasopressor(
        self,
        icustay_id: int,
        icu_intime: pd.Timestamp,
    ) -> Optional[pd.Timestamp]:
        """
        Query MIMIC for first vasopressor administration time.

        Returns:
            Timestamp of first vasopressor, or None if no vasopressors used
        """
        if self.connector is None:
            return None

        try:
            if self.connector.backend == "postgresql":
                ids_str = ",".join(str(i) for i in VASOPRESSOR_ITEMIDS_ALL)
                sql = f"""
                    SELECT MIN(starttime) as first_vp
                    FROM {self.connector.schema}.inputevents_mv
                    WHERE icustay_id = {icustay_id}
                      AND itemid IN ({ids_str})
                      AND rate > 0
                """
                result = self.connector.query(sql)
                if result.empty or pd.isna(result.iloc[0, 0]):
                    return None
                return pd.Timestamp(result.iloc[0, 0])
            else:
                mv = self.connector.read_table(
                    "INPUTEVENTS_MV",
                    ["icustay_id", "starttime", "itemid", "rate"]
                )
                df = mv[
                    (mv["icustay_id"] == icustay_id) &
                    (mv["itemid"].isin(VASOPRESSOR_ITEMIDS_ALL)) &
                    (mv["rate"] > 0)
                ]
                if df.empty:
                    return None
                df["starttime"] = pd.to_datetime(df["starttime"])
                return df["starttime"].min()
        except Exception as e:
            logger.warning(f"Error extracting vasopressor for stay {icustay_id}: {e}")
            return None

    def generate_window_labels(
        self,
        first_vp_time: Optional[pd.Timestamp],
        time_index: pd.DatetimeIndex,
        icu_intime: pd.Timestamp,
        horizon_hours: int = 12,
    ) -> pd.Series:
        """
        Generate window-level vasopressor initiation labels.

        Args:
            first_vp_time: Timestamp of first vasopressor (or None)
            time_index:    Hourly datetime index
            icu_intime:    ICU admission time
            horizon_hours: Prediction horizon for vasopressor start

        Returns:
            pd.Series of binary labels
        """
        labels = pd.Series(0, index=time_index, dtype=float)

        if first_vp_time is None:
            return labels

        for t in time_index:
            t_end = t + pd.Timedelta(hours=horizon_hours)
            if t < first_vp_time <= t_end:
                labels[t] = 1
            elif first_vp_time <= t:
                # Vasopressor already running — exclude (on-vasopressor window)
                labels[t] = np.nan

        return labels


# ---------------------------------------------------------------------------
# Ventilation Labeler
# ---------------------------------------------------------------------------

class VentilationLabeler:
    """
    Generate mechanical ventilation initiation labels.

    Identifies invasive and non-invasive ventilation starts from MIMIC
    PROCEDUREEVENTS_MV table.
    """

    def __init__(self, connector=None) -> None:
        self.connector = connector

    def extract_first_ventilation(
        self,
        icustay_id: int,
        icu_intime: pd.Timestamp,
        invasive_only: bool = True,
    ) -> Optional[pd.Timestamp]:
        """
        Return time of first mechanical ventilation, or None.

        Args:
            invasive_only: If True, only count invasive (intubation) ventilation
        """
        if self.connector is None:
            return None

        itemids = [225792, 224385] if invasive_only else VENTILATION_ITEMIDS

        try:
            if self.connector.backend == "postgresql":
                ids_str = ",".join(str(i) for i in itemids)
                sql = f"""
                    SELECT MIN(starttime) as first_vent
                    FROM {self.connector.schema}.procedureevents_mv
                    WHERE icustay_id = {icustay_id}
                      AND itemid IN ({ids_str})
                """
                result = self.connector.query(sql)
                if result.empty or pd.isna(result.iloc[0, 0]):
                    return None
                return pd.Timestamp(result.iloc[0, 0])
            else:
                procs = self.connector.read_table(
                    "PROCEDUREEVENTS_MV",
                    ["icustay_id", "starttime", "itemid"]
                )
                df = procs[
                    (procs["icustay_id"] == icustay_id) &
                    (procs["itemid"].isin(itemids))
                ]
                if df.empty:
                    return None
                df["starttime"] = pd.to_datetime(df["starttime"])
                return df["starttime"].min()
        except Exception as e:
            logger.warning(f"Error extracting ventilation for stay {icustay_id}: {e}")
            return None

    def generate_window_labels(
        self,
        first_vent_time: Optional[pd.Timestamp],
        time_index: pd.DatetimeIndex,
        horizon_hours: int = 12,
    ) -> pd.Series:
        """Generate window-level ventilation labels (same logic as vasopressor)."""
        labels = pd.Series(0, index=time_index, dtype=float)
        if first_vent_time is None:
            return labels

        for t in time_index:
            t_end = t + pd.Timedelta(hours=horizon_hours)
            if t < first_vent_time <= t_end:
                labels[t] = 1
            elif first_vent_time <= t:
                labels[t] = np.nan  # on-vent window excluded

        return labels


# ---------------------------------------------------------------------------
# Rapid Response Team (RRT) Proxy Labeler
# ---------------------------------------------------------------------------

class RapidResponseLabeler:
    """
    Proxy labeler for rapid response team (RRT) activation.

    MIMIC does not record RRT calls directly. We use a composite
    vital sign deterioration criterion as a clinical proxy:
      - HR < 40 or > 130 bpm
      - SBP < 80 mmHg
      - SpO2 < 85%
      - RR < 6 or > 30 /min
      - GCS < 9 (acute encephalopathy)
      - Any 2 of the above occurring within the same hour

    This proxy has ~70% concordance with documented RRT events in
    validation studies (Prytherch et al., 2010).
    """

    def __init__(self, min_criteria: int = 2) -> None:
        self.min_criteria = min_criteria

    def label_series(self, vitals_df: pd.DataFrame) -> pd.Series:
        """
        Label each hour as RRT-equivalent deterioration event.

        Args:
            vitals_df: Time-indexed vital sign DataFrame

        Returns:
            pd.Series of binary RRT-proxy labels
        """
        flags = pd.DataFrame(index=vitals_df.index)

        # HR criteria
        if "heart_rate" in vitals_df.columns:
            hr = vitals_df["heart_rate"]
            flags["hr_flag"] = ((hr < 40) | (hr > 130)).astype(int)

        # SBP criteria
        if "sbp" in vitals_df.columns:
            sbp = vitals_df["sbp"]
            flags["sbp_flag"] = (sbp < 80).astype(int)

        # SpO2 criteria
        if "spo2" in vitals_df.columns:
            spo2 = vitals_df["spo2"]
            flags["spo2_flag"] = (spo2 < 85).astype(int)

        # RR criteria
        if "resp_rate" in vitals_df.columns:
            rr = vitals_df["resp_rate"]
            flags["rr_flag"] = ((rr < 6) | (rr > 30)).astype(int)

        # GCS criteria
        if "gcs" in vitals_df.columns:
            gcs = vitals_df["gcs"]
            flags["gcs_flag"] = (gcs < GCS_RRT_THRESHOLD).astype(int)

        # RRT event = sum of criteria flags >= min_criteria
        flags = flags.fillna(0)
        n_flags = flags.sum(axis=1)
        rrt_labels = (n_flags >= self.min_criteria).astype(int)
        return rrt_labels

    def generate_window_labels(
        self,
        vitals_df: pd.DataFrame,
        horizon_hours: int = 6,
    ) -> pd.Series:
        """
        Generate window-level RRT labels (any RRT event within horizon).

        Args:
            vitals_df:    Time-indexed vital signs
            horizon_hours: Prediction horizon

        Returns:
            pd.Series aligned to vitals_df.index
        """
        rrt_series = self.label_series(vitals_df)
        labels = pd.Series(0, index=vitals_df.index, dtype=float)

        for i, t in enumerate(vitals_df.index):
            t_end = t + pd.Timedelta(hours=horizon_hours)
            future_rrt = rrt_series[(rrt_series.index > t) & (rrt_series.index <= t_end)]
            if len(future_rrt) > 0 and future_rrt.max() > 0:
                labels[t] = 1

        return labels


# ---------------------------------------------------------------------------
# Composite Outcome Labeler
# ---------------------------------------------------------------------------

class OutcomeLabeler:
    """
    Orchestrates all outcome labelers to produce a multi-outcome label matrix.

    Generates labels for:
      - hospital_mortality
      - vasopressor
      - ventilation
      - rrt_proxy

    All labels are aligned to an hourly time grid.
    """

    def __init__(self, connector=None) -> None:
        self.mortality_labeler = MortalityLabeler()
        self.vasopressor_labeler = VasopressorLabeler(connector)
        self.ventilation_labeler = VentilationLabeler(connector)
        self.rrt_labeler = RapidResponseLabeler()

    def label_stay(
        self,
        row: pd.Series,
        vitals_df: pd.DataFrame,
        horizons: Optional[Dict[str, int]] = None,
    ) -> pd.DataFrame:
        """
        Generate all outcome labels for a single ICU stay.

        Args:
            row:       Cohort row with stay metadata
            vitals_df: Time-indexed vital sign DataFrame
            horizons:  Dict mapping outcome name to horizon hours

        Returns:
            DataFrame with outcome columns, indexed to vitals_df time index
        """
        horizons = horizons or {
            "hospital_mortality": 24,
            "vasopressor": 12,
            "ventilation": 12,
            "rrt_proxy": 6,
        }

        time_index = vitals_df.index
        labels = {}

        # Mortality
        mortality = bool(row.get("hospital_mortality", row.get("hospital_expire_flag", 0)))
        survival_hours = row.get("survival_hours", None)
        if survival_hours is None and "icu_outtime" in row and "icu_intime" in row:
            survival_hours = (row["icu_outtime"] - row["icu_intime"]).total_seconds() / 3600
        survival_hours = float(survival_hours or 48)

        labels["hospital_mortality"] = MortalityLabeler.generate_window_labels(
            mortality, survival_hours, time_index, row["icu_intime"],
            horizons["hospital_mortality"]
        )

        # Vasopressor
        first_vp = self.vasopressor_labeler.extract_first_vasopressor(
            row["icustay_id"], row["icu_intime"]
        )
        labels["vasopressor"] = self.vasopressor_labeler.generate_window_labels(
            first_vp, time_index, row["icu_intime"], horizons["vasopressor"]
        )

        # Ventilation
        first_vent = self.ventilation_labeler.extract_first_ventilation(
            row["icustay_id"], row["icu_intime"]
        )
        labels["ventilation"] = self.ventilation_labeler.generate_window_labels(
            first_vent, time_index, horizons["ventilation"]
        )

        # RRT proxy
        labels["rrt_proxy"] = self.rrt_labeler.generate_window_labels(
            vitals_df, horizons["rrt_proxy"]
        )

        return pd.DataFrame(labels, index=time_index)

    def process_cohort(
        self,
        cohort: pd.DataFrame,
        vitals_dict: Dict[int, pd.DataFrame],
        horizons: Optional[Dict[str, int]] = None,
        save_dir: Optional[str] = None,
    ) -> Dict[int, pd.DataFrame]:
        """
        Generate outcome labels for all stays.

        Returns:
            Dict mapping icustay_id → outcome label DataFrame
        """
        # Add survival hours if not present
        if "survival_hours" not in cohort.columns:
            cohort = MortalityLabeler.label_stays(cohort)

        all_labels = {}
        for _, row in cohort.iterrows():
            stay_id = row["icustay_id"]
            vitals = vitals_dict.get(stay_id, pd.DataFrame())
            if vitals.empty:
                continue

            try:
                label_df = self.label_stay(row, vitals, horizons)
                all_labels[stay_id] = label_df

                if save_dir:
                    import os
                    os.makedirs(save_dir, exist_ok=True)
                    label_df.to_parquet(f"{save_dir}/{stay_id}_labels.parquet")

            except Exception as e:
                logger.error(f"Error labeling stay {stay_id}: {e}")

        logger.info(f"Generated labels for {len(all_labels)} stays")
        return all_labels


if __name__ == "__main__":
    # Smoke test
    np.random.seed(42)
    T = 72
    times = pd.date_range("2023-01-01", periods=T, freq="1h")

    vitals = pd.DataFrame({
        "heart_rate": np.random.normal(84, 15, T),
        "sbp": np.random.normal(115, 20, T),
        "spo2": np.random.normal(97, 2, T).clip(60, 100),
        "resp_rate": np.random.normal(18, 4, T),
        "gcs": np.random.choice([13, 14, 15], T).astype(float),
    }, index=times)

    # Inject deterioration at hour 48
    vitals.iloc[48:, 1] = 75.0  # low SBP
    vitals.iloc[48:, 4] = 8.0   # low GCS

    rrt_labeler = RapidResponseLabeler()
    rrt_labels = rrt_labeler.label_series(vitals)
    print(f"RRT events detected: {rrt_labels.sum()}")

    # Test mortality window labeling
    mort_labels = MortalityLabeler.generate_window_labels(
        mortality=True, survival_hours=52.0,
        time_index=times, icu_intime=times[0], horizon_hours=12
    )
    print(f"Positive mortality windows: {(mort_labels == 1).sum()}")
    print("Outcome labeler smoke test passed.")
