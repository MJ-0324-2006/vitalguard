"""
Sepsis-3 Label Generation for MIMIC-III/IV.

Implements the Sepsis-3 consensus definition:
  Sepsis = Suspected infection + Acute organ dysfunction (SOFA score increase >= 2)

Reference:
  Singer M, et al. (2016). The Third International Consensus Definitions for
  Sepsis and Septic Shock (Sepsis-3). JAMA, 315(8), 801-810.

  Seymour CW, et al. (2016). Assessment of Clinical Criteria for Sepsis.
  JAMA, 315(8), 762-774.

MIMIC-III Sepsis implementation adapted from:
  Johnson AE, et al. (2017). MIMIC-III Clinical Database.
  Harutyunyan H, et al. (2019). Multitask Learning and Benchmarking
  with Clinical Time Series Data. Scientific Data.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SOFA Score Components
# ---------------------------------------------------------------------------

class SOFAScore:
    """
    Sequential Organ Failure Assessment (SOFA) Score computation.

    Computes SOFA sub-scores for 6 organ systems using MIMIC clinical data.
    Total SOFA ranges from 0-24 (higher = more organ failure).

    Sub-scores:
      Respiration:    PaO2/FiO2 ratio
      Coagulation:    Platelet count
      Liver:          Bilirubin (total)
      Cardiovascular: MAP or vasopressor use
      CNS:            GCS
      Renal:          Creatinine or urine output
    """

    @staticmethod
    def respiratory_score(pao2_fio2: float) -> int:
        """PaO2/FiO2 ratio → SOFA respiratory sub-score (0-4)."""
        if np.isnan(pao2_fio2):
            return 0  # treat missing as normal
        if pao2_fio2 < 100:
            return 4
        elif pao2_fio2 < 200:
            return 3
        elif pao2_fio2 < 300:
            return 2
        elif pao2_fio2 < 400:
            return 1
        return 0

    @staticmethod
    def coagulation_score(platelets: float) -> int:
        """Platelet count (K/uL) → SOFA coagulation sub-score (0-4)."""
        if np.isnan(platelets):
            return 0
        if platelets < 20:
            return 4
        elif platelets < 50:
            return 3
        elif platelets < 100:
            return 2
        elif platelets < 150:
            return 1
        return 0

    @staticmethod
    def liver_score(bilirubin: float) -> int:
        """Total bilirubin (mg/dL) → SOFA liver sub-score (0-4)."""
        if np.isnan(bilirubin):
            return 0
        if bilirubin >= 12.0:
            return 4
        elif bilirubin >= 6.0:
            return 3
        elif bilirubin >= 2.0:
            return 2
        elif bilirubin >= 1.2:
            return 1
        return 0

    @staticmethod
    def cardiovascular_score(
        map_val: float,
        dopamine_dose: float = 0.0,
        dobutamine_dose: float = 0.0,
        norepinephrine_dose: float = 0.0,
        epinephrine_dose: float = 0.0,
    ) -> int:
        """MAP or vasopressor dose → SOFA cardiovascular sub-score (0-4)."""
        if norepinephrine_dose > 0.1 or epinephrine_dose > 0.1:
            return 4
        elif norepinephrine_dose > 0 or epinephrine_dose > 0:
            return 3
        elif dopamine_dose > 15 or dobutamine_dose > 0:
            return 3
        elif dopamine_dose > 5:
            return 2
        elif dopamine_dose > 0:
            return 2
        elif not np.isnan(map_val) and map_val < 70:
            return 1
        return 0

    @staticmethod
    def cns_score(gcs: float) -> int:
        """GCS total → SOFA CNS sub-score (0-4)."""
        if np.isnan(gcs):
            return 0
        if gcs < 6:
            return 4
        elif gcs < 10:
            return 3
        elif gcs < 13:
            return 2
        elif gcs < 15:
            return 1
        return 0

    @staticmethod
    def renal_score(
        creatinine: float,
        urine_output_24h: float = np.nan,
    ) -> int:
        """Creatinine (mg/dL) or urine output → SOFA renal sub-score (0-4)."""
        if not np.isnan(urine_output_24h):
            if urine_output_24h < 200:
                return 4
            elif urine_output_24h < 500:
                return 3

        if np.isnan(creatinine):
            return 0
        if creatinine >= 5.0:
            return 4
        elif creatinine >= 3.5:
            return 3
        elif creatinine >= 2.0:
            return 2
        elif creatinine >= 1.2:
            return 1
        return 0

    @classmethod
    def compute(
        cls,
        pao2_fio2: float = np.nan,
        platelets: float = np.nan,
        bilirubin: float = np.nan,
        map_val: float = np.nan,
        gcs: float = np.nan,
        creatinine: float = np.nan,
        urine_output_24h: float = np.nan,
        vasopressor_flags: Optional[dict] = None,
    ) -> Dict[str, int]:
        """
        Compute full SOFA score.

        Returns:
            dict with 'total' and component scores
        """
        vp = vasopressor_flags or {}
        resp = cls.respiratory_score(pao2_fio2)
        coag = cls.coagulation_score(platelets)
        liver = cls.liver_score(bilirubin)
        cardio = cls.cardiovascular_score(
            map_val,
            dopamine_dose=vp.get("dopamine", 0.0),
            dobutamine_dose=vp.get("dobutamine", 0.0),
            norepinephrine_dose=vp.get("norepinephrine", 0.0),
            epinephrine_dose=vp.get("epinephrine", 0.0),
        )
        cns = cls.cns_score(gcs)
        renal = cls.renal_score(creatinine, urine_output_24h)

        total = resp + coag + liver + cardio + cns + renal
        return {
            "total": total,
            "respiratory": resp,
            "coagulation": coag,
            "liver": liver,
            "cardiovascular": cardio,
            "cns": cns,
            "renal": renal,
        }

    @classmethod
    def compute_series(
        cls,
        labs_df: pd.DataFrame,
        vitals_df: pd.DataFrame,
        vasopressor_df: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """
        Compute hourly SOFA scores for an ICU stay.

        Args:
            labs_df:        Time-indexed lab values (platelet, bilirubin, creatinine, pao2, fio2)
            vitals_df:      Time-indexed vital signs (map, gcs)
            vasopressor_df: Time-indexed vasopressor events

        Returns:
            hourly_sofa: pd.Series of SOFA scores indexed by time
        """
        # Align all inputs to common hourly index
        if vitals_df.empty:
            return pd.Series(dtype=float)

        hourly_index = vitals_df.index

        def get_series(df, col):
            if df is None or col not in df.columns:
                return pd.Series(np.nan, index=hourly_index)
            s = df[col].reindex(hourly_index)
            return s.ffill(limit=24)  # carry lab values forward up to 24h

        pao2 = get_series(labs_df, "pao2")
        fio2 = get_series(labs_df, "fio2").fillna(0.21)  # room air
        pao2_fio2 = pao2 / fio2

        platelets = get_series(labs_df, "platelet")
        bilirubin = get_series(labs_df, "bilirubin_total")
        creatinine = get_series(labs_df, "creatinine")
        map_s = get_series(vitals_df, "map")
        gcs_s = get_series(vitals_df, "gcs")

        sofa_scores = []
        for t in hourly_index:
            vp_flags = {}
            if vasopressor_df is not None and not vasopressor_df.empty:
                active = vasopressor_df[
                    (vasopressor_df.index <= t) &
                    (vasopressor_df.get("endtime", pd.Series(t)) >= t)
                ]
                for _, row in active.iterrows():
                    vp_flags[row.get("drug_name", "vasopressor")] = 1.0

            score = cls.compute(
                pao2_fio2=pao2_fio2.get(t, np.nan),
                platelets=platelets.get(t, np.nan),
                bilirubin=bilirubin.get(t, np.nan),
                map_val=map_s.get(t, np.nan),
                gcs=gcs_s.get(t, np.nan),
                creatinine=creatinine.get(t, np.nan),
                vasopressor_flags=vp_flags,
            )
            sofa_scores.append(score["total"])

        return pd.Series(sofa_scores, index=hourly_index, name="sofa_total")


# ---------------------------------------------------------------------------
# qSOFA
# ---------------------------------------------------------------------------

class qSOFAScore:
    """
    Quick SOFA (qSOFA) as a bedside screening tool.

    3 criteria, each scoring 1 point:
      - Respiratory rate >= 22 /min
      - Altered consciousness (GCS < 15)
      - Systolic BP <= 100 mmHg

    qSOFA >= 2 suggests high risk of poor outcome — use to trigger SOFA assessment.
    """

    @staticmethod
    def compute(rr: float, gcs: float, sbp: float) -> Dict[str, int]:
        rr_flag = int(not np.isnan(rr) and rr >= 22)
        gcs_flag = int(not np.isnan(gcs) and gcs < 15)
        sbp_flag = int(not np.isnan(sbp) and sbp <= 100)
        total = rr_flag + gcs_flag + sbp_flag
        return {
            "total": total,
            "rr": rr_flag,
            "gcs": gcs_flag,
            "sbp": sbp_flag,
            "positive": total >= 2,
        }

    @classmethod
    def compute_series(cls, vitals_df: pd.DataFrame) -> pd.Series:
        """Compute hourly qSOFA for a stay."""
        results = []
        for t, row in vitals_df.iterrows():
            score = cls.compute(
                rr=row.get("resp_rate", np.nan),
                gcs=row.get("gcs", np.nan),
                sbp=row.get("sbp", np.nan),
            )
            results.append(score["total"])
        return pd.Series(results, index=vitals_df.index, name="qsofa_total")


# ---------------------------------------------------------------------------
# Infection Detection
# ---------------------------------------------------------------------------

class SuspectedInfectionDetector:
    """
    Detect suspected infection events based on antibiotic + culture ordering.

    Sepsis-3 definition: suspected infection =
      (antibiotic administration) AND (blood culture obtained) within ±24h.

    Simplified approach (without culture data):
      Any antibiotic initiation event is treated as suspected infection proxy.
    """

    # MIMIC-III antibiotic itemids (INPUTEVENTS + PRESCRIPTIONS)
    ANTIBIOTIC_NAMES = [
        "vancomycin", "piperacillin", "meropenem", "cefepime",
        "ciprofloxacin", "metronidazole", "levofloxacin", "azithromycin",
        "ampicillin", "clindamycin", "fluconazole", "micafungin",
    ]

    def __init__(self, connector=None) -> None:
        self.connector = connector

    def extract_antibiotic_events(
        self,
        hadm_id: int,
        icu_intime: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Extract antibiotic administration events for a hospital admission.

        Returns DataFrame: starttime, antibiotic_name, route
        """
        if self.connector is None:
            return pd.DataFrame(columns=["starttime", "drug_name"])

        try:
            prescriptions = self.connector.read_table(
                "PRESCRIPTIONS",
                ["hadm_id", "startdate", "drug", "route"]
            )
            df = prescriptions[prescriptions["hadm_id"] == hadm_id].copy()

            # Filter to antibiotics
            abx_mask = df["drug"].str.lower().str.contains(
                "|".join(self.ANTIBIOTIC_NAMES), na=False
            )
            df = df[abx_mask]
            df["starttime"] = pd.to_datetime(df["startdate"])
            df = df[df["starttime"] >= icu_intime]
            return df[["starttime", "drug"]].rename(columns={"drug": "drug_name"})
        except Exception as e:
            logger.warning(f"Could not extract antibiotics for hadm_id={hadm_id}: {e}")
            return pd.DataFrame(columns=["starttime", "drug_name"])

    def get_first_infection_time(
        self, hadm_id: int, icu_intime: pd.Timestamp
    ) -> Optional[pd.Timestamp]:
        """Return time of first suspected infection event, or None."""
        abx = self.extract_antibiotic_events(hadm_id, icu_intime)
        if abx.empty:
            return None
        return abx["starttime"].min()


# ---------------------------------------------------------------------------
# Main Sepsis Labeler
# ---------------------------------------------------------------------------

class SepsisLabeler:
    """
    Generate Sepsis-3 onset labels for ICU stays.

    Sepsis-3 criteria:
      1. Suspected infection (antibiotic + culture, or antibiotic alone as proxy)
      2. Acute organ dysfunction: SOFA total >= 2 OR SOFA increase >= 2 from baseline

    Onset time: earlier of (antibiotic start time) and (SOFA >= 2 time),
    within a ±24h window.

    Output labels:
      - Binary: sepsis event (0/1) within horizon window
      - Onset time: timestamp of sepsis onset
      - Hours to onset: hours from window end to onset
    """

    def __init__(
        self,
        sofa_threshold: int = 2,
        onset_window_hours: float = 24.0,
        use_qsofa_screen: bool = True,
        connector=None,
    ) -> None:
        self.sofa_threshold = sofa_threshold
        self.onset_window_hours = onset_window_hours
        self.use_qsofa_screen = use_qsofa_screen
        self.infection_detector = SuspectedInfectionDetector(connector)

    def label_stay(
        self,
        icustay_id: int,
        hadm_id: int,
        icu_intime: pd.Timestamp,
        vitals_df: pd.DataFrame,
        labs_df: pd.DataFrame,
        vasopressor_df: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        Determine sepsis onset for a single ICU stay.

        Returns dict:
            'sepsis':          bool — did sepsis occur?
            'onset_time':      pd.Timestamp or None
            'hours_to_onset':  float or None
            'sofa_series':     pd.Series of hourly SOFA
            'qsofa_series':    pd.Series of hourly qSOFA
            'max_sofa':        float
            'baseline_sofa':   float (first 24h)
        """
        # Compute SOFA series
        sofa_series = SOFAScore.compute_series(labs_df, vitals_df, vasopressor_df)
        qsofa_series = qSOFAScore.compute_series(vitals_df) if not vitals_df.empty else pd.Series()

        baseline_sofa = sofa_series.iloc[:24].mean() if len(sofa_series) >= 1 else 0
        max_sofa = sofa_series.max() if len(sofa_series) > 0 else 0

        # Check organ dysfunction (SOFA >= threshold OR increase >= threshold)
        sofa_acute = sofa_series[
            (sofa_series >= self.sofa_threshold) |
            ((sofa_series - baseline_sofa) >= self.sofa_threshold)
        ]

        organ_dysfunction_time = sofa_acute.index.min() if len(sofa_acute) > 0 else None

        # Check suspected infection
        infection_time = self.infection_detector.get_first_infection_time(hadm_id, icu_intime)

        # Apply Sepsis-3 criteria
        sepsis = False
        onset_time = None

        if organ_dysfunction_time is not None and infection_time is not None:
            time_diff = abs((organ_dysfunction_time - infection_time).total_seconds() / 3600)
            if time_diff <= self.onset_window_hours:
                sepsis = True
                onset_time = min(organ_dysfunction_time, infection_time)
        elif organ_dysfunction_time is not None and infection_time is None:
            # Fallback: organ dysfunction alone (less specific but clinically relevant)
            # Use only if SOFA >= 4 (to reduce false positives without infection data)
            if sofa_series.loc[organ_dysfunction_time] >= 4:
                sepsis = True
                onset_time = organ_dysfunction_time

        hours_to_onset = None
        if onset_time is not None:
            hours_to_onset = (onset_time - icu_intime).total_seconds() / 3600

        return {
            "icustay_id": icustay_id,
            "sepsis": sepsis,
            "onset_time": onset_time,
            "hours_to_onset": hours_to_onset,
            "sofa_series": sofa_series,
            "qsofa_series": qsofa_series,
            "max_sofa": float(max_sofa),
            "baseline_sofa": float(baseline_sofa),
            "organ_dysfunction_time": organ_dysfunction_time,
            "infection_time": infection_time,
        }

    def generate_window_labels(
        self,
        sepsis_result: Dict,
        time_index: pd.DatetimeIndex,
        horizon_hours: int = 24,
        negative_buffer_hours: int = 6,
    ) -> pd.Series:
        """
        Generate binary window labels aligned to a time index.

        For each time t:
          label=1 if sepsis onset occurs in (t, t+horizon_hours]
          label=0 if t is >= negative_buffer_hours BEFORE onset
          label=NaN (exclude) if t is within buffer zone before onset

        Args:
            sepsis_result:          Output of label_stay()
            time_index:             Hourly datetime index of vital sign data
            horizon_hours:          Prediction horizon
            negative_buffer_hours:  Buffer zone before onset (excluded from negatives)

        Returns:
            pd.Series of binary labels
        """
        labels = pd.Series(0, index=time_index, dtype=float)

        if not sepsis_result["sepsis"] or sepsis_result["onset_time"] is None:
            return labels  # all negative

        onset = sepsis_result["onset_time"]

        for t in time_index:
            t_end = t + pd.Timedelta(hours=horizon_hours)
            t_buffer_start = onset - pd.Timedelta(hours=negative_buffer_hours)

            if t < t_buffer_start and t_end >= onset:
                # Window contains onset → positive
                labels[t] = 1
            elif t_buffer_start <= t <= onset:
                # Buffer zone: exclude from training
                labels[t] = np.nan

        return labels

    def process_cohort(
        self,
        cohort: pd.DataFrame,
        vitals_dict: Dict[int, pd.DataFrame],
        labs_dict: Dict[int, pd.DataFrame],
        vasopressor_dict: Optional[Dict[int, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """
        Process entire cohort to generate sepsis labels.

        Args:
            cohort:          DataFrame with icustay_id, hadm_id, icu_intime
            vitals_dict:     {icustay_id: vitals_df}
            labs_dict:       {icustay_id: labs_df}
            vasopressor_dict:{icustay_id: vasopressor_df}

        Returns:
            DataFrame with columns: icustay_id, sepsis, onset_time, hours_to_onset,
            max_sofa, baseline_sofa
        """
        results = []
        for _, row in cohort.iterrows():
            stay_id = row["icustay_id"]
            vitals = vitals_dict.get(stay_id, pd.DataFrame())
            labs = labs_dict.get(stay_id, pd.DataFrame())
            vp = (vasopressor_dict or {}).get(stay_id)

            try:
                result = self.label_stay(
                    icustay_id=stay_id,
                    hadm_id=row["hadm_id"],
                    icu_intime=row["icu_intime"],
                    vitals_df=vitals,
                    labs_df=labs,
                    vasopressor_df=vp,
                )
                # Drop series for summary DataFrame
                summary = {k: v for k, v in result.items()
                           if not isinstance(v, pd.Series)}
                results.append(summary)
            except Exception as e:
                logger.error(f"Error labeling stay {stay_id}: {e}")

        df = pd.DataFrame(results)
        logger.info(
            f"Labeled {len(df)} stays: "
            f"{df['sepsis'].sum()} sepsis ({df['sepsis'].mean()*100:.1f}%)"
        )
        return df


if __name__ == "__main__":
    # Smoke test with synthetic data
    np.random.seed(42)
    T = 72
    times = pd.date_range("2023-01-01", periods=T, freq="1h")

    vitals = pd.DataFrame({
        "map": np.random.normal(75, 10, T),
        "gcs": np.random.choice([13, 14, 15], T).astype(float),
        "sbp": np.random.normal(115, 20, T),
        "resp_rate": np.random.normal(18, 4, T),
    }, index=times)

    # Deterioration at hour 48
    vitals.loc[vitals.index[48:], "map"] = 60.0
    vitals.loc[vitals.index[48:], "gcs"] = 10.0

    labs = pd.DataFrame({
        "pao2": np.random.normal(90, 15, T),
        "fio2": np.full(T, 0.21),
        "platelet": np.random.normal(200, 50, T),
        "bilirubin_total": np.random.normal(1.0, 0.5, T).clip(0.1, 5),
        "creatinine": np.random.normal(1.0, 0.3, T).clip(0.3, 5),
    }, index=times)

    sofa = SOFAScore.compute_series(labs, vitals)
    qsofa = qSOFAScore.compute_series(vitals)

    print(f"Max SOFA: {sofa.max():.1f}")
    print(f"Baseline SOFA (first 6h): {sofa.iloc[:6].mean():.1f}")
    print(f"Max qSOFA: {qsofa.max()}")

    labeler = SepsisLabeler()
    result = labeler.label_stay(
        icustay_id=99999, hadm_id=88888,
        icu_intime=times[0],
        vitals_df=vitals,
        labs_df=labs,
    )
    print(f"Sepsis detected: {result['sepsis']}")
    print(f"SOFA max: {result['max_sofa']:.1f}")
    print("Sepsis labeler smoke test passed.")

SOFA_THRESHOLD = 2  # Sepsis-3 definition
