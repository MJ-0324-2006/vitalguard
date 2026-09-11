"""
Clinical Time-Series Preprocessing Pipeline.

Handles the full preprocessing pipeline for ICU vital sign data:
  1. Physiological range filtering (channel-specific bounds)
  2. Irregular time-series resampling (forward fill + linear interpolation)
  3. Missing value imputation (forward fill, mean, indicator variables)
  4. Normalization (per-patient, per-population, robust z-score)
  5. Sliding window extraction for model input

Design principles:
  - No information leakage: normalization statistics from training set only
  - Missing data as informative signal: add binary indicator columns
  - Physiological constraints respected throughout
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physiological bounds (reference: clinical standards)
# ---------------------------------------------------------------------------

PHYSIOLOGICAL_BOUNDS: Dict[str, Tuple[float, float]] = {
    "heart_rate":   (20.0,  300.0),   # bpm
    "sbp":          (40.0,  300.0),   # mmHg
    "dbp":          (20.0,  200.0),   # mmHg
    "map":          (20.0,  200.0),   # mmHg
    "spo2":         (60.0,  100.0),   # %
    "resp_rate":    (4.0,   60.0),    # /min
    "temperature":  (25.0,  45.0),    # °C
    "gcs":          (3.0,   15.0),    # total score
    "lactate":      (0.1,   30.0),    # mmol/L
    "wbc":          (0.1,   200.0),   # K/uL
    "creatinine":   (0.1,   30.0),    # mg/dL
    "bilirubin_total": (0.1, 50.0),   # mg/dL
    "platelet":     (1.0,   2000.0),  # K/uL
    "pao2":         (20.0,  700.0),   # mmHg
    "fio2":         (0.21,  1.0),     # fraction
    "inr":          (0.5,   20.0),    # ratio
}

# Reference population statistics (MIMIC-III, for initialisation)
# These are overridden by fit() from actual training data
POPULATION_STATS: Dict[str, Tuple[float, float]] = {
    "heart_rate":   (84.0,  17.0),
    "sbp":          (121.0, 22.0),
    "dbp":          (63.0,  14.0),
    "map":          (82.0,  15.0),
    "spo2":         (97.0,  3.0),
    "resp_rate":    (18.0,  5.0),
    "temperature":  (37.0,  0.5),
    "gcs":          (13.0,  3.0),
}

VITAL_CHANNELS = [
    "heart_rate", "sbp", "dbp", "map",
    "spo2", "resp_rate", "temperature", "gcs"
]


# ---------------------------------------------------------------------------
# Range Filter
# ---------------------------------------------------------------------------

class PhysiologicalRangeFilter:
    """Remove values outside physiological bounds (plausible range filter)."""

    def __init__(self, bounds: Optional[Dict[str, Tuple[float, float]]] = None) -> None:
        self.bounds = bounds or PHYSIOLOGICAL_BOUNDS

    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace out-of-range values with NaN.

        Args:
            df: DataFrame with vital sign columns

        Returns:
            Filtered DataFrame (same shape, OOR values → NaN)
        """
        df = df.copy()
        for col in df.columns:
            if col in self.bounds:
                lo, hi = self.bounds[col]
                mask = (df[col] < lo) | (df[col] > hi)
                n_removed = mask.sum()
                if n_removed > 0:
                    logger.debug(f"Removed {n_removed} out-of-range values in {col}")
                df.loc[mask, col] = np.nan
        return df

    def filter_batch(self, arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Filter a dict of channel arrays."""
        result = {}
        for ch, arr in arrays.items():
            if ch in self.bounds:
                lo, hi = self.bounds[ch]
                filtered = arr.copy()
                filtered[(filtered < lo) | (filtered > hi)] = np.nan
                result[ch] = filtered
            else:
                result[ch] = arr
        return result


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------

class ClinicalImputer:
    """
    Multi-strategy missing value imputation for clinical time series.

    Strategies:
      'forward_fill': Last observation carried forward (LOCF) — clinically standard
      'linear':       Linear interpolation between observations
      'mean':         Replace with channel population mean
      'indicator':    Add binary {channel}_missing columns (for model awareness)
      'zero':         Replace with 0 (for indicator variables)

    Recommended: 'forward_fill' for vitals (last known value); 'indicator' always.
    """

    def __init__(
        self,
        strategy: str = "forward_fill",
        add_indicator: bool = True,
        max_gap_hours: float = 8.0,
        population_means: Optional[Dict[str, float]] = None,
    ) -> None:
        self.strategy = strategy
        self.add_indicator = add_indicator
        self.max_gap_hours = max_gap_hours  # don't forward-fill beyond this
        self.population_means = population_means or {}
        self._fitted_means: Dict[str, float] = {}

    def fit(self, dfs: List[pd.DataFrame]) -> "ClinicalImputer":
        """Compute population means from training set."""
        all_data = pd.concat(dfs, axis=0, ignore_index=True)
        self._fitted_means = all_data.mean(numeric_only=True).to_dict()
        logger.info(f"Fitted imputer on {len(dfs)} stays")
        return self

    def transform(self, df: pd.DataFrame, resample_freq: str = "1h") -> pd.DataFrame:
        """
        Apply imputation to a vital sign DataFrame.

        Args:
            df:             Time-indexed DataFrame
            resample_freq:  Original resampling frequency (for gap computation)

        Returns:
            Imputed DataFrame, optionally with indicator columns added
        """
        df = df.copy()
        result_cols = {}

        for col in df.columns:
            if col.endswith("_missing"):
                continue

            if self.add_indicator:
                result_cols[f"{col}_missing"] = df[col].isna().astype(float)

            series = df[col].copy()

            if self.strategy == "forward_fill":
                # Forward fill with max gap limit
                max_steps = int(self.max_gap_hours)
                series = series.ffill(limit=max_steps)
                # Backward fill only for leading NaNs at start
                series = series.bfill(limit=3)
            elif self.strategy == "linear":
                series = series.interpolate(method="linear", limit=8)
                series = series.ffill(limit=6).bfill(limit=3)
            elif self.strategy == "mean":
                mean_val = self._fitted_means.get(col, self.population_means.get(col, 0))
                series = series.fillna(mean_val)
            elif self.strategy == "zero":
                series = series.fillna(0)
            else:
                raise ValueError(f"Unknown imputation strategy: {self.strategy}")

            df[col] = series

        # Add indicator columns
        for col, vals in result_cols.items():
            df[col] = vals

        return df

    def missingness_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute missingness statistics per channel."""
        stats = []
        for col in df.columns:
            if not col.endswith("_missing"):
                missing_pct = df[col].isna().mean() * 100
                stats.append({"channel": col, "missing_pct": missing_pct})
        return pd.DataFrame(stats)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

class VitalSignNormalizer:
    """
    Normalise physiological signals.

    Strategies:
      'z_score':         (x - mean) / std  — population-level
      'robust':          (x - median) / IQR — more robust to outliers
      'per_patient':     Z-score using patient's own statistics (first N hours)
      'min_max':         Scale to [0, 1] range (physiological bounds)

    Always fit on training set only. Save/load stats for deployment.
    """

    def __init__(
        self,
        strategy: str = "z_score",
        channels: Optional[List[str]] = None,
    ) -> None:
        self.strategy = strategy
        self.channels = channels or VITAL_CHANNELS
        self._stats: Dict[str, Dict[str, float]] = {}

    def fit(self, dfs: List[pd.DataFrame]) -> "VitalSignNormalizer":
        """Compute normalization statistics from training DataFrames."""
        all_data = pd.concat(dfs, axis=0, ignore_index=True)

        for ch in self.channels:
            if ch not in all_data.columns:
                continue
            vals = all_data[ch].dropna()
            if len(vals) == 0:
                continue

            if self.strategy in ("z_score", "per_patient"):
                self._stats[ch] = {
                    "mean": float(vals.mean()),
                    "std": max(float(vals.std()), 1e-6),
                }
            elif self.strategy == "robust":
                q25, q50, q75 = vals.quantile([0.25, 0.5, 0.75])
                iqr = max(q75 - q25, 1e-6)
                self._stats[ch] = {"median": float(q50), "iqr": iqr}
            elif self.strategy == "min_max":
                lo, hi = PHYSIOLOGICAL_BOUNDS.get(ch, (vals.min(), vals.max()))
                self._stats[ch] = {"min": lo, "max": hi}

        logger.info(f"Fitted normalizer on {len(dfs)} stays, {len(self._stats)} channels")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply normalisation to a DataFrame."""
        df = df.copy()
        for ch in self.channels:
            if ch not in df.columns or ch not in self._stats:
                continue
            s = self._stats[ch]

            if self.strategy == "z_score":
                df[ch] = (df[ch] - s["mean"]) / s["std"]
            elif self.strategy == "robust":
                df[ch] = (df[ch] - s["median"]) / s["iqr"]
            elif self.strategy == "min_max":
                df[ch] = (df[ch] - s["min"]) / (s["max"] - s["min"] + 1e-6)
                df[ch] = df[ch].clip(0, 1)
            elif self.strategy == "per_patient":
                mean = df[ch].mean()
                std = max(df[ch].std(), 1e-6)
                df[ch] = (df[ch] - mean) / std

        return df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reverse normalisation (for interpretability)."""
        df = df.copy()
        for ch in self.channels:
            if ch not in df.columns or ch not in self._stats:
                continue
            s = self._stats[ch]

            if self.strategy == "z_score":
                df[ch] = df[ch] * s["std"] + s["mean"]
            elif self.strategy == "robust":
                df[ch] = df[ch] * s["iqr"] + s["median"]
            elif self.strategy == "min_max":
                df[ch] = df[ch] * (s["max"] - s["min"]) + s["min"]

        return df

    def save(self, path: Union[str, Path]) -> None:
        """Save normalisation statistics to JSON."""
        import json
        with open(path, "w") as f:
            json.dump(self._stats, f, indent=2)

    def load(self, path: Union[str, Path]) -> "VitalSignNormalizer":
        """Load normalisation statistics from JSON."""
        import json
        with open(path) as f:
            self._stats = json.load(f)
        return self


# ---------------------------------------------------------------------------
# Sliding Window Extraction
# ---------------------------------------------------------------------------

class SlidingWindowExtractor:
    """
    Extract sliding windows from time-indexed vital sign DataFrames.

    For each window, returns (input_window, label, metadata).
    Windows are extracted with a configurable step size to balance
    coverage and computational cost.
    """

    def __init__(
        self,
        window_size: int = 6,       # hours
        horizon: int = 24,          # hours ahead for event labeling
        step_size: int = 1,         # hours between consecutive windows
        min_completeness: float = 0.5,  # min fraction of non-missing values
        channels: Optional[List[str]] = None,
    ) -> None:
        self.window_size = window_size
        self.horizon = horizon
        self.step_size = step_size
        self.min_completeness = min_completeness
        self.channels = channels or VITAL_CHANNELS

    def extract_windows(
        self,
        vitals_df: pd.DataFrame,
        labels: Optional[pd.Series] = None,
        stay_metadata: Optional[dict] = None,
    ) -> List[Dict]:
        """
        Extract all valid windows from a stay's vital sign DataFrame.

        Args:
            vitals_df:     Time-indexed DataFrame with vital sign columns
            labels:        Time-indexed Series with binary event labels (optional)
            stay_metadata: Dict with stay-level metadata (icustay_id, etc.)

        Returns:
            List of dicts:
              'x': (window_size, n_channels) numpy array
              'label': int or None
              'time': window end timestamp
              'missing_mask': (window_size, n_channels) bool array
              'metadata': dict
        """
        windows = []
        ch_subset = [c for c in self.channels if c in vitals_df.columns]
        missing_cols = [f"{c}_missing" for c in ch_subset if f"{c}_missing" in vitals_df.columns]

        T = len(vitals_df)
        n_channels = len(ch_subset)

        for start_idx in range(0, T - self.window_size, self.step_size):
            end_idx = start_idx + self.window_size
            window = vitals_df.iloc[start_idx:end_idx]

            # Check completeness (ignore indicator columns)
            vital_window = window[ch_subset]
            completeness = (~vital_window.isna()).mean().mean()
            if completeness < self.min_completeness:
                continue

            x = vital_window.values.astype(np.float32)

            # Missing mask
            if missing_cols:
                missing_mask = window[missing_cols].values.astype(bool)
            else:
                missing_mask = vital_window.isna().values

            # Label: any event in the horizon after window end
            label = None
            if labels is not None:
                window_end_time = vitals_df.index[end_idx - 1]
                future_labels = labels[
                    (labels.index > window_end_time) &
                    (labels.index <= window_end_time + pd.Timedelta(hours=self.horizon))
                ]
                label = int(future_labels.max()) if len(future_labels) > 0 else 0

            windows.append({
                "x": x,
                "label": label,
                "time": vitals_df.index[end_idx - 1],
                "missing_mask": missing_mask,
                "metadata": stay_metadata or {},
                "channels": ch_subset,
            })

        return windows

    def extract_last_window(
        self, vitals_df: pd.DataFrame, channels: Optional[List[str]] = None
    ) -> Optional[np.ndarray]:
        """
        Extract the most recent window for real-time inference.

        Returns:
            (window_size, n_channels) array, or None if insufficient data
        """
        ch_subset = channels or [c for c in self.channels if c in vitals_df.columns]
        if len(vitals_df) < self.window_size:
            return None
        window = vitals_df[ch_subset].iloc[-self.window_size:].values.astype(np.float32)
        return window


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class ICUTimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for ICU time-series windows.

    Supports:
      - Balanced sampling (for imbalanced outcomes)
      - Class-weighted sampling (via sample_weights property)
      - Lazy loading from disk
    """

    def __init__(
        self,
        windows: List[Dict],
        transform=None,
        outcome: str = "label",
    ) -> None:
        """
        Args:
            windows:   List of window dicts from SlidingWindowExtractor
            transform: Optional transform to apply to x tensors
            outcome:   Key to use as label from window dict
        """
        self.windows = windows
        self.transform = transform
        self.outcome = outcome

        labels = [w.get(outcome, 0) or 0 for w in windows]
        self._labels = np.array(labels)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        w = self.windows[idx]
        x = torch.from_numpy(w["x"]).float()
        if self.transform is not None:
            x = self.transform(x)

        label = w.get(self.outcome, 0) or 0
        missing = torch.from_numpy(w["missing_mask"]).float()

        # Zero-fill remaining NaNs (after imputation, should be rare)
        x = torch.nan_to_num(x, nan=0.0)

        return {
            "x": x,
            "label": torch.tensor(label, dtype=torch.long),
            "missing_mask": missing,
        }

    @property
    def sample_weights(self) -> np.ndarray:
        """Per-sample weights for WeightedRandomSampler (class balancing)."""
        n_pos = max(self._labels.sum(), 1)
        n_neg = max(len(self._labels) - n_pos, 1)
        weights = np.where(self._labels == 1, 1.0 / n_pos, 1.0 / n_neg)
        return weights / weights.sum() * len(weights)

    @property
    def class_counts(self) -> Tuple[int, int]:
        n_pos = int(self._labels.sum())
        return (len(self._labels) - n_pos, n_pos)

    @property
    def pos_weight(self) -> float:
        """BCEWithLogitsLoss positive weight for class imbalance."""
        n_neg, n_pos = self.class_counts
        return n_neg / max(n_pos, 1)


# ---------------------------------------------------------------------------
# Full Preprocessing Pipeline
# ---------------------------------------------------------------------------

class PreprocessingPipeline:
    """
    Orchestrates the full preprocessing sequence:
      range_filter → impute → normalize → extract_windows

    Fit on training set; transform all splits consistently.
    """

    def __init__(
        self,
        window_size: int = 6,
        horizon: int = 24,
        step_size: int = 1,
        imputation_strategy: str = "forward_fill",
        normalization_strategy: str = "z_score",
        add_missing_indicator: bool = True,
        min_completeness: float = 0.5,
        channels: Optional[List[str]] = None,
    ) -> None:
        self.channels = channels or VITAL_CHANNELS
        self.range_filter = PhysiologicalRangeFilter()
        self.imputer = ClinicalImputer(
            strategy=imputation_strategy,
            add_indicator=add_missing_indicator,
        )
        self.normalizer = VitalSignNormalizer(
            strategy=normalization_strategy,
            channels=self.channels,
        )
        self.window_extractor = SlidingWindowExtractor(
            window_size=window_size,
            horizon=horizon,
            step_size=step_size,
            min_completeness=min_completeness,
            channels=self.channels,
        )

    def fit(self, train_dfs: List[pd.DataFrame]) -> "PreprocessingPipeline":
        """Fit imputer and normalizer on training DataFrames."""
        filtered = [self.range_filter.filter(df) for df in train_dfs]
        imputed = [self.imputer.transform(df) for df in filtered]
        self.imputer.fit(filtered)
        self.normalizer.fit(imputed)
        return self

    def transform(
        self,
        df: pd.DataFrame,
        labels: Optional[pd.Series] = None,
        metadata: Optional[dict] = None,
    ) -> List[Dict]:
        """Apply full preprocessing and extract windows."""
        df = self.range_filter.filter(df)
        df = self.imputer.transform(df)
        df = self.normalizer.transform(df)
        windows = self.window_extractor.extract_windows(df, labels, metadata)
        return windows

    def save(self, dir_path: Union[str, Path]) -> None:
        """Save all fitted statistics to directory."""
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        self.normalizer.save(dir_path / "normalizer_stats.json")
        logger.info(f"Preprocessing pipeline saved to {dir_path}")

    def load(self, dir_path: Union[str, Path]) -> "PreprocessingPipeline":
        """Load fitted statistics from directory."""
        dir_path = Path(dir_path)
        self.normalizer.load(dir_path / "normalizer_stats.json")
        return self


if __name__ == "__main__":
    # Smoke test with synthetic data
    np.random.seed(42)
    T = 200  # 200 hours

    times = pd.date_range("2023-01-01", periods=T, freq="1h")
    df = pd.DataFrame({
        "heart_rate":  np.random.normal(84, 15, T),
        "sbp":         np.random.normal(120, 20, T),
        "dbp":         np.random.normal(70, 10, T),
        "map":         np.random.normal(85, 12, T),
        "spo2":        np.random.normal(97, 2, T).clip(60, 100),
        "resp_rate":   np.random.normal(18, 4, T).clip(4, 60),
        "temperature": np.random.normal(37, 0.4, T),
        "gcs":         np.random.choice([13, 14, 15], T).astype(float),
    }, index=times)

    # Inject some NaNs and OOR values
    df.iloc[10:15, 0] = np.nan
    df.iloc[30, 1] = 999  # out-of-range

    pipeline = PreprocessingPipeline(window_size=6, horizon=12, step_size=1)
    pipeline.fit([df])

    labels = pd.Series(np.zeros(T), index=times)
    labels.iloc[48:72] = 1  # simulate event window

    windows = pipeline.transform(df, labels, metadata={"icustay_id": 12345})
    print(f"Extracted {len(windows)} windows")
    print(f"Window shape: {windows[0]['x'].shape}")
    print(f"Positive windows: {sum(w['label'] == 1 for w in windows)}")
    print("Preprocessing pipeline smoke test passed.")

# Physiological limits for outlier rejection
HR_RANGE = (20, 300)
SBP_RANGE = (40, 300)

# include GCS (Glasgow Coma Scale) in feature set
# was missing from original feature list - important for neuro monitoring
GCS_COMPONENTS = ['gcs_motor', 'gcs_verbal', 'gcs_eye']
GCS_TOTAL = 'gcs_total'

def add_gcs_features(df):
    """add GCS score columns to feature dataframe
    
    GCS total = motor (1-6) + verbal (1-5) + eye (1-4), range 3-15
    lower scores indicate worse neurological status
    """
    import numpy as np
    if all(c in df.columns for c in GCS_COMPONENTS):
        df[GCS_TOTAL] = df[GCS_COMPONENTS].sum(axis=1)
    elif GCS_TOTAL not in df.columns:
        # placeholder if GCS not in source data
        df[GCS_TOTAL] = np.nan
    return df
