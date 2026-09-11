"""
Temporal Analysis for ICU Early Warning System Evaluation.

Analyses:
  - Performance vs. time-before-event curves (AUROC at T-1h, T-2h, T-4h, T-8h, T-12h)
  - Early detection sensitivity analysis
  - Kaplan-Meier survival curves stratified by model risk score
  - Calibration over time (does performance degrade further from event?)
  - Lead time distribution analysis
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Performance vs. Time-Before-Event
# ---------------------------------------------------------------------------

class TemporalPerformanceAnalyzer:
    """
    Compute model performance at different time horizons before a clinical event.

    For each horizon h in {1, 2, 4, 8, 12, 24} hours before event:
      - Collect (score, label) pairs from windows ending exactly h hours before event
      - Compute AUROC, AUPRC, sensitivity at fixed specificity

    This answers: "How well does the model detect the event when given only
    information from >= h hours before the event?"
    """

    def __init__(
        self,
        horizons: Optional[List[int]] = None,
        min_samples: int = 30,
    ) -> None:
        self.horizons = horizons or [1, 2, 4, 6, 8, 12, 24]
        self.min_samples = min_samples

    def compute(
        self,
        scores_by_stay: Dict[int, pd.Series],
        labels_by_stay: Dict[int, pd.Series],
        event_times_by_stay: Dict[int, Optional[float]],
        icu_intimes: Dict[int, pd.Timestamp],
    ) -> pd.DataFrame:
        """
        Compute AUROC at each horizon.

        Args:
            scores_by_stay:        {icustay_id: pd.Series of hourly anomaly scores}
            labels_by_stay:        {icustay_id: pd.Series of binary event labels}
            event_times_by_stay:   {icustay_id: hours from ICU admit to event (or None)}
            icu_intimes:           {icustay_id: ICU admission timestamp}

        Returns:
            DataFrame with columns: horizon_hours, auroc, auprc, n_positive, n_total
        """
        from sklearn.metrics import roc_auc_score, average_precision_score

        results = []

        for h in self.horizons:
            h_scores = []
            h_labels = []

            for stay_id, scores in scores_by_stay.items():
                event_h = event_times_by_stay.get(stay_id)
                icu_intime = icu_intimes.get(stay_id)

                if icu_intime is None:
                    continue

                # Identify the window that ends exactly h hours before event
                if event_h is not None:
                    target_time = icu_intime + pd.Timedelta(hours=event_h - h)
                    if target_time in scores.index:
                        score_val = scores.loc[target_time]
                        h_scores.append(float(score_val))
                        h_labels.append(1)
                    else:
                        # Find closest time
                        closest_idx = scores.index.get_indexer([target_time], method="nearest")[0]
                        if abs((scores.index[closest_idx] - target_time).total_seconds()) <= 3600:
                            h_scores.append(float(scores.iloc[closest_idx]))
                            h_labels.append(1)
                else:
                    # Control patient: sample a random window at horizon h
                    if len(scores) > h:
                        idx = max(0, len(scores) - h - 1)
                        h_scores.append(float(scores.iloc[idx]))
                        h_labels.append(0)

            if len(h_scores) < self.min_samples:
                continue

            y_true = np.array(h_labels)
            y_score = np.array(h_scores)

            if len(np.unique(y_true)) < 2:
                continue

            try:
                auroc = roc_auc_score(y_true, y_score)
                auprc = average_precision_score(y_true, y_score)
            except Exception:
                continue

            results.append({
                "horizon_hours": h,
                "auroc": auroc,
                "auprc": auprc,
                "n_positive": int(y_true.sum()),
                "n_total": len(y_true),
            })

        return pd.DataFrame(results).sort_values("horizon_hours")

    def auroc_curve_from_windows(
        self,
        all_scores: np.ndarray,
        all_labels: np.ndarray,
        all_hours_to_event: np.ndarray,
    ) -> pd.DataFrame:
        """
        Compute AUROC at each horizon from pre-extracted window data.

        Args:
            all_scores:         (N,) anomaly scores for all windows
            all_labels:         (N,) binary labels (1 = event patient, 0 = control)
            all_hours_to_event: (N,) hours from window end to event onset
                                (NaN for controls or no event)

        Returns:
            DataFrame with auroc per horizon
        """
        from sklearn.metrics import roc_auc_score

        results = []
        for h in self.horizons:
            # Select windows near this horizon: ±0.5h tolerance
            if h == 1:
                mask = (all_hours_to_event >= 0) & (all_hours_to_event <= 1)
            elif h <= 4:
                mask = (all_hours_to_event >= h - 0.5) & (all_hours_to_event <= h + 0.5)
            else:
                mask = (all_hours_to_event >= h - 1) & (all_hours_to_event <= h + 1)

            # Include all controls (no event)
            control_mask = np.isnan(all_hours_to_event)
            combined_mask = mask | control_mask

            y_s = all_scores[combined_mask]
            y_t = all_labels[combined_mask]

            if len(y_s) < self.min_samples or len(np.unique(y_t)) < 2:
                continue

            try:
                auroc = roc_auc_score(y_t, y_s)
                results.append({"horizon_hours": h, "auroc": auroc,
                                 "n": len(y_s), "n_pos": int(y_t.sum())})
            except Exception:
                pass

        return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Kaplan-Meier Analysis
# ---------------------------------------------------------------------------

class KaplanMeierAnalyzer:
    """
    Kaplan-Meier survival analysis stratified by model risk score.

    Stratifies ICU patients into risk quintiles (or tertiles) based on
    their peak anomaly score in the first 6 hours of their stay, then
    plots the survival function for each stratum.

    High risk stratum should show significantly lower survival probability.
    """

    def __init__(self, n_strata: int = 3) -> None:
        self.n_strata = n_strata

    def compute(
        self,
        peak_scores: np.ndarray,
        survival_hours: np.ndarray,
        events: np.ndarray,
        labels: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute KM curves stratified by risk score.

        Args:
            peak_scores:    (N,) peak anomaly score per patient (first 6h)
            survival_hours: (N,) time to event or censoring (hours)
            events:         (N,) binary event indicator
            labels:         Stratum labels (e.g., ['Low', 'Medium', 'High'])

        Returns:
            dict mapping stratum label → KM DataFrame with columns:
                time, survival, ci_lo, ci_hi, n_at_risk
        """
        # Assign strata based on score quantiles
        quantiles = np.linspace(0, 1, self.n_strata + 1)
        thresholds = np.quantile(peak_scores, quantiles)
        strata_labels = labels or [f"Q{i+1}" for i in range(self.n_strata)]

        results = {}
        for i in range(self.n_strata):
            lo = thresholds[i]
            hi = thresholds[i + 1] if i < self.n_strata - 1 else np.inf
            mask = (peak_scores >= lo) if i == 0 else (peak_scores > lo)
            mask &= (peak_scores <= hi)

            stratum_surv = survival_hours[mask]
            stratum_ev = events[mask]

            km_df = self._km_estimate(stratum_surv, stratum_ev)
            results[strata_labels[i]] = km_df

        return results

    @staticmethod
    def _km_estimate(
        survival_hours: np.ndarray,
        events: np.ndarray,
    ) -> pd.DataFrame:
        """
        Compute Kaplan-Meier estimate with Greenwood confidence intervals.

        Args:
            survival_hours: (N,) observed times
            events:         (N,) 1 = event, 0 = censored

        Returns:
            DataFrame: time, survival, ci_lo, ci_hi, n_at_risk
        """
        # Sort by time
        order = np.argsort(survival_hours)
        t = survival_hours[order]
        e = events[order]
        N = len(t)

        times = [0]
        km_probs = [1.0]
        n_at_risk = [N]
        greenwood = [0.0]  # variance accumulator

        for i, (ti, ei) in enumerate(zip(t, e)):
            if ei == 0:
                continue  # censored: skip
            n_i = N - i  # at risk at time ti
            if n_i <= 0:
                break
            # KM update
            new_prob = km_probs[-1] * (1 - 1 / n_i)
            km_probs.append(new_prob)
            times.append(float(ti))
            n_at_risk.append(n_i)
            # Greenwood variance
            gw = greenwood[-1] + 1 / (n_i * (n_i - 1)) if n_i > 1 else greenwood[-1]
            greenwood.append(gw)

        km_arr = np.array(km_probs)
        gw_arr = np.array(greenwood)

        # 95% CI via log-log transformation (Borgan & Liestol, 1990)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_log_s = np.log(-np.log(km_arr + 1e-10))
            sigma = gw_arr / (np.log(km_arr + 1e-10) ** 2 + 1e-10)
            ci_lo = np.exp(-np.exp(log_log_s + 1.96 * np.sqrt(sigma)))
            ci_hi = np.exp(-np.exp(log_log_s - 1.96 * np.sqrt(sigma)))

        return pd.DataFrame({
            "time": times,
            "survival": km_arr.tolist(),
            "ci_lo": np.clip(ci_lo, 0, 1).tolist(),
            "ci_hi": np.clip(ci_hi, 0, 1).tolist(),
            "n_at_risk": n_at_risk,
        })

    @staticmethod
    def log_rank_test(
        group1_times: np.ndarray,
        group1_events: np.ndarray,
        group2_times: np.ndarray,
        group2_events: np.ndarray,
    ) -> Dict[str, float]:
        """
        Log-rank test for difference in survival between two groups.

        Returns:
            dict: statistic, p_value, significant (p < 0.05)
        """
        # Combine and sort all event times
        all_times = np.unique(
            np.concatenate([
                group1_times[group1_events == 1],
                group2_times[group2_events == 1],
            ])
        )

        observed1, observed2 = 0, 0
        expected1, expected2 = 0.0, 0.0
        var1 = 0.0

        for t in all_times:
            n1 = (group1_times >= t).sum()
            n2 = (group2_times >= t).sum()
            N = n1 + n2
            if N <= 1:
                continue

            d1 = ((group1_times == t) & (group1_events == 1)).sum()
            d2 = ((group2_times == t) & (group2_events == 1)).sum()
            d = d1 + d2

            e1 = d * n1 / N
            observed1 += d1
            observed2 += d2
            expected1 += e1
            expected2 += d - e1

            # Variance term
            var1 += d * n1 * n2 * (N - d) / (N ** 2 * (N - 1) + 1e-10)

        if var1 <= 0:
            return {"statistic": float("nan"), "p_value": float("nan"), "significant": False}

        from scipy.stats import chi2
        stat = (observed1 - expected1) ** 2 / var1
        p_value = 1 - chi2.cdf(stat, df=1)

        return {
            "statistic": float(stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
        }


# ---------------------------------------------------------------------------
# Early Detection Sensitivity Analysis
# ---------------------------------------------------------------------------

class EarlyDetectionAnalyzer:
    """
    Analyse how sensitivity changes as a function of lead time before event.

    Computes the proportion of events detected at each lead time,
    given a fixed false positive rate (specificity).
    """

    def __init__(
        self,
        fixed_specificity: float = 0.90,
        lead_times: Optional[List[int]] = None,
    ) -> None:
        self.fixed_specificity = fixed_specificity
        self.lead_times = lead_times or [1, 2, 4, 6, 8, 12, 24]

    def compute(
        self,
        scores_df: pd.DataFrame,
        event_onset_hours: Dict[int, Optional[float]],
        icu_intimes: Dict[int, pd.Timestamp],
    ) -> pd.DataFrame:
        """
        Compute sensitivity at each lead time before event onset.

        Args:
            scores_df:          DataFrame with columns = icustay_ids, rows = hourly scores
            event_onset_hours:  {icustay_id: hours to event onset, or None}
            icu_intimes:        {icustay_id: ICU admission timestamp}

        Returns:
            DataFrame: lead_time_hours, sensitivity, n_detected, n_events
        """
        from sklearn.metrics import roc_curve

        # Collect all scores and determine threshold at target specificity
        all_scores = scores_df.values.flatten()
        all_scores = all_scores[~np.isnan(all_scores)]

        # Use 10th percentile of negative scores as approximate threshold
        # (simplified; in practice use a labeled validation set)
        threshold = np.percentile(all_scores, self.fixed_specificity * 100)

        rows = []
        for lt in self.lead_times:
            n_detected = 0
            n_events = 0

            for stay_id in scores_df.columns:
                event_h = event_onset_hours.get(stay_id)
                icu_intime = icu_intimes.get(stay_id)
                if event_h is None or icu_intime is None:
                    continue

                n_events += 1
                event_time = icu_intime + pd.Timedelta(hours=event_h)
                window_end = event_time - pd.Timedelta(hours=lt)

                # Check if any alert in [window_end - lt, window_end]
                stay_scores = scores_df[stay_id].dropna()
                alert_window = stay_scores[
                    stay_scores.index <= window_end
                ]
                if len(alert_window) > 0 and alert_window.max() >= threshold:
                    n_detected += 1

            sens = n_detected / max(n_events, 1)
            rows.append({
                "lead_time_hours": lt,
                "sensitivity": sens,
                "n_detected": n_detected,
                "n_events": n_events,
            })

        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Channel Importance Over Time
# ---------------------------------------------------------------------------

def channel_importance_timeline(
    attention_weights: np.ndarray,
    channel_names: List[str],
    hours_to_event: np.ndarray,
    n_bins: int = 12,
) -> pd.DataFrame:
    """
    Compute average channel attention weight as a function of hours before event.

    Shows which physiological channels become more important as the patient
    approaches a clinical event (e.g., MAP becomes dominant 2-4h before
    vasopressor initiation).

    Args:
        attention_weights: (N, C) per-sample per-channel attention weights
        channel_names:     List of C channel names
        hours_to_event:    (N,) hours from window to event onset (NaN = control)
        n_bins:            Number of time bins

    Returns:
        DataFrame: columns = channel_names, rows = time bins
    """
    valid = ~np.isnan(hours_to_event)
    weights = attention_weights[valid]
    times = hours_to_event[valid]

    # Bin the time-to-event
    bins = np.linspace(0, times.max() + 1, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    rows = []
    for i in range(n_bins):
        mask = (times >= bins[i]) & (times < bins[i + 1])
        if mask.sum() < 3:
            continue
        mean_weights = weights[mask].mean(axis=0)
        row = {"hours_to_event": float(bin_centers[i])}
        for j, ch in enumerate(channel_names):
            row[ch] = float(mean_weights[j])
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def generate_temporal_report(
    temporal_df: pd.DataFrame,
    model_name: str = "Transformer",
    baseline_auroc: float = 0.742,
) -> str:
    """Generate a formatted text report of temporal performance."""
    lines = [
        f"\n{'='*60}",
        f"TEMPORAL PERFORMANCE ANALYSIS: {model_name}",
        f"{'='*60}",
        f"\nAUROC vs. Hours Before Event (lower = further from event):",
        f"{'Horizon':>12} {'AUROC':>10} {'vs. Baseline':>14} {'N':>8}",
        f"{'─'*50}",
    ]

    for _, row in temporal_df.iterrows():
        delta = row["auroc"] - baseline_auroc
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"{row['horizon_hours']:>10.0f}h "
            f"{row['auroc']:>10.4f} "
            f"{sign}{delta:>12.4f} "
            f"{row.get('n', '?'):>8}"
        )

    lines.append(f"{'='*60}\n")
    return "\n".join(lines)


if __name__ == "__main__":
    np.random.seed(42)
    N = 500

    # Simulate temporal data
    survival_hours = np.random.exponential(48, N)
    events = np.random.binomial(1, 0.15, N)

    km = KaplanMeierAnalyzer(n_strata=3)
    peak_scores = np.random.randn(N)
    km_curves = km.compute(peak_scores, survival_hours, events,
                           labels=["Low risk", "Medium risk", "High risk"])

    for stratum, df in km_curves.items():
        print(f"\nKM Curve: {stratum}")
        print(df.head(5).to_string(index=False))

    # Log-rank test between low and high risk
    low_mask = peak_scores <= np.percentile(peak_scores, 33)
    high_mask = peak_scores >= np.percentile(peak_scores, 67)

    lr_result = KaplanMeierAnalyzer.log_rank_test(
        survival_hours[low_mask], events[low_mask],
        survival_hours[high_mask], events[high_mask],
    )
    print(f"\nLog-rank test: stat={lr_result['statistic']:.3f}, p={lr_result['p_value']:.4f}")
    print("Temporal analysis smoke test passed.")
