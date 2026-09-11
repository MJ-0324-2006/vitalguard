"""
Clinical Evaluation Metrics for ICU Anomaly Detection.

Computes:
  - AUROC and AUPRC per clinical endpoint
  - Sensitivity at fixed specificity (95%, 99%)
  - Alarm rate analysis (alarms per patient-day)
  - Time-to-detection (hours before clinical recognition)
  - Alarm fatigue metrics (NNE, positive predictive value)
  - Calibration (ECE, reliability diagrams)
  - Confidence intervals via bootstrap

These metrics are the standard reporting framework for clinical early
warning system papers (cf. Harutyunyan et al. 2019, Rajpurkar et al. 2017).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core classification metrics
# ---------------------------------------------------------------------------

def compute_auroc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    ci: bool = False,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
) -> Dict[str, float]:
    """
    Compute AUROC with optional bootstrap confidence interval.

    Args:
        y_true:      Binary labels (0/1)
        y_score:     Predicted scores (higher = more likely positive)
        ci:          If True, compute bootstrap CI
        n_bootstrap: Bootstrap iterations
        ci_level:    Confidence level (e.g., 0.95 for 95% CI)

    Returns:
        dict with 'auroc', and optionally 'auroc_ci_lo', 'auroc_ci_hi'
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    # Remove NaN labels
    valid = ~(np.isnan(y_true) | np.isnan(y_score))
    y_true = y_true[valid].astype(int)
    y_score = y_score[valid]

    if len(np.unique(y_true)) < 2:
        return {"auroc": float("nan")}

    auroc = roc_auc_score(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    result = {
        "auroc": float(auroc),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "n_positive": int(y_true.sum()),
        "n_negative": int((1 - y_true).sum()),
    }

    if ci:
        boot_scores = []
        rng = np.random.RandomState(42)
        for _ in range(n_bootstrap):
            idx = rng.randint(0, len(y_true), len(y_true))
            if len(np.unique(y_true[idx])) < 2:
                continue
            try:
                boot_scores.append(roc_auc_score(y_true[idx], y_score[idx]))
            except Exception:
                pass
        alpha = (1 - ci_level) / 2
        result["auroc_ci_lo"] = float(np.percentile(boot_scores, 100 * alpha))
        result["auroc_ci_hi"] = float(np.percentile(boot_scores, 100 * (1 - alpha)))

    return result


def compute_auprc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    ci: bool = False,
    n_bootstrap: int = 1000,
) -> Dict[str, float]:
    """
    Compute AUPRC (area under precision-recall curve).

    AUPRC is more informative than AUROC for imbalanced clinical datasets.
    A random classifier achieves AUPRC = prevalence (not 0.5).

    Returns:
        dict with 'auprc', 'ap' (average precision), 'prevalence'
    """
    from sklearn.metrics import average_precision_score, precision_recall_curve

    valid = ~(np.isnan(y_true) | np.isnan(y_score))
    y_true = y_true[valid].astype(int)
    y_score = y_score[valid]

    if len(np.unique(y_true)) < 2:
        return {"auprc": float("nan"), "ap": float("nan")}

    ap = average_precision_score(y_true, y_score)
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    prevalence = y_true.mean()

    result = {
        "auprc": float(ap),
        "ap": float(ap),
        "prevalence": float(prevalence),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "pr_thresholds": thresholds.tolist(),
    }

    if ci:
        boot_scores = []
        rng = np.random.RandomState(42)
        for _ in range(n_bootstrap):
            idx = rng.randint(0, len(y_true), len(y_true))
            if len(np.unique(y_true[idx])) < 2:
                continue
            try:
                boot_scores.append(average_precision_score(y_true[idx], y_score[idx]))
            except Exception:
                pass
        result["auprc_ci_lo"] = float(np.percentile(boot_scores, 2.5))
        result["auprc_ci_hi"] = float(np.percentile(boot_scores, 97.5))

    return result


def sensitivity_at_fixed_specificity(
    y_true: np.ndarray,
    y_score: np.ndarray,
    specificities: List[float] = (0.90, 0.95, 0.99),
) -> Dict[str, Dict[str, float]]:
    """
    Compute sensitivity (recall) at fixed specificity values.

    Clinically meaningful because:
      - 90% specificity → 10 false alarms per 100 non-events
      - 95% specificity → 5 false alarms per 100 non-events
      - 99% specificity → 1 false alarm per 100 non-events

    Returns:
        dict mapping specificity_str → {'sensitivity', 'threshold', 'ppv'}
    """
    from sklearn.metrics import roc_curve

    valid = ~(np.isnan(y_true) | np.isnan(y_score))
    y_true = y_true[valid].astype(int)
    y_score = y_score[valid]

    if len(np.unique(y_true)) < 2:
        return {}

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    spec = 1 - fpr  # specificity = 1 - FPR

    results = {}
    for target_spec in specificities:
        # Find the threshold achieving >= target specificity
        idx = np.where(spec >= target_spec)[0]
        if len(idx) == 0:
            continue
        # Among thresholds with spec >= target, take the one with highest sensitivity
        best_idx = idx[np.argmax(tpr[idx])]

        sens = float(tpr[best_idx])
        thresh = float(thresholds[best_idx])

        # Compute PPV at this threshold
        y_pred = (y_score >= thresh).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        results[f"spec_{int(target_spec*100)}"] = {
            "sensitivity": sens,
            "specificity": float(spec[best_idx]),
            "threshold": thresh,
            "ppv": ppv,
            "nne": 1 / ppv if ppv > 0 else float("inf"),  # Number Needed to Evaluate
        }

    return results


# ---------------------------------------------------------------------------
# Alarm Rate Analysis
# ---------------------------------------------------------------------------

def compute_alarm_rate(
    y_score: np.ndarray,
    threshold: float,
    patient_hours: float,
    min_gap_hours: int = 1,
    time_index: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute alarm rate (alarms per patient-day) at a given threshold.

    Applies minimum inter-alarm gap to prevent alarm bursts counting as
    multiple alarms (clinically, a persistent alarm is one alarm event).

    Args:
        y_score:       (N,) anomaly scores (hourly)
        threshold:     Alert trigger threshold
        patient_hours: Total patient-hours of observation
        min_gap_hours: Minimum hours between counted alarms
        time_index:    (N,) hour indices (default = 0,1,2,...N)

    Returns:
        dict: alarms_per_day, total_alarms, alarm_duration_hours
    """
    alerts = y_score >= threshold

    if time_index is None:
        time_index = np.arange(len(y_score))

    # Apply minimum gap filtering (alarm persistence deduplication)
    alert_times = time_index[alerts]
    if len(alert_times) == 0:
        return {
            "alarms_per_day": 0.0,
            "total_alarms": 0,
            "alarm_hours_per_day": 0.0,
        }

    # Deduplicate with min gap
    deduplicated_times = [alert_times[0]]
    for t in alert_times[1:]:
        if t - deduplicated_times[-1] >= min_gap_hours:
            deduplicated_times.append(t)

    total_alarms = len(deduplicated_times)
    patient_days = patient_hours / 24
    alarms_per_day = total_alarms / max(patient_days, 1)
    alarm_hours_per_day = alerts.sum() / max(patient_days, 1)

    return {
        "alarms_per_day": float(alarms_per_day),
        "total_alarms": total_alarms,
        "alarm_hours_per_day": float(alarm_hours_per_day),
        "patient_days": float(patient_days),
    }


def alarm_rate_vs_sensitivity(
    y_true: np.ndarray,
    y_score: np.ndarray,
    patient_hours: float,
    sensitivity_targets: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Compute alarm rate at various sensitivity levels.

    Returns DataFrame with columns:
        sensitivity, specificity, threshold, alarms_per_day, ppv, nne
    """
    from sklearn.metrics import roc_curve

    if sensitivity_targets is None:
        sensitivity_targets = [0.70, 0.80, 0.85, 0.90, 0.92, 0.95]

    valid = ~(np.isnan(y_true) | np.isnan(y_score))
    y_true = y_true[valid].astype(int)
    y_score = y_score[valid]

    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    rows = []
    for target_sens in sensitivity_targets:
        idx = np.where(tpr >= target_sens)[0]
        if len(idx) == 0:
            continue
        best_idx = idx[0]
        thresh = float(thresholds[best_idx])

        y_pred = (y_score >= thresh).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        alarm_stats = compute_alarm_rate(y_score, thresh, patient_hours)

        rows.append({
            "sensitivity": float(tpr[best_idx]),
            "specificity": float(1 - fpr[best_idx]),
            "threshold": thresh,
            "alarms_per_day": alarm_stats["alarms_per_day"],
            "ppv": ppv,
            "nne": 1 / ppv if ppv > 0 else float("inf"),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Time-to-Detection
# ---------------------------------------------------------------------------

def compute_time_to_detection(
    event_times: List[float],
    detection_times: List[Optional[float]],
    max_horizon: float = 12.0,
) -> Dict[str, float]:
    """
    Compute time-to-detection statistics across all detected events.

    Args:
        event_times:     List of actual event onset times (hours from ICU admit)
        detection_times: List of first alert times for each event (None if missed)
        max_horizon:     Maximum hours before event to count as early detection

    Returns:
        dict: median_ttd, mean_ttd, detected_frac,
              early_detection_frac (TTD > 2h), very_early_frac (TTD > 4h)
    """
    ttd_values = []
    detected = 0

    for event_time, detect_time in zip(event_times, detection_times):
        if detect_time is None:
            continue
        if detect_time < event_time:  # alert before event
            ttd = event_time - detect_time
            if ttd <= max_horizon:
                ttd_values.append(ttd)
                detected += 1

    if len(ttd_values) == 0:
        return {
            "median_ttd_hours": 0.0,
            "mean_ttd_hours": 0.0,
            "detected_frac": 0.0,
            "early_detection_frac": 0.0,
            "very_early_frac": 0.0,
        }

    ttd_arr = np.array(ttd_values)
    return {
        "median_ttd_hours": float(np.median(ttd_arr)),
        "mean_ttd_hours": float(np.mean(ttd_arr)),
        "std_ttd_hours": float(np.std(ttd_arr)),
        "detected_frac": detected / len(event_times),
        "early_detection_frac": float((ttd_arr >= 2.0).mean()),   # >= 2h early
        "very_early_frac": float((ttd_arr >= 4.0).mean()),        # >= 4h early
        "n_detected": detected,
        "n_events": len(event_times),
    }


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Compute Expected Calibration Error (ECE).

    ECE = sum_b (|B_b| / N) * |acc_b - conf_b|

    Well-calibrated model: ECE < 0.03 (3%).

    Returns:
        dict: ece, mce (maximum calibration error), bin_data
    """
    valid = ~(np.isnan(y_true) | np.isnan(y_prob))
    y_true = y_true[valid].astype(float)
    y_prob = y_prob[valid]

    bins = np.linspace(0, 1, n_bins + 1)
    bin_data = []
    ece = 0.0
    mce = 0.0
    N = len(y_true)

    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_conf = y_prob[mask].mean()
        bin_acc = y_true[mask].mean()
        bin_n = mask.sum()
        weight = bin_n / N

        cal_err = abs(bin_acc - bin_conf)
        ece += weight * cal_err
        mce = max(mce, cal_err)

        bin_data.append({
            "bin_lo": bins[i], "bin_hi": bins[i+1],
            "confidence": bin_conf, "accuracy": bin_acc, "n": bin_n,
        })

    return {
        "ece": float(ece),
        "mce": float(mce),
        "bin_data": bin_data,
        "n_bins": n_bins,
    }


# ---------------------------------------------------------------------------
# Number Needed to Evaluate (NNE)
# ---------------------------------------------------------------------------

def compute_nne(ppv: float) -> float:
    """
    Number Needed to Evaluate (NNE) = 1 / PPV.

    Clinically: how many patients evaluated for every true positive found.
    Lower NNE = better precision = less burden on clinical staff.
    NNE of 1.0 = every alarm is a true positive (ideal).
    NEWS2 typically has NNE ~8-12; better systems achieve NNE ~4-6.
    """
    return 1.0 / ppv if ppv > 0 else float("inf")


# ---------------------------------------------------------------------------
# Comprehensive Evaluation
# ---------------------------------------------------------------------------

class ClinicalEvaluator:
    """
    Compute full clinical evaluation metrics for an ICU anomaly detection model.

    Usage:
        evaluator = ClinicalEvaluator(outcomes=["mortality", "sepsis", "vasopressor"])
        report = evaluator.evaluate(
            predictions={"mortality": scores_arr, "sepsis": sepsis_scores},
            labels={"mortality": y_mortality, "sepsis": y_sepsis},
            patient_hours=total_hours,
        )
        evaluator.print_report(report)
    """

    def __init__(
        self,
        outcomes: Optional[List[str]] = None,
        sensitivity_targets: Optional[List[float]] = None,
        bootstrap_ci: bool = True,
        n_bootstrap: int = 1000,
    ) -> None:
        self.outcomes = outcomes or ["mortality", "sepsis", "vasopressor"]
        self.sensitivity_targets = sensitivity_targets or [0.80, 0.90, 0.95]
        self.bootstrap_ci = bootstrap_ci
        self.n_bootstrap = n_bootstrap

    def evaluate(
        self,
        predictions: Dict[str, np.ndarray],
        labels: Dict[str, np.ndarray],
        patient_hours: Optional[float] = None,
        event_times: Optional[Dict[str, List[float]]] = None,
        detection_times: Optional[Dict[str, List[Optional[float]]]] = None,
    ) -> Dict[str, Dict]:
        """
        Compute full evaluation for all outcomes.

        Args:
            predictions:    Dict outcome → (N,) score array
            labels:         Dict outcome → (N,) binary label array
            patient_hours:  Total observation hours for alarm rate computation
            event_times:    Dict outcome → list of event onset hours
            detection_times:Dict outcome → list of first alert hours

        Returns:
            report: nested dict, metrics per outcome
        """
        report = {}

        for outcome in self.outcomes:
            if outcome not in predictions or outcome not in labels:
                continue

            y_score = np.array(predictions[outcome])
            y_true = np.array(labels[outcome])

            valid = ~np.isnan(y_true)
            y_score = y_score[valid]
            y_true = y_true[valid]

            if len(np.unique(y_true.astype(int))) < 2:
                logger.warning(f"Skipping {outcome}: only one class present")
                continue

            outcome_report = {}

            # AUROC
            outcome_report["auroc"] = compute_auroc(
                y_true, y_score, ci=self.bootstrap_ci, n_bootstrap=self.n_bootstrap
            )

            # AUPRC
            outcome_report["auprc"] = compute_auprc(
                y_true, y_score, ci=self.bootstrap_ci, n_bootstrap=self.n_bootstrap
            )

            # Sensitivity at fixed specificity
            outcome_report["sens_at_spec"] = sensitivity_at_fixed_specificity(
                y_true, y_score, [0.90, 0.95, 0.99]
            )

            # Alarm rate curve
            if patient_hours is not None:
                outcome_report["alarm_rate_curve"] = alarm_rate_vs_sensitivity(
                    y_true, y_score, patient_hours, self.sensitivity_targets
                )

            # Calibration
            # Convert scores to probabilities for calibration
            from scipy.special import expit
            y_prob = expit(y_score) if y_score.max() > 1 or y_score.min() < 0 else y_score
            outcome_report["calibration"] = expected_calibration_error(y_true, y_prob)

            # Time to detection
            if (event_times and outcome in event_times and
                    detection_times and outcome in detection_times):
                outcome_report["ttd"] = compute_time_to_detection(
                    event_times[outcome], detection_times[outcome]
                )

            report[outcome] = outcome_report

        return report

    @staticmethod
    def print_report(report: Dict[str, Dict]) -> None:
        """Print formatted evaluation report to stdout."""
        print("\n" + "="*70)
        print("CLINICAL EVALUATION REPORT")
        print("="*70)

        for outcome, metrics in report.items():
            print(f"\n{'─'*50}")
            print(f"Outcome: {outcome.upper()}")
            print(f"{'─'*50}")

            if "auroc" in metrics:
                auroc_data = metrics["auroc"]
                auroc_val = auroc_data.get("auroc", float("nan"))
                ci_lo = auroc_data.get("auroc_ci_lo", float("nan"))
                ci_hi = auroc_data.get("auroc_ci_hi", float("nan"))
                n_pos = auroc_data.get("n_positive", "?")
                n_neg = auroc_data.get("n_negative", "?")
                print(f"  AUROC:  {auroc_val:.4f} (95% CI: {ci_lo:.4f}–{ci_hi:.4f})")
                print(f"  N:      {n_pos} positive, {n_neg} negative")

            if "auprc" in metrics:
                auprc_data = metrics["auprc"]
                print(
                    f"  AUPRC:  {auprc_data.get('auprc', float('nan')):.4f} "
                    f"(prevalence: {auprc_data.get('prevalence', float('nan'))*100:.1f}%)"
                )

            if "sens_at_spec" in metrics:
                print("\n  Sensitivity at fixed specificity:")
                for spec_key, vals in metrics["sens_at_spec"].items():
                    print(
                        f"    {spec_key}: Sens={vals['sensitivity']:.3f} "
                        f"PPV={vals['ppv']:.3f} "
                        f"NNE={vals['nne']:.1f}"
                    )

            if "alarm_rate_curve" in metrics and isinstance(metrics["alarm_rate_curve"], pd.DataFrame):
                df = metrics["alarm_rate_curve"]
                print("\n  Alarm rate vs. sensitivity:")
                for _, row in df.iterrows():
                    print(
                        f"    Sens={row['sensitivity']:.2f}: "
                        f"{row['alarms_per_day']:.2f} alarms/pt-day "
                        f"(PPV={row['ppv']:.3f})"
                    )

            if "calibration" in metrics:
                cal = metrics["calibration"]
                print(f"\n  ECE: {cal.get('ece', float('nan')):.4f}  "
                      f"MCE: {cal.get('mce', float('nan')):.4f}")

            if "ttd" in metrics:
                ttd = metrics["ttd"]
                print(
                    f"\n  Time to Detection: median {ttd['median_ttd_hours']:.1f}h "
                    f"({ttd['detected_frac']*100:.0f}% detected, "
                    f"{ttd['very_early_frac']*100:.0f}% >= 4h early)"
                )

        print("\n" + "="*70)

    def to_dataframe(self, report: Dict[str, Dict]) -> pd.DataFrame:
        """Convert report to summary DataFrame for easy export."""
        rows = []
        for outcome, metrics in report.items():
            row = {"outcome": outcome}
            if "auroc" in metrics:
                row["auroc"] = metrics["auroc"].get("auroc")
                row["auroc_ci_lo"] = metrics["auroc"].get("auroc_ci_lo")
                row["auroc_ci_hi"] = metrics["auroc"].get("auroc_ci_hi")
            if "auprc" in metrics:
                row["auprc"] = metrics["auprc"].get("auprc")
            if "sens_at_spec" in metrics and "spec_90" in metrics["sens_at_spec"]:
                s90 = metrics["sens_at_spec"]["spec_90"]
                row["sens_at_90spec"] = s90.get("sensitivity")
                row["ppv_at_90spec"] = s90.get("ppv")
                row["nne_at_90spec"] = s90.get("nne")
            if "calibration" in metrics:
                row["ece"] = metrics["calibration"].get("ece")
            rows.append(row)
        return pd.DataFrame(rows)


if __name__ == "__main__":
    np.random.seed(42)
    N = 1000

    # Simulate predictions for 3 outcomes
    predictions = {}
    labels = {}
    for outcome, prevalence in [("mortality", 0.10), ("sepsis", 0.15), ("vasopressor", 0.18)]:
        y = np.random.binomial(1, prevalence, N).astype(float)
        # Good model: score correlates with label
        score = y * np.random.normal(1.5, 1.0, N) + (1-y) * np.random.normal(0, 1.0, N)
        predictions[outcome] = score
        labels[outcome] = y

    evaluator = ClinicalEvaluator(bootstrap_ci=True, n_bootstrap=200)
    report = evaluator.evaluate(
        predictions, labels, patient_hours=N * 1.5  # 1.5h per window on average
    )
    evaluator.print_report(report)

    df = evaluator.to_dataframe(report)
    print("\nSummary DataFrame:")
    print(df.to_string(index=False))

# NEWS2 (National Early Warning Score 2) baseline comparison
# used as a clinical benchmark against the ML model
NEWS2_THRESHOLDS = {
    'rr': [(0, 8, 3), (8, 11, 1), (12, 20, 0), (21, 24, 2), (25, float('inf'), 3)],
    'spo2': [(0, 91, 3), (92, 93, 2), (94, 95, 1), (96, 100, 0)],
    'sbp': [(0, 90, 3), (91, 100, 2), (101, 110, 1), (111, 219, 0), (220, float('inf'), 3)],
    'hr': [(0, 40, 3), (41, 50, 1), (51, 90, 0), (91, 110, 1), (111, 130, 2), (131, float('inf'), 3)],
    'temp': [(0, 35, 3), (35.1, 36.0, 1), (36.1, 38.0, 0), (38.1, 39.0, 1), (39.1, float('inf'), 2)],
}

def compute_news2_score(vitals_row):
    """compute NEWS2 score from a single row of vital signs"""
    total = 0
    for param, ranges in NEWS2_THRESHOLDS.items():
        if param in vitals_row:
            val = vitals_row[param]
            for low, high, score in ranges:
                if low <= val < high:
                    total += score
                    break
    return total
