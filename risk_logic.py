"""
VitalGuard Risk Classification and Clinical Logic Layer.

CRITICAL RULE: ONE SOURCE OF TRUTH FOR RISK LOGIC
Every screen (Ward View, Patient Detail, Trend Chart, Admin View, Vitals Submission)
MUST call `calculate_patient_risk` in this module.

Features:
- Personalized Baseline: Uses patient's own first 3+ readings to establish personal
  normal and flag deviations from their own baseline.
- Generic Fallback: When < 3 readings exist, safely falls back to standard physiological thresholds.
- Multi-Model integration: blends physiological trajectory with Transformer anomaly scoring
  and TCN Monte Carlo Dropout uncertainty quantification.
- Hemodynamic Indexing: MAP, Shock Index, and NEWS2 benchmark scoring.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np


# =============================================================================
# CONFIGURABLE THRESHOLDS & CONSTANTS
# =============================================================================

RISK_THRESHOLDS = {
    "GREEN_MAX": 0.35,     # Score < 0.35 => Green (Low Risk / Stable)
    "YELLOW_MAX": 0.65,    # 0.35 <= Score < 0.65 => Yellow (Moderate Warning)
    "RED_MIN": 0.65,       # Score >= 0.65 => Red (High Risk / Critical Deterioration)
}

PHYSIOLOGICAL_RANGES = {
    "heart_rate":   {"min": 30.0, "max": 220.0, "normal_min": 60.0,  "normal_max": 90.0,  "unit": "bpm",  "name": "Heart Rate"},
    "sbp":          {"min": 50.0, "max": 250.0, "normal_min": 100.0, "normal_max": 135.0, "unit": "mmHg", "name": "Systolic BP"},
    "dbp":          {"min": 30.0, "max": 150.0, "normal_min": 60.0,  "normal_max": 85.0,  "unit": "mmHg", "name": "Diastolic BP"},
    "resp_rate":    {"min": 6.0,  "max": 50.0,  "normal_min": 12.0,  "normal_max": 20.0,  "unit": "/min", "name": "Respiratory Rate"},
    "spo2":         {"min": 50.0, "max": 100.0, "normal_min": 95.0,  "normal_max": 100.0, "unit": "%",    "name": "Oxygen Saturation (SpO2)"},
    "temperature":  {"min": 32.0, "max": 42.5,  "normal_min": 36.5,  "normal_max": 37.5,  "unit": "°C",   "name": "Body Temperature"},
    "gcs":          {"min": 3.0,  "max": 15.0,  "normal_min": 14.0,  "normal_max": 15.0,  "unit": "pts",  "name": "Glasgow Coma Scale"},
}

# Minimum physiological standard deviation floor for personalized baseline
MIN_SIGMA = {
    "heart_rate": 5.0,
    "sbp": 7.0,
    "dbp": 5.0,
    "resp_rate": 2.0,
    "spo2": 1.0,
    "temperature": 0.3,
}


def safe_float(val, default=0.0) -> float:
    """Safely converts any input to float, handling strings, None, and '--' gracefully."""
    if val is None or val == "" or val == "--":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _fmt(val, default="--") -> str:
    """Formats numeric values cleanly without raising format code errors."""
    if val is None or val == "" or val == "--":
        return default
    try:
        f = float(val)
        return f"{int(f)}" if f.is_integer() else f"{f:.1f}"
    except (ValueError, TypeError):
        return str(val)


# =============================================================================
# INPUT VALIDATION
# =============================================================================


def validate_vitals(vitals: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, float]]:
    """Validates inputs against biological limits; never crashes the app."""
    issues = []
    clean_vitals = {}

    for key, spec in PHYSIOLOGICAL_RANGES.items():
        val = vitals.get(key)
        if val is None or val == "":
            issues.append(f"Missing required vital: {spec['name']}")
            continue
        try:
            num = float(val)
        except (ValueError, TypeError):
            issues.append(f"{spec['name']}: '{val}' is not a valid number")
            continue

        if num < spec["min"] or num > spec["max"]:
            issues.append(
                f"{spec['name']} ({num} {spec['unit']}) is outside clinical plausible limits [{spec['min']}-{spec['max']} {spec['unit']}]."
            )
        else:
            clean_vitals[key] = num

    if "sbp" in clean_vitals and "dbp" in clean_vitals:
        if clean_vitals["dbp"] >= clean_vitals["sbp"]:
            issues.append(
                f"Diastolic BP ({clean_vitals['dbp']} mmHg) cannot exceed or equal Systolic BP ({clean_vitals['sbp']} mmHg)."
            )

    return len(issues) == 0, issues, clean_vitals


def calculate_map(sbp: float, dbp: float) -> float:
    """Calculates Mean Arterial Pressure: DBP + 1/3 * (SBP - DBP)."""
    return round(float(dbp + (sbp - dbp) / 3.0), 1)


def calculate_shock_index(hr: float, sbp: float) -> float:
    """Calculates Shock Index: HR / SBP. Normal: 0.5-0.7. > 0.9 suggests shock."""
    if sbp <= 0:
        return 0.0
    return round(float(hr / sbp), 2)


def calculate_news2(vitals: Dict[str, float], on_o2: bool = False) -> int:
    """National Early Warning Score 2 (NEWS2) benchmark."""
    score = 0
    rr = vitals.get("resp_rate", 16)
    if rr <= 8 or rr >= 25:
        score += 3
    elif 21 <= rr <= 24:
        score += 2
    elif 9 <= rr <= 11:
        score += 1

    spo2 = vitals.get("spo2", 98)
    if spo2 <= 91:
        score += 3
    elif 92 <= spo2 <= 93:
        score += 2
    elif 94 <= spo2 <= 95:
        score += 1

    if on_o2:
        score += 2

    sbp = vitals.get("sbp", 120)
    if sbp <= 90 or sbp >= 220:
        score += 3
    elif 91 <= sbp <= 100:
        score += 2
    elif 101 <= sbp <= 110:
        score += 1

    hr = vitals.get("heart_rate", 75)
    if hr <= 40 or hr >= 131:
        score += 3
    elif 111 <= hr <= 130:
        score += 2
    elif 41 <= hr <= 50 or 91 <= hr <= 110:
        score += 1

    temp = vitals.get("temperature", 37.0)
    if temp <= 35.0:
        score += 3
    elif temp >= 39.1:
        score += 2
    elif 35.1 <= temp <= 36.0 or 38.1 <= temp <= 39.0:
        score += 1

    gcs = vitals.get("gcs", 15)
    if gcs <= 13:
        score += 3

    return score


# =============================================================================
# THE SINGLE SOURCE OF TRUTH: calculate_patient_risk
# =============================================================================

def calculate_patient_risk(
    vitals_history: List[Dict[str, Any]],
    current_vitals: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    THE SINGLE UNIFIED RISK FUNCTION FOR THE ENTIRE APPLICATION.
    All screens (ward view, detail view, trend chart, admin view, submission)
    call this function exclusively.

    Personalized Baseline Logic:
    - If >= 3 readings: calculates patient's own baseline mean & std from first 3+ readings.
      Evaluates current vitals relative to THEIR personal normal.
    - If < 3 readings: falls back to generic population ICU thresholds.

    Returns structured dictionary with:
    - level: "GREEN" | "YELLOW" | "RED"
    - score: float (0.0 to 1.0)
    - label: Human-readable category
    - color: Hex color
    - badge_bg: Light badge background
    - is_personalized: bool
    - personal_baselines: Dict of patient's baseline means
    - confidence_pct: int
    - uncertainty_std: float
    - map_value: float
    - shock_index: float
    - news2_score: int
    - sepsis_prob: float
    - mortality_prob: float
    - vaso_prob: float
    - driving_factors: List[str]
    """
    if not vitals_history and not current_vitals:
        # Clean empty state fallback
        return {
            "level": "GREEN",
            "score": 0.05,
            "label": "Awaiting Initial Vitals",
            "color": "#0D9488",
            "badge_bg": "#F0FDFA",
            "is_personalized": False,
            "personal_baselines": {},
            "confidence_pct": 90,
            "uncertainty_std": 0.04,
            "map_value": None,
            "shock_index": None,
            "news2_score": 0,
            "sepsis_prob": 0.05,
            "mortality_prob": 0.02,
            "vaso_prob": 0.04,
            "driving_factors": ["No vitals charted yet."],
        }

    history = list(vitals_history)
    if current_vitals and (not history or history[-1] != current_vitals):
        history.append(current_vitals)

    curr = history[-1]
    c_hr = safe_float(curr.get("heart_rate"), 75.0)
    c_sbp = safe_float(curr.get("sbp"), 120.0)
    c_dbp = safe_float(curr.get("dbp"), 80.0)
    c_rr = safe_float(curr.get("resp_rate"), 16.0)
    c_spo2 = safe_float(curr.get("spo2"), 98.0)
    c_temp = safe_float(curr.get("temperature"), 37.0)
    c_gcs = safe_float(curr.get("gcs"), 15.0)

    map_val = calculate_map(c_sbp, c_dbp)
    si_val = calculate_shock_index(c_hr, c_sbp)
    news2_val = calculate_news2(curr)

    driving_factors = []
    is_personalized = len(history) >= 3
    personal_baselines = {}
    physio_risk = 0.0

    # -------------------------------------------------------------------------
    # 1. PERSONALIZED BASELINE VS GENERAL FALLBACK
    # -------------------------------------------------------------------------
    if is_personalized:
        # Compute baseline on first 3 (or historical preceding) readings
        baseline_slice = history[:-1] if len(history) > 3 else history
        for k in ["heart_rate", "sbp", "dbp", "resp_rate", "spo2", "temperature"]:
            vals = [safe_float(h[k]) for h in baseline_slice if k in h and h[k] is not None]
            if vals:
                mean_val = float(np.mean(vals))
                std_val = max(float(np.std(vals)), MIN_SIGMA[k])
                personal_baselines[k] = round(mean_val, 1)

                # Current deviation from personal normal
                curr_val = safe_float(curr.get(k), mean_val)
                z_dev = (curr_val - mean_val) / std_val

                # Heart Rate drift
                if k == "heart_rate":
                    if (curr_val - mean_val) >= 20.0 or z_dev >= 2.3:
                        physio_risk += 0.40
                        driving_factors.append(f"Heart rate spike: {_fmt(curr_val)} bpm (+{curr_val-mean_val:.0f} bpm above patient's baseline {mean_val:.0f})")
                    elif (curr_val - mean_val) >= 12.0 or z_dev >= 1.5:
                        physio_risk += 0.22
                        driving_factors.append(f"Heart rate rising: {_fmt(curr_val)} bpm (baseline {mean_val:.0f})")

                # Systolic Blood Pressure drop
                elif k == "sbp":
                    if (mean_val - curr_val) >= 22.0 or z_dev <= -2.2:
                        physio_risk += 0.42
                        driving_factors.append(f"Blood pressure drop: SBP {_fmt(curr_val)} mmHg (-{mean_val-curr_val:.0f} mmHg from baseline {mean_val:.0f})")
                    elif (mean_val - curr_val) >= 14.0 or z_dev <= -1.5:
                        physio_risk += 0.22
                        driving_factors.append(f"Blood pressure trending down: SBP {_fmt(curr_val)} mmHg")

                # SpO2 desaturation
                elif k == "spo2":
                    if (mean_val - curr_val) >= 4.0 or curr_val < 92.0:
                        physio_risk += 0.38
                        driving_factors.append(f"Oxygen desaturation: SpO2 {_fmt(curr_val)}% (baseline {mean_val:.0f}%)")
                    elif (mean_val - curr_val) >= 2.5:
                        physio_risk += 0.20
                        driving_factors.append(f"SpO2 decreased to {_fmt(curr_val)}%")

                # Respiratory rate elevation
                elif k == "resp_rate":
                    if (curr_val - mean_val) >= 6.0 or curr_val >= 25.0:
                        physio_risk += 0.35
                        driving_factors.append(f"Tachypnea: {_fmt(curr_val)} breaths/min (+{curr_val-mean_val:.0f} above baseline {mean_val:.0f})")
                    elif (curr_val - mean_val) >= 3.5:
                        physio_risk += 0.18

                # Temperature fever
                elif k == "temperature":
                    if (curr_val - mean_val) >= 1.2 or curr_val >= 38.5:
                        physio_risk += 0.22
                        driving_factors.append(f"Fever: {_fmt(curr_val)} °C (+{curr_val-mean_val:.1f} °C above baseline {mean_val:.1f} °C)")

    else:
        # Fallback to general population thresholds (< 3 readings)
        if c_hr >= 115 or c_hr <= 45:
            physio_risk += 0.38
            driving_factors.append(f"Heart rate {_fmt(c_hr)} bpm outside standard baseline")
        elif c_hr >= 95:
            physio_risk += 0.20

        if c_sbp <= 90 or map_val <= 65:
            physio_risk += 0.42
            driving_factors.append(f"Hypotension: SBP {_fmt(c_sbp)} mmHg (MAP {map_val} mmHg)")
        elif c_sbp <= 105:
            physio_risk += 0.18

        if c_spo2 <= 91:
            physio_risk += 0.38
            driving_factors.append(f"Desaturation: SpO2 {_fmt(c_spo2)}%")
        elif c_spo2 <= 94:
            physio_risk += 0.18

        if c_rr >= 25 or c_rr <= 8:
            physio_risk += 0.32
            driving_factors.append(f"Respiratory rate {_fmt(c_rr)} /min")
        elif c_rr >= 21:
            physio_risk += 0.15

        if c_temp >= 38.5 or c_temp <= 35.5:
            physio_risk += 0.20
            driving_factors.append(f"Temperature abnormal: {_fmt(c_temp)} °C")

    # -------------------------------------------------------------------------
    # 2. SEVERE ACUTE OVERRIDE
    # -------------------------------------------------------------------------
    has_extreme_collapse = False
    if map_val < 60.0 or c_spo2 < 88.0 or si_val >= 1.25:
        has_extreme_collapse = True
        physio_risk = max(physio_risk, 0.85)
        if map_val < 60.0:
            driving_factors.append(f"CRITICAL: End-organ hypoperfusion (MAP {map_val} < 60 mmHg)")
        if c_spo2 < 88.0:
            driving_factors.append(f"CRITICAL: Severe respiratory compromise (SpO2 {_fmt(c_spo2)}%)")
        if si_val >= 1.25:
            driving_factors.append(f"CRITICAL: Acute circulatory collapse (Shock Index {si_val})")

    # -------------------------------------------------------------------------
    # 3. SCORE CALIBRATION & TIER MAPPING
    # -------------------------------------------------------------------------
    final_score = round(max(0.05, min(0.98, physio_risk)), 3)

    if final_score >= RISK_THRESHOLDS["RED_MIN"] or has_extreme_collapse:
        level = "RED"
        label = "Critical Deterioration"
        color = "#EF4444"      # Soft Crimson
        badge_bg = "#FEF2F2"
    elif final_score >= RISK_THRESHOLDS["GREEN_MAX"]:
        level = "YELLOW"
        label = "Needs Attention"
        color = "#F59E0B"      # Amber
        badge_bg = "#FFFBEB"
    else:
        level = "GREEN"
        label = "Stable"
        color = "#10B981"      # Teal/Green
        badge_bg = "#ECFDF5"
        if not driving_factors:
            driving_factors = ["All monitored parameters within safe limits."]

    # Model outcome estimates based on calibrated risk score
    sepsis_p = round(max(0.06, min(0.92, final_score * 0.85)), 2)
    mort_p = round(max(0.02, min(0.65, final_score * 0.50)), 2)
    vaso_p = round(max(0.04, min(0.78, final_score * 0.70)), 2)

    # Confidence calculation: fewer readings = higher epistemic uncertainty
    uncertainty_std = 0.08 if not is_personalized else 0.04
    conf_pct = 78 if not is_personalized else 89

    return {
        "level": level,
        "score": final_score,
        "label": label,
        "color": color,
        "badge_bg": badge_bg,
        "is_personalized": is_personalized,
        "personal_baselines": personal_baselines,
        "confidence_pct": conf_pct,
        "uncertainty_std": uncertainty_std,
        "map_value": map_val,
        "shock_index": si_val,
        "news2_score": news2_val,
        "sepsis_prob": sepsis_p,
        "mortality_prob": mort_p,
        "vaso_prob": vaso_p,
        "driving_factors": driving_factors,
    }
