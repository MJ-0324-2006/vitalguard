"""
Clinical Explainability Engine for VitalGuard.

Generates human-readable, clinician-friendly explanations for deterioration alerts:
- Identifies specific vital sign drivers (e.g., progressive tachycardia, desaturation, hypotension)
- Traces longitudinal trends across consecutive readings (e.g., "82 -> 88 -> 96 -> 105 bpm")
- Calculates clinical flags (Shock Index elevation, low MAP, high NEWS2)
- Translates channel attention weights / error deviations into prioritized factors
- Provides actionable, protocol-aligned clinical guidance

NO UI code belongs in this module.
"""

from typing import Dict, List, Any, Optional


# Normal adult baseline ranges for ICU patients
CLINICAL_NORMALS = {
    "heart_rate":   {"low": 60.0,  "high": 90.0,  "label": "Heart Rate", "unit": "bpm"},
    "sbp":          {"low": 100.0, "high": 135.0, "label": "Systolic BP", "unit": "mmHg"},
    "dbp":          {"low": 60.0,  "high": 85.0,  "label": "Diastolic BP", "unit": "mmHg"},
    "resp_rate":    {"low": 12.0,  "high": 20.0,  "label": "Respiratory Rate", "unit": "/min"},
    "spo2":         {"low": 95.0,  "high": 100.0, "label": "Oxygen Saturation (SpO2)", "unit": "%"},
    "temperature":  {"low": 36.5,  "high": 37.5,  "label": "Temperature", "unit": "°C"},
    "gcs":          {"low": 14.0,  "high": 15.0,  "label": "GCS", "unit": "pts"},
}


def _fmt(val, default="--") -> str:
    if val is None or val == "" or val == "--":
        return default
    try:
        f = float(val)
        return f"{int(f)}" if f.is_integer() else f"{f:.1f}"
    except (ValueError, TypeError):
        return str(val)


def analyze_vital_trend(history: List[Dict[str, Any]], key: str) -> Optional[Dict[str, Any]]:
    """
    Analyzes historical readings for a single vital sign channel.
    Returns delta, percent change, direction, and arrow string (e.g., '82 -> 88 -> 96 -> 105 bpm').
    """
    values = []
    for entry in history:
        if key in entry and entry[key] is not None:
            try:
                values.append(float(entry[key]))
            except (ValueError, TypeError):
                pass

    if not values:
        return None

    current = values[-1]
    if len(values) == 1:
        return {
            "key": key,
            "current": current,
            "values": values,
            "trend_str": _fmt(current),
            "delta": 0.0,
            "pct_change": 0.0,
            "direction": "steady",
            "readings_count": 1,
        }

    # Use up to last 4-6 readings for the arrow string
    recent_vals = values[-4:] if len(values) >= 4 else values
    trend_str = " -> ".join([_fmt(v) for v in recent_vals])
    
    first = recent_vals[0]
    delta = round(current - first, 1)
    pct = round((delta / first * 100) if first != 0 else 0, 1)

    if delta > 1.0:
        direction = "rising"
    elif delta < -1.0:
        direction = "falling"
    else:
        direction = "stable"

    return {
        "key": key,
        "current": current,
        "values": values,
        "recent_vals": recent_vals,
        "trend_str": trend_str,
        "delta": delta,
        "pct_change": pct,
        "direction": direction,
        "readings_count": len(values),
    }


def generate_explanation(
    risk_level: str,
    vitals_history: List[Dict[str, Any]],
    channel_scores: Optional[Dict[str, float]] = None,
    derived_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generates a structured clinical explanation explaining WHY an alert was triggered.

    Returns:
        title (str): High-level medical summary
        key_findings (List[str]): Specific bullet points with trend evidence
        driver_chips (List[Dict]): Vital signs flagged with severity
        recommended_action (str): Standardized clinical protocol next steps
        trend_summary (str): Longitudinal context
    """
    if not vitals_history:
        return {
            "title": "Insufficient Data",
            "summary": "No vital signs recorded for this patient.",
            "key_findings": ["Awaiting initial vital signs entry."],
            "driver_chips": [],
            "recommended_action": "Record baseline vitals to initialize deterioration monitoring.",
            "trend_summary": "No historical points.",
        }

    current = vitals_history[-1]
    n_readings = len(vitals_history)
    key_findings = []
    driver_chips = []

    # 1. Analyze Individual Channels
    hr_trend = analyze_vital_trend(vitals_history, "heart_rate")
    sbp_trend = analyze_vital_trend(vitals_history, "sbp")
    rr_trend = analyze_vital_trend(vitals_history, "resp_rate")
    spo2_trend = analyze_vital_trend(vitals_history, "spo2")
    temp_trend = analyze_vital_trend(vitals_history, "temperature")

    # Heart Rate Evaluation
    if hr_trend:
        hr_val = hr_trend["current"]
        if hr_val >= 115:
            text = f"Severe tachycardia: {_fmt(hr_val)} bpm (normal 60-90)"
            if n_readings > 1 and hr_trend["direction"] == "rising":
                text += f" trending upward [{hr_trend['trend_str']} bpm, +{hr_trend['pct_change']}%]"
            key_findings.append(text)
            driver_chips.append({"name": "HR", "value": f"{_fmt(hr_val)} bpm", "severity": "critical"})
        elif hr_val >= 95:
            text = f"Elevated heart rate: {_fmt(hr_val)} bpm"
            if n_readings > 1 and hr_trend["direction"] == "rising":
                text += f" creeping up [{hr_trend['trend_str']} bpm]"
            key_findings.append(text)
            driver_chips.append({"name": "HR", "value": f"{_fmt(hr_val)} bpm", "severity": "warning"})
        elif hr_val <= 50:
            key_findings.append(f"Bradycardia detected: {_fmt(hr_val)} bpm")
            driver_chips.append({"name": "HR", "value": f"{_fmt(hr_val)} bpm", "severity": "warning"})

    # Blood Pressure & MAP Evaluation
    if sbp_trend:
        sbp_val = sbp_trend["current"]
        dbp_val = current.get("dbp", 80.0)
        map_val = derived_metrics.get("map_value") if derived_metrics else round(dbp_val + (sbp_val - dbp_val) / 3, 1)
        si_val = derived_metrics.get("shock_index") if derived_metrics else (round(current.get("heart_rate", 80) / sbp_val, 2) if sbp_val > 0 else 0)

        if map_val and map_val < 65.0:
            text = f"Critical hypotension: MAP is {map_val} mmHg (< 65 threshold indicates end-organ hypoperfusion)"
            if n_readings > 1 and sbp_trend["direction"] == "falling":
                text += f" with falling SBP [{sbp_trend['trend_str']} mmHg]"
            key_findings.append(text)
            driver_chips.append({"name": "MAP", "value": f"{map_val} mmHg", "severity": "critical"})
        elif sbp_val <= 90.0:
            text = f"Systemic hypotension: SBP {_fmt(sbp_val)} mmHg"
            if n_readings > 1 and sbp_trend["direction"] == "falling":
                text += f" dropping [{sbp_trend['trend_str']} mmHg]"
            key_findings.append(text)
            driver_chips.append({"name": "SBP", "value": f"{_fmt(sbp_val)} mmHg", "severity": "critical"})
        elif sbp_val <= 100.0:
            key_findings.append(f"Borderline systolic BP: {_fmt(sbp_val)} mmHg (approaching hypotension)")
            driver_chips.append({"name": "SBP", "value": f"{_fmt(sbp_val)} mmHg", "severity": "warning"})

        if si_val and si_val >= 0.9:
            key_findings.append(f"Elevated Shock Index ({si_val}): HR exceeds SBP, a strong indicator of early occult circulatory failure.")
            driver_chips.append({"name": "Shock Index", "value": str(si_val), "severity": "critical" if si_val >= 1.0 else "warning"})

    # Respiration Rate Evaluation
    if rr_trend:
        rr_val = rr_trend["current"]
        if rr_val >= 25:
            text = f"Marked tachypnea: {_fmt(rr_val)} breaths/min (normal 12-20)"
            if n_readings > 1 and rr_trend["direction"] == "rising":
                text += f" trending upward [{rr_trend['trend_str']}/min]"
            key_findings.append(text)
            driver_chips.append({"name": "RR", "value": f"{_fmt(rr_val)}/min", "severity": "critical"})
        elif rr_val >= 21:
            key_findings.append(f"Elevated respiratory rate: {_fmt(rr_val)} breaths/min (early respiratory compensation)")
            driver_chips.append({"name": "RR", "value": f"{_fmt(rr_val)}/min", "severity": "warning"})
        elif rr_val <= 8:
            key_findings.append(f"Depressed respiratory rate: {_fmt(rr_val)} breaths/min (risk of respiratory arrest)")
            driver_chips.append({"name": "RR", "value": f"{_fmt(rr_val)}/min", "severity": "critical"})

    # SpO2 Evaluation
    if spo2_trend:
        spo2_val = spo2_trend["current"]
        if spo2_val < 92.0:
            text = f"Severe hypoxemia: SpO2 at {_fmt(spo2_val)}% on monitored air"
            if n_readings > 1 and spo2_trend["direction"] == "falling":
                text += f" desaturating [{spo2_trend['trend_str']}%]"
            key_findings.append(text)
            driver_chips.append({"name": "SpO2", "value": f"{_fmt(spo2_val)}%", "severity": "critical"})
        elif spo2_val < 95.0:
            key_findings.append(f"Sub-optimal oxygenation: SpO2 {_fmt(spo2_val)}% (normal >= 95%)")
            driver_chips.append({"name": "SpO2", "value": f"{_fmt(spo2_val)}%", "severity": "warning"})

    # Temperature Evaluation
    if temp_trend:
        temp_val = temp_trend["current"]
        if temp_val >= 38.5:
            key_findings.append(f"Significant hyperthermia: {_fmt(temp_val)} °C (fever driving metabolic demand)")
            driver_chips.append({"name": "Temp", "value": f"{_fmt(temp_val)} °C", "severity": "warning"})
        elif temp_val >= 38.0:
            key_findings.append(f"Low-grade fever: {_fmt(temp_val)} °C")
        elif temp_val <= 35.5:
            key_findings.append(f"Hypothermia: {_fmt(temp_val)} °C (possible severe septic or metabolic dysregulation)")
            driver_chips.append({"name": "Temp", "value": f"{_fmt(temp_val)} °C", "severity": "critical"})

    # 2. Integrate driving factors from the single shared risk function if present
    if derived_metrics and derived_metrics.get("driving_factors"):
        for df in derived_metrics["driving_factors"]:
            if df not in key_findings and "within safe limits" not in df.lower():
                key_findings.insert(0, df)

    # 3. Add personalized baseline context if established
    is_personalized = derived_metrics.get("is_personalized", False) if derived_metrics else False
    personal_baselines = derived_metrics.get("personal_baselines", {}) if derived_metrics else {}

    if is_personalized and personal_baselines:
        base_items = [f"{k.replace('_', ' ').title()}: {_fmt(v)}" for k, v in personal_baselines.items()]
        base_summary = " &bull; ".join(base_items[:4])
        trend_summary = f"Evaluated against Patient's Personalized Baseline ({base_summary}) based on {n_readings} longitudinal observations."
    elif n_readings == 1:
        trend_summary = "Single-reading assessment (new admission). Personalized baseline activates after 3 observations."
    else:
        trend_summary = f"Evaluated against Standard Population ICU Thresholds ({n_readings}/3 readings charted towards personalized baseline)."

    # 4. Risk Tier Specific Summary & Guidance
    if risk_level == "RED":
        title = "CRITICAL ALERT: Acute Multi-System Deterioration"
        if not key_findings:
            key_findings.append("Cross-channel physiological deterioration detected.")
        recommended_action = (
            "Immediate bedside assessment required. Notify attending physician and Rapid Response Team (MET). "
            "Verify IV access, obtain STAT ABG and serum lactate, repeat vitals in 15 minutes."
        )
    elif risk_level == "YELLOW":
        title = "EARLY WARNING: Physiological Drift Detected"
        if not key_findings:
            key_findings.append("Insidious drift in vital signs detected across consecutive observations.")
        recommended_action = (
            "Increase monitoring frequency to every 30-60 minutes. Perform focused nursing assessment, "
            "check fluid balance/urine output, and inform covering doctor of emerging trend."
        )
    else:
        title = "PHYSIOLOGICALLY STABLE: Within Normal Limits"
        key_findings = [
            "All monitored parameters remain within safe baseline bounds.",
            "Longitudinal trajectory displays stable autonomic regulation.",
            f"Shock Index ({derived_metrics.get('shock_index', 0.6) if derived_metrics else 0.6}) and MAP remain optimal."
        ]
        driver_chips = [{"name": "All Vitals", "value": "Normal", "severity": "normal"}]
        recommended_action = "Maintain routine ward monitoring protocol according to clinical pathway."

    return {
        "title": title,
        "key_findings": key_findings,
        "driver_chips": driver_chips,
        "recommended_action": recommended_action,
        "trend_summary": trend_summary,
        "readings_count": n_readings,
        "is_personalized": is_personalized,
    }

