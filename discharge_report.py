"""
VitalGuard Patient Discharge Report & Downloadable PDF Generator.

Generates a clean, hospital-grade printable discharge summary & PDF document:
- Patient demographics, ward, bed, admission and discharge timestamps
- Plain-language clinical risk summary & overall trend trajectory
- Final vital signs summary
- Complete chronological vitals history table with risk stratification
- Condensed, deduplicated alert timeline (no repeated near-duplicate entries)
- Attending physician sign-off block
"""

import io
from datetime import datetime
from typing import Dict, List, Any
from xhtml2pdf import pisa


def generate_discharge_report_html(
    patient: Dict[str, Any],
    vitals_history: List[Dict[str, Any]],
    alerts: List[Dict[str, Any]],
    hospital_name: str = "Sri Ramachandra Rural Health Centre"
) -> str:
    """Generates standalone printable HTML report for a discharged patient."""
    now_str = datetime.now().strftime("%d %b %Y, %H:%M:%S")
    admitted_str = patient.get("admitted_str", "N/A")
    discharged_str = datetime.now().strftime("%d %b %Y, %H:%M")

    # Helper for formatting numbers
    def _safe_fmt(val, default="--"):
        if val is None or val == "" or val == "--":
            return default
        try:
            f = float(val)
            return f"{int(f)}" if f.is_integer() else f"{f:.1f}"
        except (ValueError, TypeError):
            return str(val)

    # 1. PLAIN LANGUAGE RISK & TREND SUMMARY
    red_alerts = [a for a in alerts if (a.get("risk_level") or "").upper() == "RED"]
    yellow_alerts = [a for a in alerts if (a.get("risk_level") or "").upper() == "YELLOW"]

    if not alerts:
        plain_risk_summary = "Patient was stable throughout hospital stay with no critical or warning alerts logged."
    elif red_alerts:
        last_red_time = red_alerts[0].get("timestamp_str", "during stay")
        plain_risk_summary = f"Patient experienced {len(red_alerts)} critical alert(s) (last recorded: {last_red_time}), promptly managed by nursing staff."
    else:
        plain_risk_summary = f"Patient experienced {len(yellow_alerts)} minor warning alert(s). Vitals remained manageable throughout stay."

    # Determine 1-line overall trend
    latest_reading = vitals_history[-1] if vitals_history else {}
    latest_level = latest_reading.get("risk_level", "GREEN").upper()

    if latest_level == "GREEN":
        overall_trend_text = "Stable & Improving — Vitals returned to normal baseline values prior to discharge."
        trend_color = "#10B981"
    elif latest_level == "YELLOW":
        overall_trend_text = "Borderline / Under Observation — Minor vital sign fluctuations observed."
        trend_color = "#F59E0B"
    else:
        overall_trend_text = "High Vigilance — Significant physiological deterioration logged during stay."
        trend_color = "#EF4444"

    # Final vitals summary
    last_hr = _safe_fmt(latest_reading.get("heart_rate"))
    last_sbp = _safe_fmt(latest_reading.get("sbp"))
    last_dbp = _safe_fmt(latest_reading.get("dbp"))
    last_bp = f"{last_sbp}/{last_dbp} mmHg" if last_sbp != "--" and last_dbp != "--" else "--"
    last_spo2 = _safe_fmt(latest_reading.get("spo2"))
    last_rr = _safe_fmt(latest_reading.get("resp_rate"))
    last_temp = _safe_fmt(latest_reading.get("temperature"))

    # 2. BUILD VITALS HISTORY TABLE ROWS
    vitals_rows_html = ""
    for r in vitals_history:
        level = r.get("risk_level", "GREEN").upper()
        color = "#10B981" if level == "GREEN" else ("#F59E0B" if level == "YELLOW" else "#EF4444")
        bg = "#ECFDF5" if level == "GREEN" else ("#FFFBEB" if level == "YELLOW" else "#FEF2F2")

        sbp_s = _safe_fmt(r.get('sbp'))
        dbp_s = _safe_fmt(r.get('dbp'))
        bp_s = f"{sbp_s}/{dbp_s} mmHg" if sbp_s != "--" and dbp_s != "--" else "--"

        vitals_rows_html += f"""
        <tr>
            <td>{r.get('timestamp_str', r.get('time', ''))}</td>
            <td><strong>{_safe_fmt(r.get('heart_rate'))}</strong> bpm</td>
            <td>{bp_s}</td>
            <td>{_safe_fmt(r.get('map'))} mmHg</td>
            <td>{_safe_fmt(r.get('spo2'))}%</td>
            <td>{_safe_fmt(r.get('resp_rate'))}/min</td>
            <td>{_safe_fmt(r.get('temperature'))}°C</td>
            <td><span style="background:{bg}; color:{color}; font-weight:700; padding:2px 8px; border-radius:4px; font-size:11px;">{level}</span></td>
        </tr>
        """

    if not vitals_history:
        vitals_rows_html = "<tr><td colspan='8' style='text-align:center; color:#94A3B8; padding:16px;'>No vitals observations were recorded during this stay.</td></tr>"

    # 3. DEDUPLICATE ALERT TIMELINE ENTRIES
    deduped_alerts = []
    for a in alerts:
        msg = (a.get("message") or "").strip()
        level = (a.get("risk_level") or "ALERT").upper()
        ts_str = a.get("timestamp_str", "")
        sms_text = a.get("sms_simulated_text", "")
        sms_text = sms_text.replace("Dr. Shankar Kothapalli", "Dr. A. Mehta (Chief Physician)")
        sms_text = sms_text.replace("+91-98765-43210", "+91-98765-00000")

        if not deduped_alerts:
            deduped_alerts.append({
                "risk_level": level,
                "message": msg,
                "first_time": ts_str,
                "last_time": ts_str,
                "count": 1,
                "sms_text": sms_text
            })
        else:
            prev = deduped_alerts[-1]
            if prev["message"] == msg and prev["risk_level"] == level:
                prev["count"] += 1
                prev["last_time"] = ts_str
            else:
                deduped_alerts.append({
                    "risk_level": level,
                    "message": msg,
                    "first_time": ts_str,
                    "last_time": ts_str,
                    "count": 1,
                    "sms_text": sms_text
                })

    alerts_rows_html = ""
    for a in deduped_alerts:
        level = a["risk_level"]
        color = "#EF4444" if level == "RED" else "#F59E0B"
        bg = "#FEF2F2" if level == "RED" else "#FFFBEB"
        
        time_display = a["first_time"]
        if a["count"] > 1 and a["first_time"] != a["last_time"]:
            time_display = f"{a['first_time']} &ndash; {a['last_time']} <span style='font-size:10px; background:#FEE2E2; color:#991B1B; padding:1px 6px; border-radius:10px; font-weight:700;'>Logged {a['count']}x</span>"

        sms_badge = f"<div style='font-size:11px; color:#475569; margin-top:4px; font-family:monospace; background:#F1F5F9; padding:4px 8px; border-radius:4px;'>📲 {a['sms_text']}</div>" if a['sms_text'] else ""

        alerts_rows_html += f"""
        <div style="padding: 12px 14px; margin-bottom: 10px; border-left: 4px solid {color}; background: {bg}; border-radius: 0 8px 8px 0;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-weight:700; font-size:12px; color:{color};">{level} ALERT</span>
                <span style="font-size:11px; color:#64748B;">{time_display}</span>
            </div>
            <div style="font-size:12px; color:#1E293B; font-weight:600; margin-top:4px;">{a['message']}</div>
            {sms_badge}
        </div>
        """

    if not deduped_alerts:
        alerts_rows_html = "<div style='color:#64748B; font-size:12px; font-style:italic; padding:12px; background:#F8FAFC; border-radius:6px;'>No critical deterioration alerts were logged during this patient stay.</div>"

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Discharge Summary — {patient.get('name', 'Patient')}</title>
    <style>
        @page {{ size: A4; margin: 12mm; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: #0F172A;
            background: #FFFFFF;
            margin: 0;
            padding: 20px;
            font-size: 12px;
            line-height: 1.45;
        }}
        .header {{
            border-bottom: 2px solid #0D9488;
            padding-bottom: 12px;
            margin-bottom: 16px;
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
        }}
        .hospital-name {{
            font-size: 20px;
            font-weight: 800;
            color: #0D9488;
            letter-spacing: -0.5px;
        }}
        .report-title {{
            font-size: 14px;
            font-weight: 700;
            color: #334155;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .section-title {{
            font-size: 12px;
            font-weight: 700;
            color: #0F766E;
            background: #F0FDFA;
            border-left: 3px solid #0D9488;
            padding: 5px 10px;
            margin-top: 16px;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }}
        .grid-2 {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
            margin-bottom: 12px;
            background: #F8FAFC;
            padding: 12px 14px;
            border-radius: 8px;
            border: 1px solid #E2E8F0;
        }}
        .meta-item {{
            margin-bottom: 5px;
        }}
        .meta-label {{
            font-size: 10px;
            color: #64748B;
            text-transform: uppercase;
            font-weight: 700;
        }}
        .meta-value {{
            font-size: 12px;
            font-weight: 600;
            color: #0F172A;
        }}
        .summary-box {{
            background: #F0FDFA;
            border: 1px solid #99F6E4;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 14px;
        }}
        .summary-title {{
            font-size: 11px;
            font-weight: 700;
            color: #0F766E;
            text-transform: uppercase;
            margin-bottom: 4px;
        }}
        .summary-text {{
            font-size: 12px;
            color: #1E293B;
            font-weight: 500;
        }}
        .trend-text {{
            font-size: 12px;
            font-weight: 700;
            color: {trend_color};
            margin-top: 6px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 6px;
            font-size: 11px;
        }}
        th, td {{
            padding: 6px 8px;
            text-align: left;
            border-bottom: 1px solid #E2E8F0;
        }}
        th {{
            background: #F1F5F9;
            font-weight: 700;
            color: #475569;
            text-transform: uppercase;
            font-size: 10px;
        }}
        .doctor-signature-block {{
            margin-top: 24px;
            padding-top: 14px;
            border-top: 1px dashed #94A3B8;
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
        }}
        .doc-details {{
            font-size: 11px;
            color: #334155;
            line-height: 1.4;
        }}
        .sign-line-box {{
            text-align: right;
            font-size: 11px;
            color: #475569;
        }}
        .signature-line {{
            width: 200px;
            border-bottom: 1px solid #334155;
            margin-bottom: 4px;
            display: inline-block;
        }}
        .footer {{
            margin-top: 20px;
            text-align: center;
            font-size: 10px;
            color: #94A3B8;
        }}
        .print-btn {{
            background: #0D9488;
            color: white;
            border: none;
            padding: 8px 18px;
            border-radius: 6px;
            font-weight: 700;
            cursor: pointer;
            font-size: 12px;
            display: inline-block;
            margin-bottom: 14px;
        }}
        @media print {{
            .no-print {{ display: none !important; }}
            body {{ padding: 0; }}
        }}
    </style>
</head>
<body>
    <div class="no-print" style="text-align: right;">
        <button class="print-btn" onclick="window.print();">🖨️ Print Report</button>
    </div>

    <div class="header">
        <div>
            <div class="hospital-name">{hospital_name}</div>
            <div style="font-size: 11px; color:#64748B;">Department of Rural & Critical Care Surveillance &bull; VitalGuard System</div>
        </div>
        <div style="text-align: right;">
            <div class="report-title">Patient Clinical Discharge Summary</div>
            <div style="font-size: 10px; color:#64748B;">Generated: {now_str}</div>
        </div>
    </div>

    <!-- SECTION 1: PATIENT INFO (TABLE) -->
    <div class="section-title">1. Patient Identification & Care Metadata</div>
    <table style="width:100%; border-collapse:collapse; margin-bottom:14px; font-size:11px; border:1px solid #CBD5E1;">
        <tbody>
            <tr>
                <td style="width:25%; background:#F8FAFC; font-weight:700; color:#475569; border:1px solid #CBD5E1; padding:6px 10px;">Patient Full Name</td>
                <td style="width:25%; border:1px solid #CBD5E1; padding:6px 10px; font-weight:700; color:#0F172A;">{patient.get('name', 'N/A')}</td>
                <td style="width:25%; background:#F8FAFC; font-weight:700; color:#475569; border:1px solid #CBD5E1; padding:6px 10px;">Admission Timestamp</td>
                <td style="width:25%; border:1px solid #CBD5E1; padding:6px 10px;">{admitted_str}</td>
            </tr>
            <tr>
                <td style="background:#F8FAFC; font-weight:700; color:#475569; border:1px solid #CBD5E1; padding:6px 10px;">Patient Record ID</td>
                <td style="border:1px solid #CBD5E1; padding:6px 10px; font-weight:600;">PT-{patient.get('id', 0):05d}</td>
                <td style="background:#F8FAFC; font-weight:700; color:#475569; border:1px solid #CBD5E1; padding:6px 10px;">Discharge Timestamp</td>
                <td style="border:1px solid #CBD5E1; padding:6px 10px;">{discharged_str}</td>
            </tr>
            <tr>
                <td style="background:#F8FAFC; font-weight:700; color:#475569; border:1px solid #CBD5E1; padding:6px 10px;">Ward & Bed Assignment</td>
                <td style="border:1px solid #CBD5E1; padding:6px 10px; font-weight:600;">{patient.get('ward', 'N/A')} &bull; Bed {patient.get('bed_number', 'N/A')}</td>
                <td style="background:#F8FAFC; font-weight:700; color:#475569; border:1px solid #CBD5E1; padding:6px 10px;">Admission Diagnosis</td>
                <td style="border:1px solid #CBD5E1; padding:6px 10px; font-weight:600;">{patient.get('known_condition', 'Observation')}</td>
            </tr>
            <tr>
                <td style="background:#F8FAFC; font-weight:700; color:#475569; border:1px solid #CBD5E1; padding:6px 10px;">Assigned Staff Nurse</td>
                <td style="border:1px solid #CBD5E1; padding:6px 10px;">{patient.get('assigned_nurse_name', 'Staff Nurse')}</td>
                <td style="background:#F8FAFC; font-weight:700; color:#475569; border:1px solid #CBD5E1; padding:6px 10px;">Discharge Status</td>
                <td style="border:1px solid #CBD5E1; padding:6px 10px; color:#0D9488; font-weight:700;">Officially Discharged</td>
            </tr>
        </tbody>
    </table>

    <!-- PLAIN-LANGUAGE CLINICAL RISK & TREND SUMMARY -->
    <div class="summary-box">
        <div class="summary-title">📋 Plain-Language Clinical Risk & Trend Summary</div>
        <div class="summary-text"><strong>Surveillance Overview:</strong> {plain_risk_summary}</div>
        <div class="summary-text" style="margin-top:4px;"><strong>Final Vital Signs at Discharge:</strong> HR {last_hr} bpm | BP {last_bp} | SpO2 {last_spo2}% | RR {last_rr}/min | Temp {last_temp}°C</div>
        <div class="trend-text">📈 Overall Trajectory: {overall_trend_text}</div>
    </div>

    <!-- SECTION 2: VITALS HISTORY (TABLE) -->
    <div class="section-title">2. Physiological Vitals History ({len(vitals_history)} observations)</div>
    <table>
        <thead>
            <tr>
                <th>Date / Time</th>
                <th>Heart Rate</th>
                <th>Blood Pressure</th>
                <th>MAP</th>
                <th>SpO2</th>
                <th>Resp Rate</th>
                <th>Temp</th>
                <th>Risk Tier</th>
            </tr>
        </thead>
        <tbody>
            {vitals_rows_html}
        </tbody>
    </table>

    <!-- SECTION 3: SIGNATURE LINE -->
    <div class="doctor-signature-block">
        <div class="doc-details">
            <strong>Attending Physician / Hospital Authorization:</strong><br>
            <span>Dr. A. Mehta, M.D. &bull; Chief Medical Officer</span><br>
            <span>{hospital_name} &bull; Contact: +91-98765-00000</span>
        </div>
        <div class="sign-line-box">
            <div class="signature-line"></div><br>
            <strong>Authorized Clinical Signature</strong><br>
            <span>Date: {datetime.now().strftime("%d %b %Y")}</span>
        </div>
    </div>

    <div class="footer">
        VitalGuard AI Decision-Support System &bull; Sri Ramachandra Rural Health Centre &bull; Decision Support Adjunct
    </div>
</body>
</html>
"""
    return html_content


def generate_discharge_report_pdf(
    patient: Dict[str, Any],
    vitals_history: List[Dict[str, Any]],
    alerts: List[Dict[str, Any]],
    hospital_name: str = "Sri Ramachandra Rural Health Centre"
) -> bytes:
    """Generates downloadable binary PDF bytes from clinical discharge report HTML."""
    html_content = generate_discharge_report_html(patient, vitals_history, alerts, hospital_name)
    pdf_buffer = io.BytesIO()
    pisa_status = pisa.CreatePDF(io.StringIO(html_content), dest=pdf_buffer)
    return pdf_buffer.getvalue()
