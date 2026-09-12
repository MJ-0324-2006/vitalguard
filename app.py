"""
VitalGuard — AI Early-Warning System for Patient Deterioration.
Main Multi-User Clinical Hospital Application.

Team Orbit | National Level Ideathon 5.0 | CBIT Hyderabad
Theme: Smart Healthcare & Biomedical Innovation
"""

from ui_components import render_mobile_bottom_nav
import base64
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import importlib
import streamlit as st

# Setup system path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ui_components
importlib.reload(ui_components)

from db import (
    init_db,
    has_any_admin,
    admit_patient,
    get_active_patients_by_ward,
    get_all_active_patients,
    get_patient_by_id,
    get_patient_vitals_history,
    add_vitals_reading,
    log_alert,
    auto_resolve_patient_alerts,
    get_alerts_for_ward,
    acknowledge_alert,
    discharge_patient,
    search_patients,
    get_nurses_with_patient_counts,
    get_hospital_wards,
    get_raw_table_data,
    seed_demo_data,
)
from auth import (
    init_auth_session,
    is_authenticated,
    get_current_user,
    logout_user,
    render_login_page,
)
from risk_logic import (
    calculate_patient_risk,
    validate_vitals,
    calculate_map,
    calculate_shock_index,
    calculate_news2,
    PHYSIOLOGICAL_RANGES,
    safe_float,
    _fmt,
)
from explainability import generate_explanation
from discharge_report import generate_discharge_report_html, generate_discharge_report_pdf
from ui_components import (
    inject_custom_css,
    render_top_bar,
    render_mobile_header,
    render_mobile_bottom_nav,
    render_risk_badge,
    render_overview_cards,
    render_simulated_sms_banner,
    render_critical_alert_banner,
    render_explainability_card,
    render_bedside_monitor_display,
    render_compact_alerts_log,
    render_confidence_and_predictions,
    render_empty_state,
    render_disclaimer_footer,
    render_model_validation_card,
    render_design_system_showcase,
    render_split_login_left_panel,
    get_svg_icon,
)

# Page configuration
st.set_page_config(
    page_title="VitalGuard — AI Early-Warning System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ensure database tables exist and prototype demo data is seeded
init_db()

# Inject rose/pink-coral design system styles
inject_custom_css()

# Initialize authentication session state
init_auth_session()

# =============================================================================
# SAFE NUMERIC FORMATTING HELPERS (ELIMINATES VALUEERROR FORMAT CODE 'g')
# =============================================================================

def fmt_num(val: Any, default: str = "--", unit: str = "") -> str:
    """Safely formats any vital value without raising ValueError format code errors."""
    if val is None or val == "" or val == "--":
        return default
    try:
        f = float(val)
        formatted = f"{int(f)}" if f.is_integer() else f"{f:.1f}"
        return f"{formatted}{unit}"
    except (ValueError, TypeError):
        return f"{val}{unit}" if str(val) != default else default


def fmt_bp(sbp: Any, dbp: Any, default: str = "--") -> str:
    """Safely formats blood pressure (e.g. 120/80) without crashing on '--' or strings."""
    if sbp is None or sbp == "" or sbp == "--" or dbp is None or dbp == "" or dbp == "--":
        return default
    try:
        s = float(sbp)
        d = float(dbp)
        s_str = f"{int(s)}" if s.is_integer() else f"{s:.1f}"
        d_str = f"{int(d)}" if d.is_integer() else f"{d:.1f}"
        return f"{s_str}/{d_str}"
    except (ValueError, TypeError):
        return default


# =============================================================================
# SYNCHRONIZED PROGRAMMATIC NAVIGATION (FIXES NON-FUNCTIONAL BUTTONS)
# =============================================================================

init_auth_session()

if "selected_patient_id" not in st.session_state:
    st.session_state.selected_patient_id = None

if "nav_tab" not in st.session_state:
    st.session_state.nav_tab = "Home"

if "discharged_pdf_data" not in st.session_state:
    st.session_state.discharged_pdf_data = None

if "last_simulated_sms" not in st.session_state:
    st.session_state.last_simulated_sms = None

if "open_admit_form" not in st.session_state:
    st.session_state.open_admit_form = False

if "admin_ward_filter" not in st.session_state:
    st.session_state.admin_ward_filter = "All Wards"


def navigate_to(tab: str, patient_id: Optional[int] = None, open_admit: bool = False):
    """Safely transitions between app tabs without Streamlit radio key desync."""
    st.session_state.nav_tab = tab
    if patient_id is not None:
        st.session_state.selected_patient_id = patient_id
    if open_admit:
        st.session_state.open_admit_form = True
    st.rerun()


# =============================================================================
# AUTHENTICATION GATE
# =============================================================================

if not is_authenticated():
    render_login_page()
    st.stop()


# Current authenticated user details
user = get_current_user()
if not user:
    render_login_page()
    st.stop()

is_admin = (user.get("role") == "admin")
hospital_id = user.get("hospital_id", 1)
user_ward = user.get("ward", "Ward 3A")


# =============================================================================
# TOP HEADER & HORIZONTAL NAVIGATION BAR (DESKTOP-FOCUSED WEBSITE HEADER)
# =============================================================================

active_scope_label = user_ward

# Unread alert counter calculation
filter_ward_alerts = None if (is_admin and st.session_state.admin_ward_filter == "All Wards") else (st.session_state.admin_ward_filter if is_admin else user_ward)
ward_alerts_all = get_alerts_for_ward(hospital_id, filter_ward_alerts, limit=50)
unread_count = len([a for a in ward_alerts_all if not a.get("acknowledged")])
alerts_label = f"🚨 Alerts ({unread_count})" if unread_count > 0 else "🚨 Alerts"
now_str = datetime.now().strftime("%a %d %b • %H:%M")

# Unified persistent dark navy top header bar
desktop_nav_container = st.container(key="desktop_top_nav")
with desktop_nav_container:
    header_cols = st.columns([2.6, 0.9, 0.9, 1.0, 0.9, 2.2, 0.9])

    with header_cols[0]:
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:10px; padding:2px 0;">
            <div class="vg-brand-logo">VG</div>
            <div>
                <div class="vg-brand-title">VitalGuard Pro <span class="vg-version-tag">v2.4</span></div>
                <div class="vg-brand-sub">{user['hospital_name']} &bull; {user_ward}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with header_cols[1]:
        if st.button("🏠 Home", use_container_width=True, type="primary" if st.session_state.nav_tab == "Home" else "secondary", key="nav_btn_home"):
            st.session_state.nav_tab = "Home"
            st.rerun()

    with header_cols[2]:
        if st.button("👥 Patients", use_container_width=True, type="primary" if st.session_state.nav_tab == "Patients" else "secondary", key="nav_btn_patients"):
            st.session_state.nav_tab = "Patients"
            st.rerun()

    with header_cols[3]:
        if st.button(alerts_label, use_container_width=True, type="primary" if st.session_state.nav_tab == "Alerts" else "secondary", key="nav_btn_alerts"):
            st.session_state.nav_tab = "Alerts"
            st.rerun()

    with header_cols[4]:
        if st.button("👤 Profile", use_container_width=True, type="primary" if st.session_state.nav_tab == "Profile" else "secondary", key="nav_btn_profile"):
            st.session_state.nav_tab = "Profile"
            st.rerun()

    with header_cols[5]:
        role_badge = "ADMIN" if is_admin else "NURSE"
        st.markdown(f"""
        <div style="display:flex; align-items:center; justify-content:space-between; gap:6px; background:#FFFFFF; border:1px solid #CBD5E1; padding:4px 10px; border-radius:9999px; margin-top:2px;">
            <span style="font-weight:700; font-size:0.75rem; color:#0F172A; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">👤 {user['name']}</span>
            <span style="background:#E11D48; color:#FFFFFF; font-size:0.62rem; font-weight:800; padding:2px 6px; border-radius:4px; white-space:nowrap;">{role_badge}</span>
        </div>
        <div style="font-size:0.68rem; color:#475569; text-align:right; margin-top:2px; font-weight:600;">🕒 {now_str}</div>
        """, unsafe_allow_html=True)

    with header_cols[6]:
        if st.button("🚪 Logout", use_container_width=True, key="nav_btn_logout"):
            logout_user()

st.markdown("<hr style='margin: 0.5rem 0 1.25rem 0; border:none; border-top:1px solid #E2E8F0;'>", unsafe_allow_html=True)

mobile_header_container = st.container(key="mobile_header")
with mobile_header_container:
    render_mobile_header(user)

mobile_alert_count = len(get_alerts_for_ward(hospital_id, None if is_admin else user_ward, limit=50))
nav_click = render_mobile_bottom_nav(active_tab=st.session_state.nav_tab.lower(), alert_count=mobile_alert_count)
if nav_click:
    st.session_state.nav_tab = nav_click.capitalize()
    st.rerun()


# =============================================================================
# VIEW 1: HOME SCREEN (OVERVIEW CARDS & WARD PATIENT CARDS)
# =============================================================================

if st.session_state.nav_tab == "Home":
    # Header with shift context and action buttons
    hdr_c1, hdr_c2 = st.columns([3.5, 1.5])
    with hdr_c1:
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:12px;">
            <h2 style="margin:0; font-weight:800; color:#0F172A; letter-spacing:-0.5px;">Patient Overview</h2>
            <span style="background:#EEF2FF; color:#4F46E5; border:1px solid #C7D2FE; font-size:0.72rem; font-weight:700; padding:3px 10px; border-radius:9999px;">
                ⏱️ Morning Shift &bull; 07:00 - 15:00
            </span>
        </div>
        <p style="margin:2px 0 0 0; font-size:0.85rem; color:#64748B;">Surveillance scope: <strong>{active_scope_label}</strong> at <strong>{user['hospital_name']}</strong>.</p>
        """, unsafe_allow_html=True)
    with hdr_c2:
        btn_a1, btn_a2 = st.columns(2)
        with btn_a1:
            if st.button("➕ Add Patient", type="primary", use_container_width=True, key="home_add_pt_top"):
                navigate_to("Patients", open_admit=True)
        with btn_a2:
            if st.button("🔄 Refresh", type="secondary", use_container_width=True, key="home_refresh_btn"):
                st.rerun()

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Show simulated SMS banner if recently triggered
    if st.session_state.last_simulated_sms:
        sms = st.session_state.last_simulated_sms
        render_simulated_sms_banner(
            doctor_name=sms["doctor_name"],
            phone=sms["phone"],
            patient_name=sms["patient_name"],
            bed_display=sms["bed_display"],
            risk_level=sms["risk_level"]
        )

    # Fetch active patients based on role and ward filter
    if is_admin:
        if st.session_state.admin_ward_filter == "All Wards":
            patients = get_all_active_patients(hospital_id)
        else:
            patients = get_active_patients_by_ward(hospital_id, st.session_state.admin_ward_filter)
    else:
        patients = get_active_patients_by_ward(hospital_id, user_ward)

    # Compute risk using the SINGLE SHARED FUNCTION for each patient
    critical_count = 0
    warning_count = 0
    stable_count = 0

    patient_risk_cards = []
    for p in patients:
        history = get_patient_vitals_history(p["id"])
        curr_v = history[-1] if history else None
        risk = calculate_patient_risk(history, curr_v)

        if risk["level"] == "RED":
            critical_count += 1
        elif risk["level"] == "YELLOW":
            warning_count += 1
        else:
            stable_count += 1

        patient_risk_cards.append((p, history, risk))

    # Soft colored summary stat panels
    render_overview_cards(critical_count, warning_count, stable_count)

    # Focus telemetry monitor on most critical patient
    if patient_risk_cards:
        red_pts = [item for item in patient_risk_cards if item[2]["level"] == "RED"]
        target_pt, target_hist, target_risk = red_pts[0] if red_pts else patient_risk_cards[0]
        latest_pt_vitals = target_hist[-1] if target_hist else None
        st.markdown("---")
        st.markdown(f"#### 📟 Real-Time Bedside Telemetry Monitor &bull; {target_pt['bed_display']} ({target_pt['name']})")
        st.caption("Live ICU telemetry feed showing parameter values, trend deltas, and normal bounds.")
        render_bedside_monitor_display(latest_pt_vitals, target_hist)

    st.markdown("---")

    # Patient Directory Controls & Grid
    d_hdr1, d_hdr2 = st.columns([3, 2])
    with d_hdr1:
        st.markdown(f"#### 👥 Ward Patient Directory ({len(patient_risk_cards)} Active)")
    with d_hdr2:
        search_filter_term = st.text_input("🔍 Quick Search Patient", placeholder="Filter by name, bed...", key="home_quick_search")

    if not patient_risk_cards:
        render_empty_state(
            icon=get_svg_icon("hospital", 32, "#94A3B8"),
            title=f"No Active Patients in {active_scope_label}",
            description="There are currently zero active patients admitted to this scope. Click 'Add Patient' to register a patient and initiate automated early-warning monitoring."
        )
    else:
        # Filter patients if search term provided
        filtered_cards = patient_risk_cards
        if search_filter_term:
            term = search_filter_term.lower()
            filtered_cards = [
                item for item in patient_risk_cards
                if term in item[0]['name'].lower() or term in item[0]['bed_display'].lower() or term in item[0]['known_condition'].lower()
            ]

        # Display Patient Cards (2 per row on desktop)
        cols = st.columns(2)
        for idx, (p, hist, risk) in enumerate(filtered_cards):
            with cols[idx % 2]:
                latest = hist[-1] if hist else {}
                time_str = latest.get("time", "No readings")
                last_time_label = f"Last charted: {time_str}" if hist else "No vitals recorded"

                hr_str = fmt_num(latest.get('heart_rate'))
                bp_str = fmt_bp(latest.get('sbp'), latest.get('dbp'))
                spo2_str = fmt_num(latest.get('spo2'), unit="%")
                rr_str = fmt_num(latest.get('resp_rate'))
                temp_str = fmt_num(latest.get('temperature'), unit="°C")

                strat_mode = "Personalized Baseline" if risk['is_personalized'] else "Population Threshold"
                risk_lvl_class = f"patient-card-{risk['level'].lower()}"

                st.markdown(f"""
                <div class="patient-card {risk_lvl_class}">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
                        <div>
                            <div style="font-weight: 800; font-size: 1.08rem; color: #0F172A;">{p['bed_display']} &bull; {p['name']}</div>
                            <div style="font-size: 0.76rem; color: #64748B;">MRN: <code>PT-{p['id']:05d}</code> &bull; Diagnosis: {p['known_condition']}</div>
                        </div>
                        <div>
                            {render_risk_badge(risk['level'], risk['score'])}
                        </div>
                    </div>
                    <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 10px; margin-top: 10px;">
                        <div class="metric-chip">
                            <span class="metric-chip-label">HR</span>
                            <span class="metric-chip-val">{hr_str}</span>
                        </div>
                        <div class="metric-chip">
                            <span class="metric-chip-label">BP</span>
                            <span class="metric-chip-val">{bp_str}</span>
                        </div>
                        <div class="metric-chip">
                            <span class="metric-chip-label">SpO2</span>
                            <span class="metric-chip-val">{spo2_str}</span>
                        </div>
                        <div class="metric-chip">
                            <span class="metric-chip-label">RR</span>
                            <span class="metric-chip-val">{rr_str}</span>
                        </div>
                        <div class="metric-chip">
                            <span class="metric-chip-label">Temp</span>
                            <span class="metric-chip-val">{temp_str}</span>
                        </div>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.74rem; color: #94A3B8; margin-top: 8px;">
                        <span>{last_time_label} ({len(hist)} observation{'s' if len(hist)!=1 else ''})</span>
                        <span style="color:#E11D48; font-weight:600;">{strat_mode}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                c1, c2 = st.columns(2)
                with c1:
                    if st.button("📊 View Detail & Trend", key=f"btn_detail_{p['id']}", use_container_width=True):
                        navigate_to("Patients", patient_id=p["id"])
                with c2:
                    if st.button("⚡ Chart Vitals", key=f"btn_vitals_{p['id']}", use_container_width=True):
                        navigate_to("Patients", patient_id=p["id"])

    # Recent Alerts in Ward
    st.markdown("---")
    st.markdown(f"#### 🚨 Recent Deterioration Escalations ({active_scope_label})")
    filter_ward_alerts = None if (is_admin and st.session_state.admin_ward_filter == "All Wards") else (st.session_state.admin_ward_filter if is_admin else user_ward)
    recent_alerts = get_alerts_for_ward(hospital_id, filter_ward_alerts, limit=5)
    
    if not recent_alerts:
        render_empty_state(
            icon=get_svg_icon("shield-check", 28, "#10B981"),
            title="No Unresolved Alerts",
            description=f"All patients in {active_scope_label} are currently within stable limits with zero unacknowledged escalations."
        )
    else:
        for a in recent_alerts:
            a_col1, a_col2 = st.columns([5, 1.2])
            with a_col1:
                st.markdown(f"**[{a['risk_level']}] {a['ward']} &bull; Bed {a['bed_number']} &bull; {a['patient_name']}** ({a['timestamp_str']})")
                st.caption(f"{a['message']}")
                if a.get('sms_simulated_text'):
                    st.caption(f"📱 {a['sms_simulated_text']}")
            with a_col2:
                if not a["acknowledged"]:
                    if st.button("Acknowledge", key=f"ack_home_{a['id']}", use_container_width=True):
                        acknowledge_alert(a["id"])
                        st.rerun()
                else:
                    st.caption("✅ Acknowledged")


# =============================================================================
# VIEW 2: PATIENTS (SEARCH, ADMISSION & DETAIL VIEW)
# =============================================================================

elif st.session_state.nav_tab == "Patients":
    # -------------------------------------------------------------------------
    # CASE A: A SPECIFIC PATIENT IS SELECTED (DETAIL VIEW)
    # -------------------------------------------------------------------------
    if st.session_state.selected_patient_id:
        patient = get_patient_by_id(st.session_state.selected_patient_id)
        if not patient or not patient.get("is_active"):
            st.session_state.selected_patient_id = None
            st.rerun()

        # BACK BUTTON
        if st.button("⬅️ Back to Patient Directory", key="btn_back_roster"):
            st.session_state.selected_patient_id = None
            st.rerun()

        history = get_patient_vitals_history(patient["id"])
        curr_v = history[-1] if history else None

        # Call SINGLE SOURCE OF TRUTH risk function
        risk_data = calculate_patient_risk(history, curr_v)
        explanation = generate_explanation(risk_data["level"], history, derived_metrics=risk_data)

        # 1. Critical Alert Banner if RED
        if risk_data["level"] == "RED":
            render_critical_alert_banner(
                patient_name=patient["name"],
                bed_display=patient["bed_display"],
                key_reasons=explanation.get("key_findings", ["Severe vital sign deterioration"])
            )

        # 2. Patient Header Banner
        h1, h2 = st.columns([3, 2])
        with h1:
            st.markdown(f"### {patient['bed_display']} &bull; {patient['name']}")
            st.markdown(f"**Diagnosis:** {patient['known_condition']} | **Admitted:** {patient['admitted_str']} | **Record:** `PT-{patient['id']:05d}`")
            if patient.get("assigned_nurse_name"):
                st.caption(f"Primary Nurse: **{patient['assigned_nurse_name']}**")
        with h2:
            st.markdown(f"<div style='text-align:right;'>{render_risk_badge(risk_data['level'], risk_data['score'])}</div>", unsafe_allow_html=True)
            strat_label = "Personalized Baseline Activated" if risk_data['is_personalized'] else f"Population Baseline ({len(history)}/3 charted readings)"
            st.markdown(f"<div style='text-align:right; font-size:0.80rem; color:#64748B;'>Monitoring Mode: <strong>{strat_label}</strong></div>", unsafe_allow_html=True)

        st.markdown("---")

        # 3. Current Vitals Metric Cards (GUARANTEED NO FORMAT CODE 'g' CRASHES)
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Heart Rate", fmt_num(curr_v.get('heart_rate') if curr_v else None, default="--", unit=" bpm"))
        c2.metric("Blood Pressure", fmt_bp(curr_v.get('sbp'), curr_v.get('dbp')) if curr_v else "--")
        c3.metric("SpO2", fmt_num(curr_v.get('spo2') if curr_v else None, default="--", unit="%"))
        c4.metric("Resp Rate", fmt_num(curr_v.get('resp_rate') if curr_v else None, default="--", unit="/min"))
        c5.metric("MAP", fmt_num(risk_data.get('map_value'), default="--", unit=" mmHg"))
        c6.metric("Shock Index", fmt_num(risk_data.get('shock_index'), default="--"))

        # 4. Clinical Explainability Card
        render_explainability_card(explanation, risk_data["level"])

        # 5. ICU Bedside Vitals Monitor Display
        st.markdown("#### 📟 Real-Time Bedside Telemetry Monitor")
        st.caption("High-contrast ICU telemetry display pulling live observations from SQLite with parameter deltas.")
        render_bedside_monitor_display(curr_v, history)

        # 6. Neural Multi-Task Outcomes & Uncertainty
        st.markdown("---")
        render_confidence_and_predictions(risk_data)

        # 7. Bedside Vitals Entry Form WITH VISIBLE SUBMIT BUTTON
        st.markdown("---")
        st.markdown(f"#### 🩺 Chart Bedside Vitals for **{patient['name']}** ({patient['bed_display']})")
        st.caption("Chart current observations. On submission, the risk score, card color, and last updated time update automatically everywhere.")

        default_hr = float(curr_v.get("heart_rate", 75.0)) if curr_v else 75.0
        default_sbp = float(curr_v.get("sbp", 120.0)) if curr_v else 120.0
        default_dbp = float(curr_v.get("dbp", 80.0)) if curr_v else 80.0
        default_spo2 = float(curr_v.get("spo2", 98.0)) if curr_v else 98.0
        default_rr = float(curr_v.get("resp_rate", 16.0)) if curr_v else 16.0
        default_temp = float(curr_v.get("temperature", 37.0)) if curr_v else 37.0
        default_gcs = int(curr_v.get("gcs", 15)) if curr_v else 15

        with st.form("enter_vitals_form"):
            st.markdown(f"<div style='background:#F1F5F9; border-radius:8px; padding:10px 14px; margin-bottom:14px; font-size:0.92rem; color:#1E293B;'>👤 <strong>Patient:</strong> {patient['name']} &nbsp;|&nbsp; 🏥 <strong>Location:</strong> {patient['bed_display']} &nbsp;|&nbsp; 📑 <strong>Record ID:</strong> <code>PT-{patient['id']:05d}</code></div>", unsafe_allow_html=True)
            f1, f2 = st.columns(2)
            with f1:
                hr_in = st.number_input("Heart Rate (bpm)", min_value=0.0, max_value=300.0, value=default_hr, step=1.0)
                sbp_in = st.number_input("Systolic BP (mmHg)", min_value=0.0, max_value=300.0, value=default_sbp, step=1.0)
                dbp_in = st.number_input("Diastolic BP (mmHg)", min_value=0.0, max_value=200.0, value=default_dbp, step=1.0)
            with f2:
                spo2_in = st.number_input("Oxygen Saturation SpO2 (%)", min_value=0.0, max_value=100.0, value=default_spo2, step=0.5)
                rr_in = st.number_input("Respiratory Rate (/min)", min_value=0.0, max_value=80.0, value=default_rr, step=1.0)
                temp_in = st.number_input("Body Temperature (°C)", min_value=25.0, max_value=45.0, value=default_temp, step=0.1)

            gcs_in = st.number_input("Glasgow Coma Scale (GCS 3-15)", min_value=3, max_value=15, value=default_gcs, step=1)

            # CLEARLY LABELED AND VISIBLE SUBMIT BUTTON
            submit_vitals = st.form_submit_button("✅ Submit Vitals", use_container_width=True)

        if submit_vitals:
            v_dict = {
                "heart_rate": hr_in,
                "sbp": sbp_in,
                "dbp": dbp_in,
                "spo2": spo2_in,
                "resp_rate": rr_in,
                "temperature": temp_in,
                "gcs": float(gcs_in),
            }
            is_valid, issues, clean_v = validate_vitals(v_dict)
            if not is_valid:
                st.error("⚠️ Physiological Validation Notice:")
                for iss in issues:
                    st.warning(f"&bull; {iss}")
            else:
                clean_v["map"] = calculate_map(clean_v["sbp"], clean_v["dbp"])
                
                # Re-calculate risk using SINGLE SOURCE OF TRUTH function
                updated_history = history + [clean_v]
                new_risk = calculate_patient_risk(updated_history, clean_v)

                # Persist to database
                add_vitals_reading(
                    patient_id=patient["id"],
                    vitals_dict=clean_v,
                    risk_level=new_risk["level"],
                    risk_score=new_risk["score"]
                )

                # If GREEN, auto-resolve previous deterioration alerts for this patient
                if new_risk["level"] == "GREEN":
                    auto_resolve_patient_alerts(patient["id"])
                elif new_risk["level"] == "RED":
                    doc_name = "Dr. A. Mehta (Chief Physician)"
                    doc_phone = "+91-98765-00000"
                    log_alert(
                        patient_id=patient["id"],
                        risk_level="RED",
                        message="; ".join(new_risk["driving_factors"]),
                        doctor_name=doc_name,
                        doctor_phone=doc_phone,
                        patient_name=patient["name"],
                        bed_str=patient["bed_display"]
                    )
                    st.session_state.last_simulated_sms = {
                        "doctor_name": doc_name,
                        "phone": doc_phone,
                        "patient_name": patient["name"],
                        "bed_display": patient["bed_display"],
                        "risk_level": "RED"
                    }

                st.success(f"Vitals charted successfully! Re-scored: **{new_risk['level']}** ({new_risk['score']:.2f})")
                st.rerun()

        # 8. DISCHARGE PATIENT FLOW (Soft-delete & Printable Report)
        st.markdown("---")
        st.markdown("#### 📋 Patient Discharge & Clinical Audit Protocol")
        st.caption("Discharging a patient generates a printable clinical report and marks the record inactive. Audit records remain in SQLite.")

        d_col1, d_col2 = st.columns([1.5, 2])
        with d_col1:
            if st.button("🚪 Discharge Patient & Generate Report", use_container_width=True, key="discharge_patient_btn"):
                pdf_bytes = generate_discharge_report_pdf(
                    patient=patient,
                    vitals_history=history,
                    alerts=get_alerts_for_ward(hospital_id, ward=None, limit=100),
                    hospital_name=user["hospital_name"]
                )
                discharge_patient(patient["id"])
                st.session_state.discharged_pdf_data = {
                    "bytes": pdf_bytes,
                    "filename": f"discharge_report_PT-{patient['id']:05d}_{patient['name'].replace(' ', '_')}.pdf",
                    "patient_name": patient["name"],
                    "bed_display": patient["bed_display"]
                }
                st.session_state.selected_patient_id = None
                st.rerun()

    # -------------------------------------------------------------------------
    # CASE B: DIRECTORY & SEARCH VIEW
    # -------------------------------------------------------------------------
    else:
        st.subheader("👥 Ward Patient Directory & Search")
        st.markdown(f"Surveillance directory for **{active_scope_label}**.")

        # Show Discharge Report Download Banner if a patient was just discharged
        if st.session_state.get("discharged_pdf_data"):
            d_pdf = st.session_state.discharged_pdf_data
            b64_pdf = base64.b64encode(d_pdf["bytes"]).decode("utf-8")
            d_fname = d_pdf["filename"]
            d_pname = d_pdf["patient_name"]
            d_bed = d_pdf["bed_display"]
            
            st.success(f"✅ **Patient {d_pname} ({d_bed}) officially discharged!** Audit record preserved in SQLite database.")
            b1, b2, b3 = st.columns([2.5, 2.5, 1])
            with b1:
                st.download_button(
                    label=f"📄 Download PDF Report ({d_pname})",
                    data=d_pdf["bytes"],
                    file_name=d_fname,
                    mime="application/pdf",
                    use_container_width=True,
                    key="download_discharged_pdf_btn"
                )
            with b2:
                dl_link = f'<a href="data:application/pdf;base64,{b64_pdf}" download="{d_fname}" style="display:inline-block; width:100%; text-align:center; background-color:#E11D48; color:#FFFFFF; font-weight:700; padding:8px 12px; border-radius:8px; text-decoration:none; box-shadow:0 2px 6px rgba(225,29,72,0.3);">📥 Direct Save PDF File</a>'
                st.markdown(dl_link, unsafe_allow_html=True)
            with b3:
                if st.button("Dismiss", key="dismiss_discharge_pdf_btn", use_container_width=True):
                    st.session_state.discharged_pdf_data = None
                    st.rerun()
            st.markdown("---")

        # Handle clear search request before widget instantiation
        if st.session_state.get("clear_search_requested", False):
            st.session_state.patient_search_input = ""
            st.session_state.clear_search_requested = False

        # Search Bar & Controls
        s_col1, s_col2, s_col3 = st.columns([3, 1.5, 1])
        with s_col1:
            search_query = st.text_input(
                "🔍 Search by Name, Ward, or Bed Number",
                placeholder="e.g. 'Rajesh', 'Ward 3A', 'Bed 04'",
                key="patient_search_input"
            )
        with s_col2:
            if is_admin:
                hospital_wards = get_hospital_wards(hospital_id)
                ward_options = ["All Wards"] + hospital_wards
                dir_ward_filter = st.selectbox(
                    "Filter Ward",
                    options=ward_options,
                    index=ward_options.index(st.session_state.admin_ward_filter) if st.session_state.admin_ward_filter in ward_options else 0,
                    key="patient_dir_ward_filter"
                )
                if dir_ward_filter != st.session_state.admin_ward_filter:
                    st.session_state.admin_ward_filter = dir_ward_filter
                    st.rerun()
            else:
                st.text_input("Ward Scope", value=user_ward, disabled=True)
                dir_ward_filter = user_ward
        with s_col3:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("Clear Search", use_container_width=True):
                st.session_state.clear_search_requested = True
                st.rerun()


        # Query Database
        active_ward_arg = None if (is_admin and dir_ward_filter == "All Wards") else (dir_ward_filter if is_admin else user_ward)
        if search_query:
            patients = search_patients(hospital_id, search_query, ward=active_ward_arg)
        else:
            if active_ward_arg is None:
                patients = get_all_active_patients(hospital_id)
            else:
                patients = get_active_patients_by_ward(hospital_id, active_ward_arg)

        # Admission Form Expander
        open_form_flag = st.session_state.open_admit_form
        with st.expander("➕ Admit New Patient to Ward", expanded=open_form_flag):
            with st.form("admit_patient_form"):
                st.markdown("##### 📝 New Patient Registration")
                p_name = st.text_input("Patient Full Name", placeholder="e.g. Rajesh Kumar")
                
                a1, a2 = st.columns(2)
                with a1:
                    default_ward_val = user_ward if not is_admin else (st.session_state.admin_ward_filter if st.session_state.admin_ward_filter != "All Wards" else "Ward 3A")
                    p_ward = st.text_input("Ward Assignment", value=default_ward_val, disabled=not is_admin)
                with a2:
                    p_bed = st.text_input("Bed Number", placeholder="e.g. Bed 04")

                p_diag = st.text_input("Admission Diagnosis / Condition", placeholder="e.g. Post-op abdominal surgery, Sepsis protocol")

                admit_btn = st.form_submit_button("Complete Patient Admission", use_container_width=True)

            if admit_btn:
                if not p_name or not p_bed:
                    st.error("Please provide both patient full name and bed number.")
                else:
                    try:
                        pid = admit_patient(
                            name=p_name,
                            hospital_id=hospital_id,
                            ward=p_ward,
                            bed_number=p_bed,
                            assigned_nurse_id=user["id"] if user["role"] == "nurse" else None,
                            known_condition=p_diag
                        )
                        st.session_state.open_admit_form = False
                        st.success(f"Patient '{p_name}' successfully admitted to **{p_ward} &bull; {p_bed}**!")
                        st.rerun()
                    except ValueError as ve:
                        st.error(str(ve))
                    except Exception as ex:
                        st.error(f"Admission failed: {ex}")

        # Render Patient Table / Cards
        if not patients:
            render_empty_state(
                icon=get_svg_icon("search", 32, "#94A3B8"),
                title="No Matching Patients Found",
                description=f"No active patients found matching '{search_query}' in {dir_ward_filter}." if search_query else f"No active patients in {dir_ward_filter}. Click '+ Admit New Patient to Ward' to begin."
            )
        else:
            st.markdown(f"**Showing {len(patients)} active patient(s) in {dir_ward_filter}:**")
            for p in patients:
                history = get_patient_vitals_history(p["id"])
                curr_v = history[-1] if history else None
                risk = calculate_patient_risk(history, curr_v)

                col_card, col_btn = st.columns([5, 1.2])
                with col_card:
                    st.markdown(f"""
                    <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:12px; padding:12px 16px; margin-bottom:8px; display:flex; justify-content:space-between; align-items:center; box-shadow:0 1px 3px rgba(0,0,0,0.04);">
                        <div>
                            <div style="font-weight:700; font-size:1.0rem; color:#0F172A;">{p['bed_display']} &bull; {p['name']}</div>
                            <div style="font-size:0.75rem; color:#64748B;">Diagnosis: {p['known_condition']} &bull; Admitted: {p['admitted_str']}</div>
                        </div>
                        <div>
                            {render_risk_badge(risk['level'], risk['score'])}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_btn:
                    st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
                    if st.button("Open Record", key=f"open_pt_{p['id']}", use_container_width=True):
                        st.session_state.selected_patient_id = p["id"]
                        st.rerun()


# =============================================================================
# VIEW 3: ALERTS FEED & ESCALATION MANAGEMENT
# =============================================================================

elif st.session_state.nav_tab == "Alerts":
    st.subheader("🚨 Clinical Deterioration Alerts Feed")
    st.markdown(f"Surveillance alert log and direct escalation protocol management for **{user_ward}**.")

    filter_ward_alerts = None if (is_admin and st.session_state.admin_ward_filter == "All Wards") else (st.session_state.admin_ward_filter if is_admin else user_ward)
    alerts = get_alerts_for_ward(hospital_id, filter_ward_alerts, limit=100)

    # 1. Summary Stat Cards Header
    total_alerts = len(alerts)
    crit_alerts = len([a for a in alerts if a.get("risk_level") == "RED"])
    warn_alerts = len([a for a in alerts if a.get("risk_level") == "YELLOW"])
    unack_alerts = len([a for a in alerts if not a.get("acknowledged")])

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Shift Alerts", total_alerts)
    s2.metric("Critical Red Escalations", crit_alerts, delta="Immediate Doctor SMS" if crit_alerts > 0 else "Normal", delta_color="inverse" if crit_alerts > 0 else "off")
    s3.metric("Yellow Warnings", warn_alerts)
    s4.metric("Unacknowledged", unack_alerts, delta="Requires Action" if unack_alerts > 0 else "All Clear", delta_color="inverse" if unack_alerts > 0 else "off")

    st.markdown("---")

    # 2. Controls: Filter Tabs, Search, Acknowledge All & Export Report
    c_col1, c_col2, c_col3, c_col4 = st.columns([2.5, 2.0, 1.2, 1.3])
    with c_col1:
        alert_filter_tab = st.radio(
            "Filter Severity",
            options=["All Alerts", "Critical Only", "Warnings Only"],
            horizontal=True,
            key="alert_feed_filter_tab"
        )
    with c_col2:
        alert_search = st.text_input("🔍 Search Alerts", placeholder="Search by patient, bed...", key="alert_search_input")
    with c_col3:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("✅ Acknowledge All", type="primary", use_container_width=True, key="ack_all_btn"):
            unack_list = [a for a in alerts if not a.get("acknowledged")]
            for u_a in unack_list:
                acknowledge_alert(u_a["id"])
            st.success(f"Acknowledged {len(unack_list)} alert(s)!")
            st.rerun()
    with c_col4:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        shift_report_text = f"VITALGUARD CLINICAL SHIFT REPORT\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\nScope: {user_ward} ({user['hospital_name']})\nTotal Alerts: {total_alerts} (Critical: {crit_alerts}, Warnings: {warn_alerts}, Unacknowledged: {unack_alerts})\n\nALERT LOG:\n"
        for a in alerts:
            shift_report_text += f"- [{a.get('timestamp_str')}] {a.get('risk_level')} Risk | {a.get('patient_name')} ({a.get('ward')}, Bed {a.get('bed_number')}) - {a.get('message')}\n"
        
        st.download_button(
            label="📄 Export Shift Report",
            data=shift_report_text,
            file_name=f"shift_report_{user_ward.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True,
            key="export_shift_report_btn"
        )

    # 3. Filter Alert Cards
    filtered_alerts = alerts
    if alert_filter_tab == "Critical Only":
        filtered_alerts = [a for a in filtered_alerts if a.get("risk_level") == "RED"]
    elif alert_filter_tab == "Warnings Only":
        filtered_alerts = [a for a in filtered_alerts if a.get("risk_level") == "YELLOW"]

    if alert_search:
        s_term = alert_search.lower()
        filtered_alerts = [
            a for a in filtered_alerts
            if s_term in a.get("patient_name", "").lower() or s_term in a.get("bed_number", "").lower() or s_term in a.get("message", "").lower()
        ]

    # 4. Render Alert Feed Cards
    if not filtered_alerts:
        render_empty_state(
            icon=get_svg_icon("shield-check", 32, "#10B981"),
            title="No Escalation Alerts Found",
            description="There are currently zero active deterioration alerts matching your active filters."
        )
    else:
        for a in filtered_alerts:
            is_unack = not a.get("acknowledged")
            new_badge_html = '<span class="alert-badge-new">NEW</span> ' if is_unack else ''
            border_color = "#DC2626" if a.get("risk_level") == "RED" else "#F59E0B"
            bg_color = "#FEF2F2" if a.get("risk_level") == "RED" else "#FFFBEB"
            
            p_id = a.get("patient_id")

            st.markdown(f"""
            <div class="alert-feed-card" style="border-left: 5px solid {border_color}; background:{bg_color if is_unack else '#FFFFFF'};">
                <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:6px;">
                    <div>
                        <div style="font-weight:800; font-size:1.02rem; color:#0F172A;">
                            {new_badge_html}{a.get('patient_name')} &bull; {a.get('ward')} {a.get('bed_number')}
                        </div>
                        <div style="font-size:0.75rem; color:#64748B;">MRN: <code>PT-{p_id:05d}</code> &bull; Timestamp: <strong>{a.get('timestamp_str')}</strong></div>
                    </div>
                    <div>
                        {render_risk_badge(a.get('risk_level'), size="sm")}
                    </div>
                </div>
                <div style="font-size:0.86rem; color:#334155; margin-bottom:8px; line-height:1.4;">
                    <strong>Clinical Finding:</strong> {a.get('message')}
                </div>
                {f'<div style="font-size:0.75rem; color:#E11D48; font-weight:600; font-family:\'JetBrains Mono\'; mb-2">📱 {a.get("sms_simulated_text")}</div>' if a.get("sms_simulated_text") else ''}
            </div>
            """, unsafe_allow_html=True)

            a_c1, a_c2 = st.columns([1, 1])
            with a_c1:
                if st.button("📊 View Patient Detail", key=f"feed_view_pt_{a['id']}", use_container_width=True):
                    navigate_to("Patients", patient_id=p_id)
            with a_c2:
                if is_unack:
                    if st.button("✅ Acknowledge Escalation", key=f"feed_ack_{a['id']}", use_container_width=True, type="primary"):
                        acknowledge_alert(a["id"])
                        st.rerun()
                else:
                    st.caption("✅ Acknowledged")


# =============================================================================
# VIEW 4: USER PROFILE & ARCHITECTURE DOCS
# =============================================================================

elif st.session_state.nav_tab == "Profile":
    st.subheader("👤 User Profile & Clinical AI Specifications")

    p1, p2 = st.columns(2)
    with p1:
        st.markdown("##### Account Details")
        st.markdown(f"""
        - **Staff Member**: {user['name']}
        - **System Role**: {user['role'].upper()}
        - **Affiliation**: {user['hospital_name']}
        - **Assigned Ward**: {user['ward']}
        - **Contact**: {user.get('phone', 'N/A')}
        """)

        if st.button("Log Out of Session", type="primary"):
            logout_user()

    with p2:
        st.markdown("##### System & Surveillance Settings")
        st.markdown(f"""
        - **Hospital Center**: {user['hospital_name']}
        - **Active Ward Scope**: {user['ward']}
        - **Clinical Surveillance**: Active 24/7 AI Risk Stratification
        - **Alert Escalation Protocol**: Direct Doctor SMS Notification
        """)

render_disclaimer_footer()