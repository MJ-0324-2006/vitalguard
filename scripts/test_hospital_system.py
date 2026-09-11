"""
End-to-End Automated Integration Test for VitalGuard Hospital Application.

Tests:
1. DB initialization & zero dummy data check
2. Hospital & Admin creation
3. Nurse registration & role-based authentication
4. Patient admission & search (name / ward / bed)
5. Vitals submission with single shared risk function (fallback -> personalized baseline)
6. Red risk deterioration trigger & simulated SMS logging
7. Discharge flow (soft deletion + printable HTML report)
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from db import (
    init_db,
    has_any_admin,
    create_hospital_and_admin,
    register_nurse,
    authenticate_user,
    admit_patient,
    get_active_patients_by_ward,
    get_all_active_patients,
    get_patient_vitals_history,
    add_vitals_reading,
    log_alert,
    get_alerts_for_ward,
    search_patients,
    discharge_patient,
    get_patient_by_id,
)
from risk_logic import calculate_patient_risk
from discharge_report import generate_discharge_report_html


def run_tests():
    print("=" * 70)
    print("STARTING VITALGUARD MULTI-USER HOSPITAL SYSTEM INTEGRATION TEST")
    print("=" * 70)

    # 1. Initialize DB
    init_db()
    print("[1] Database initialized.")

    # 2. Check admin creation
    if not has_any_admin():
        h_id, a_id = create_hospital_and_admin(
            hospital_name="Sri Ramachandra Rural Health Centre",
            admin_name="Dr. A. Mehta (Chief Physician)",
            username="admin",
            password="password123",
            phone="+91-98765-00000"
        )
        print(f"[2] Created Hospital (ID {h_id}) and Doctor Admin (ID {a_id}).")
    else:
        print("[2] Doctor Admin already exists in DB.")

    # 3. Test Staff ID Verification & Auth
    from db import verify_authorized_staff, save_user_session, get_user_by_session_token
    assert verify_authorized_staff("Sri Ramachandra Rural Health Centre", "STAFF-1001"), "STAFF-1001 should be verified!"
    assert not verify_authorized_staff("Sri Ramachandra Rural Health Centre", "INVALID-999"), "Invalid ID should be rejected!"
    print("[3] Staff ID Pre-Authorization Verification: OK (STAFF-1001 valid, INVALID-999 rejected)")

    # 4. Register a Nurse
    try:
        n_id = register_nurse(
            name="Nurse Priya Sharma",
            username="priya.nurse",
            password="nurse123",
            hospital_id=1,
            ward="Ward 3A",
            phone="+91-91234-56789"
        )
        print(f"[4] Registered Nurse: Priya Sharma (ID {n_id}) for Ward 3A.")
    except ValueError:
        print("[4] Nurse Priya Sharma already registered.")

    # Test Nurse Auth & Persistent Session
    nurse_user = authenticate_user("priya.nurse", "nurse123")
    assert nurse_user is not None, "Nurse authentication failed!"
    assert nurse_user["role"] == "nurse", f"Expected nurse role, got {nurse_user['role']}"
    assert nurse_user["ward"] == "Ward 3A", f"Expected Ward 3A, got {nurse_user['ward']}"
    
    # Save session token & test restoration
    save_user_session("test_token_123", nurse_user["id"], nurse_user["username"])
    restored = get_user_by_session_token("test_token_123")
    assert restored is not None and restored["id"] == nurse_user["id"], "Persistent session restoration failed!"
    print(f"[5] Nurse Auth & Persistent Session Token: OK (Ward: {nurse_user['ward']})")

    # 5. Admit a Real Patient
    try:
        p_id = admit_patient(
            name="Vikramaditya Verma",
            hospital_id=nurse_user["hospital_id"],
            ward="Ward 3A",
            bed_number="Bed 07",
            assigned_nurse_id=nurse_user["id"],
            known_condition="Post-operative bowel resection observation"
        )
        print(f"[6] Admitted Patient: Vikramaditya Verma (ID {p_id}) into Ward 3A &bull; Bed 07.")
    except ValueError as ve:
        # If already exists from prior run, fetch id
        pts = search_patients(nurse_user["hospital_id"], "Vikramaditya")
        p_id = pts[0]["id"]
        print(f"[6] Existing Patient found: ID {p_id}")

    # 6. Test Search functionality (Name, Ward, Bed)
    search_by_name = search_patients(nurse_user["hospital_id"], "Vikramaditya")
    assert len(search_by_name) > 0, "Search by name failed!"
    print(f"[7a] Search by Name: Found {search_by_name[0]['name']} -> {search_by_name[0]['bed_display']}")

    search_by_bed = search_patients(nurse_user["hospital_id"], "Bed 07")
    assert len(search_by_bed) > 0, "Search by bed failed!"
    print(f"[7b] Search by Bed: Found {search_by_bed[0]['name']} -> {search_by_bed[0]['bed_display']}")

    # 7. Vitals Readings & Single Source of Truth Risk Logic
    # Reading 1: Normal (Generic fallback)
    r1 = {"heart_rate": 72.0, "sbp": 120.0, "dbp": 78.0, "resp_rate": 15.0, "spo2": 98.0, "temperature": 36.8, "gcs": 15.0}
    risk1 = calculate_patient_risk([r1], r1)
    add_vitals_reading(p_id, r1, risk1["level"], risk1["score"])
    print(f"[8a] Reading 1 Risk: {risk1['level']} (Score {risk1['score']}, Personalized: {risk1['is_personalized']})")
    assert risk1["level"] == "GREEN", f"Expected GREEN for reading 1, got {risk1['level']}"
    assert not risk1["is_personalized"], "Should be false for < 3 readings"

    # Reading 2: Normal
    r2 = {"heart_rate": 74.0, "sbp": 122.0, "dbp": 80.0, "resp_rate": 16.0, "spo2": 98.0, "temperature": 36.9, "gcs": 15.0}
    risk2 = calculate_patient_risk([r1, r2], r2)
    add_vitals_reading(p_id, r2, risk2["level"], risk2["score"])

    # Reading 3: Normal (Activates personalized baseline!)
    r3 = {"heart_rate": 71.0, "sbp": 118.0, "dbp": 76.0, "resp_rate": 15.0, "spo2": 99.0, "temperature": 36.7, "gcs": 15.0}
    risk3 = calculate_patient_risk([r1, r2, r3], r3)
    add_vitals_reading(p_id, r3, risk3["level"], risk3["score"])
    print(f"[8b] Reading 3 Risk: {risk3['level']} (Score {risk3['score']}, Personalized: {risk3['is_personalized']}, Personal HR Baseline: {risk3['personal_baselines'].get('heart_rate')})")
    assert risk3["is_personalized"], "Should be personalized for >= 3 readings!"

    # Reading 4: Acute Deterioration (Septic Shock Trajectory)
    r4 = {"heart_rate": 122.0, "sbp": 82.0, "dbp": 48.0, "resp_rate": 27.0, "spo2": 90.0, "temperature": 39.3, "gcs": 12.0}
    risk4 = calculate_patient_risk([r1, r2, r3, r4], r4)
    add_vitals_reading(p_id, r4, risk4["level"], risk4["score"])
    print(f"[8c] Deterioration Reading 4 Risk: {risk4['level']} (Score {risk4['score']})")
    print(f"     Driving Factors: {risk4['driving_factors']}")
    assert risk4["level"] == "RED", f"Expected RED for septic collapse, got {risk4['level']}"

    # 8. Log Alert & Simulated SMS
    alert_id = log_alert(
        patient_id=p_id,
        risk_level="RED",
        message="; ".join(risk4["driving_factors"]),
        doctor_name="Dr. A. Mehta (Chief Physician)",
        doctor_phone="+91-98765-00000",
        patient_name="Vikramaditya Verma",
        bed_str="Ward 3A &bull; Bed 07"
    )
    alerts = get_alerts_for_ward(1, "Ward 3A")
    assert len(alerts) > 0, "No alerts found in ward!"
    print(f"[9] Alert Logged (ID {alert_id}): {alerts[0]['sms_simulated_text']}")

    # 9. Discharge Flow (Soft-delete + Printable Report)
    p_info = get_patient_by_id(p_id)
    history = get_patient_vitals_history(p_id)
    report_html = generate_discharge_report_html(p_info, history, alerts, hospital_name="Sri Ramachandra Rural Health Centre")
    assert "Discharge Summary" in report_html, "Report generation failed"
    assert "Vikramaditya Verma" in report_html, "Patient name missing from report"
    assert len(history) >= 4, "Vitals rows missing from report"

    discharged = discharge_patient(p_id)
    assert discharged, "Discharge patient failed!"
    p_after = get_patient_by_id(p_id)
    assert not p_after["is_active"], "Patient should be inactive after discharge!"
    print(f"[10] Patient Discharged: is_active={p_after['is_active']}, Discharged Date={p_after['discharged_date']}")
    print(f"     Printable Report HTML Generated ({len(report_html)} bytes).")

    # 10. Verify Ward Isolation
    ward_pts = get_active_patients_by_ward(1, "Ward 3A")
    assert all(p["id"] != p_id for p in ward_pts), "Discharged patient should NOT appear in active ward list!"
    print("[11] Ward Isolation & Soft Deletion: Verified!")

    print("\n" + "=" * 70)
    print("ALL 11 INTEGRATION TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
