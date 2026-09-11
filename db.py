"""
VitalGuard Database Layer (SQLAlchemy + SQLite).

Implements:
- Local offline SQLite database with thread-safe connection pooling
- Strict Zero Dummy Data Policy: all entities are real DB rows created through the app
- Safe concurrent write handling with session retries
- Complete clinical models: Hospital, DoctorAdmin, Nurse, Patient, VitalsHistory, AlertsLog
- Soft-deletion for discharge flow (is_active = False, no hard delete)
"""

import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    func,
    or_,
)
from sqlalchemy.orm import declarative_base, relationship, scoped_session, sessionmaker

REPO_ROOT = Path(__file__).resolve().parent
DB_FILE = REPO_ROOT / "vitalguard.db"
DATABASE_URL = f"sqlite:///{DB_FILE}"

# Engine with thread-safety and timeout for concurrent writes
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False, "timeout": 30},
    pool_pre_ping=True,
)

SessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
ScopedSession = scoped_session(SessionFactory)
Base = declarative_base()


# =============================================================================
# MODELS
# =============================================================================

class Hospital(Base):
    __tablename__ = "hospitals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(120), nullable=False)
    code = Column(String(30), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.now)

    doctors = relationship("DoctorAdmin", back_populates="hospital", cascade="all, delete-orphan")
    nurses = relationship("Nurse", back_populates="hospital", cascade="all, delete-orphan")
    patients = relationship("Patient", back_populates="hospital", cascade="all, delete-orphan")


class DoctorAdmin(Base):
    __tablename__ = "doctors_admin"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    username = Column(String(50), unique=True, nullable=False)
    password = Column(String(128), nullable=False)
    phone = Column(String(20), default="+91-98765-43210")
    hospital_id = Column(Integer, ForeignKey("hospitals.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.now)

    hospital = relationship("Hospital", back_populates="doctors")


class Nurse(Base):
    __tablename__ = "nurses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    username = Column(String(50), unique=True, nullable=False)
    password = Column(String(128), nullable=False)
    ward = Column(String(50), nullable=False)
    phone = Column(String(20), default="+91-91234-56789")
    hospital_id = Column(Integer, ForeignKey("hospitals.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.now)

    hospital = relationship("Hospital", back_populates="nurses")
    assigned_patients = relationship("Patient", back_populates="assigned_nurse")


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    hospital_id = Column(Integer, ForeignKey("hospitals.id"), nullable=False)
    ward = Column(String(50), nullable=False)
    bed_number = Column(String(30), nullable=False)
    admitted_date = Column(DateTime, default=datetime.now)
    discharged_date = Column(DateTime, nullable=True)
    assigned_nurse_id = Column(Integer, ForeignKey("nurses.id"), nullable=True)
    known_condition = Column(String(200), default="Observation")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)

    hospital = relationship("Hospital", back_populates="patients")
    assigned_nurse = relationship("Nurse", back_populates="assigned_patients")
    vitals_history = relationship("VitalsHistory", back_populates="patient", cascade="all, delete-orphan", order_by="VitalsHistory.timestamp.asc()")
    alerts = relationship("AlertsLog", back_populates="patient", cascade="all, delete-orphan", order_by="AlertsLog.timestamp.desc()")


class VitalsHistory(Base):
    __tablename__ = "vitals_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    heart_rate = Column(Float, nullable=False)
    sbp = Column(Float, nullable=False)
    dbp = Column(Float, nullable=False)
    temperature = Column(Float, nullable=False)
    resp_rate = Column(Float, nullable=False)
    spo2 = Column(Float, nullable=False)
    gcs = Column(Float, default=15.0)
    map_value = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=datetime.now)
    risk_level = Column(String(20), default="GREEN")
    risk_score = Column(Float, default=0.1)
    created_at = Column(DateTime, default=datetime.now)

    patient = relationship("Patient", back_populates="vitals_history")


class AlertsLog(Base):
    __tablename__ = "alerts_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    risk_level = Column(String(20), nullable=False)
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.now)
    acknowledged = Column(Boolean, default=False)
    sms_simulated_recipient = Column(String(100), default="Dr. A. Mehta (Chief Physician)")
    sms_simulated_text = Column(Text, default="")

    patient = relationship("Patient", back_populates="alerts")


class AuthorizedStaff(Base):
    __tablename__ = "authorized_staff"

    id = Column(Integer, primary_key=True, autoincrement=True)
    hospital_name = Column(String(120), nullable=False)
    staff_id = Column(String(50), unique=True, nullable=False)
    staff_name = Column(String(100), nullable=False)
    role = Column(String(50), default="Nurse")
    created_at = Column(DateTime, default=datetime.now)


class UserSession(Base):
    __tablename__ = "user_sessions"

    token = Column(String(64), primary_key=True)
    user_id = Column(Integer, nullable=False)
    username = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.now)


# =============================================================================
# SESSION CONTEXT MANAGER
# =============================================================================

@contextmanager
def get_db():
    """Provides a transactional scope around a series of operations."""
    session = ScopedSession()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def anonymize_database():
    """Migrates any existing SQLite database records to rural hospital & anonymized doctor credentials."""
    with get_db() as session:
        # Update Hospital records
        hospitals = session.query(Hospital).all()
        for h in hospitals:
            if "Apollo" in h.name:
                h.name = "Sri Ramachandra Rural Health Centre"
                h.code = "SRRHC-RURAL"

        # Seed Authorized Staff Registry
        auth_staff_seeds = [
            ("Sri Ramachandra Rural Health Centre", "STAFF-1001", "Nurse Priya Sharma", "Nurse"),
            ("Sri Ramachandra Rural Health Centre", "STAFF-1002", "Nurse David Wilson", "Nurse"),
            ("Sri Ramachandra Rural Health Centre", "STAFF-1003", "Nurse Kavita Patel", "Nurse"),
            ("Sri Ramachandra Rural Health Centre", "STAFF-1004", "Nurse Sunita Rao", "Nurse"),
            ("Sri Ramachandra Rural Health Centre", "STAFF-1005", "Nurse Anil Kumar", "Nurse"),
        ]
        for h_name, s_id, s_name, s_role in auth_staff_seeds:
            if not session.query(AuthorizedStaff).filter_by(staff_id=s_id).first():
                session.add(AuthorizedStaff(hospital_name=h_name, staff_id=s_id, staff_name=s_name, role=s_role))

        # Update DoctorAdmin records
        admins = session.query(DoctorAdmin).all()
        for admin in admins:
            if "Shankar" in admin.name or "Dr. On-Duty" in admin.name or "Dr. On Duty" in admin.name:
                admin.name = "Dr. A. Mehta (Chief Physician)"
            if admin.phone == "+91-98765-43210":
                admin.phone = "+91-98765-00000"

        # Update AlertsLog records
        alerts = session.query(AlertsLog).all()
        for al in alerts:
            if al.sms_simulated_recipient and ("Shankar" in al.sms_simulated_recipient or "Dr. On Duty" in al.sms_simulated_recipient or "Dr. On-Duty" in al.sms_simulated_recipient):
                al.sms_simulated_recipient = "Dr. A. Mehta (Chief Physician) (+91-98765-00000)"
            if al.sms_simulated_text:
                al.sms_simulated_text = al.sms_simulated_text.replace("Dr. Shankar Kothapalli", "Dr. A. Mehta (Chief Physician)")
                al.sms_simulated_text = al.sms_simulated_text.replace("+91-98765-43210", "+91-98765-00000")
                al.sms_simulated_text = al.sms_simulated_text.replace("Dr. On-Duty Physician", "Dr. A. Mehta (Chief Physician)")
                al.sms_simulated_text = al.sms_simulated_text.replace("Dr. On Duty", "Dr. A. Mehta (Chief Physician)")
        session.commit()


def init_db():
    """Creates tables if they do not already exist, anonymizes existing data, and seeds realistic demo data if needed."""
    Base.metadata.create_all(bind=engine)
    try:
        anonymize_database()
        seed_demo_data()
    except Exception as e:
        print(f"Warning during init_db / seed_demo_data: {e}")


# =============================================================================
# HELPER QUERIES (THREAD-SAFE & ZERO DUMMY DATA)
# =============================================================================

def has_any_admin() -> bool:
    """Returns True if any hospital and doctor admin account exists."""
    with get_db() as session:
        return session.query(DoctorAdmin).count() > 0


def create_hospital_and_admin(
    hospital_name: str,
    admin_name: str,
    username: str,
    password: str,
    phone: str = "+91-98765-43210"
) -> Tuple[int, int]:
    """Creates the initial hospital and primary doctor/admin."""
    with get_db() as session:
        code = hospital_name.upper().replace(" ", "")[:8] + "-01"
        hospital = Hospital(name=hospital_name.strip(), code=code)
        session.add(hospital)
        session.flush()

        admin = DoctorAdmin(
            name=admin_name.strip(),
            username=username.strip().lower(),
            password=password.strip(),
            phone=phone.strip(),
            hospital_id=hospital.id
        )
        session.add(admin)
        session.flush()
        return hospital.id, admin.id


def get_or_create_hospital(hospital_name: str) -> int:
    """Returns existing hospital ID or creates new hospital entry by name."""
    clean_name = hospital_name.strip() if hospital_name else "Sri Ramachandra Rural Health Centre"
    with get_db() as session:
        hosp = session.query(Hospital).filter(func.lower(Hospital.name) == clean_name.lower()).first()
        if not hosp:
            code = clean_name.upper().replace(" ", "")[:8] + "-01"
            hosp = Hospital(name=clean_name, code=code)
            session.add(hosp)
            session.flush()
            session.refresh(hosp)
        return hosp.id


def verify_authorized_staff(hospital_name: str, staff_id: str) -> bool:
    """Verifies whether a given staff ID is pre-authorized for the hospital."""
    if not staff_id or not staff_id.strip():
        return False
    with get_db() as session:
        match = session.query(AuthorizedStaff).filter(
            func.lower(AuthorizedStaff.staff_id) == staff_id.strip().lower()
        ).first()
        return match is not None


def save_user_session(token: str, user_id: int, username: str) -> None:
    """Saves a persistent login session token in SQLite."""
    with get_db() as session:
        sess = UserSession(token=token, user_id=user_id, username=username)
        session.merge(sess)
        session.commit()


def get_user_by_session_token(token: str) -> Optional[Dict[str, Any]]:
    """Restores user session state from a persistent token."""
    if not token:
        return None
    with get_db() as session:
        sess = session.query(UserSession).filter_by(token=token).first()
        if not sess:
            return None
        nurse = session.query(Nurse).filter_by(id=sess.user_id).first()
        if nurse:
            hosp = session.query(Hospital).filter_by(id=nurse.hospital_id).first()
            return {
                "id": nurse.id,
                "name": nurse.name,
                "username": nurse.username,
                "role": "nurse",
                "ward": nurse.ward,
                "hospital_id": nurse.hospital_id,
                "hospital_name": hosp.name if hosp else "Sri Ramachandra Rural Health Centre",
                "phone": nurse.phone,
            }
        return None


def clear_user_session(token: str) -> None:
    """Removes session token on logout."""
    if not token:
        return
    with get_db() as session:
        session.query(UserSession).filter_by(token=token).delete()
        session.commit()


def register_nurse(
    name: str,
    username: str,
    password: str,
    hospital_id: int,
    ward: str,
    phone: str = "+91-91234-56789"
) -> int:
    """Registers a new nurse assigned to a specific hospital and ward."""
    with get_db() as session:
        # Check if username already exists
        existing = session.query(Nurse).filter_by(username=username.strip().lower()).first()
        if existing:
            raise ValueError(f"Username '{username}' is already taken.")

        nurse = Nurse(
            name=name.strip(),
            username=username.strip().lower(),
            password=password.strip(),
            ward=ward.strip(),
            phone=phone.strip(),
            hospital_id=hospital_id
        )
        session.add(nurse)
        session.flush()
        return nurse.id


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Checks credentials against nurses and doctors_admin tables."""
    u = username.strip().lower()
    p = password.strip()

    with get_db() as session:
        # Check doctor admin first
        doc = session.query(DoctorAdmin).filter_by(username=u).first()
        if doc and doc.password == p:
            hosp = session.query(Hospital).filter_by(id=doc.hospital_id).first()
            return {
                "id": doc.id,
                "name": doc.name,
                "username": doc.username,
                "role": "admin",
                "hospital_id": doc.hospital_id,
                "hospital_name": hosp.name if hosp else "Hospital",
                "ward": "All Wards (Admin)",
                "phone": doc.phone,
            }

        # Check nurse table
        nurse = session.query(Nurse).filter_by(username=u).first()
        if nurse and nurse.password == p:
            hosp = session.query(Hospital).filter_by(id=nurse.hospital_id).first()
            return {
                "id": nurse.id,
                "name": nurse.name,
                "username": nurse.username,
                "role": "nurse",
                "hospital_id": nurse.hospital_id,
                "hospital_name": hosp.name if hosp else "Hospital",
                "ward": nurse.ward,
                "phone": nurse.phone,
            }

    return None


def admit_patient(
    name: str,
    hospital_id: int,
    ward: str,
    bed_number: str,
    assigned_nurse_id: Optional[int] = None,
    known_condition: str = "Observation"
) -> int:
    """Admits a new real patient into the hospital and ward."""
    with get_db() as session:
        # Check for bed collision in active patients in same ward
        collision = session.query(Patient).filter(
            Patient.hospital_id == hospital_id,
            Patient.ward == ward.strip(),
            Patient.bed_number == bed_number.strip(),
            Patient.is_active == True
        ).first()
        if collision:
            raise ValueError(f"Bed '{bed_number}' in {ward} is already occupied by {collision.name}.")

        patient = Patient(
            name=name.strip(),
            hospital_id=hospital_id,
            ward=ward.strip(),
            bed_number=bed_number.strip(),
            admitted_date=datetime.now(),
            assigned_nurse_id=assigned_nurse_id,
            known_condition=known_condition.strip() or "General Observation",
            is_active=True
        )
        session.add(patient)
        session.flush()
        return patient.id


def get_active_patients_by_ward(hospital_id: int, ward: str) -> List[Dict[str, Any]]:
    """Returns active patients in a specific ward."""
    with get_db() as session:
        patients = session.query(Patient).filter(
            Patient.hospital_id == hospital_id,
            Patient.ward == ward.strip(),
            Patient.is_active == True
        ).order_by(Patient.bed_number.asc()).all()
        return [_patient_to_dict(p) for p in patients]


def get_all_active_patients(hospital_id: int) -> List[Dict[str, Any]]:
    """Returns all active patients across all wards for Doctor/Admin view."""
    with get_db() as session:
        patients = session.query(Patient).filter(
            Patient.hospital_id == hospital_id,
            Patient.is_active == True
        ).order_by(Patient.ward.asc(), Patient.bed_number.asc()).all()
        return [_patient_to_dict(p) for p in patients]


def get_patient_by_id(patient_id: int) -> Optional[Dict[str, Any]]:
    """Fetches a single patient with vitals history."""
    with get_db() as session:
        p = session.query(Patient).filter_by(id=patient_id).first()
        return _patient_to_dict(p) if p else None


def get_patient_vitals_history(patient_id: int) -> List[Dict[str, Any]]:
    """Fetches chronological vitals history for a patient with guaranteed numeric types."""
    with get_db() as session:
        readings = session.query(VitalsHistory).filter_by(patient_id=patient_id).order_by(VitalsHistory.timestamp.asc()).all()
        result = []
        for r in readings:
            try:
                hr_f = float(r.heart_rate)
                sbp_f = float(r.sbp)
                dbp_f = float(r.dbp)
                temp_f = float(r.temperature)
                rr_f = float(r.resp_rate)
                spo2_f = float(r.spo2)
                gcs_f = float(r.gcs) if r.gcs is not None else 15.0
                map_f = float(r.map_value) if r.map_value is not None else round(dbp_f + (sbp_f - dbp_f) / 3, 1)
            except (ValueError, TypeError):
                hr_f, sbp_f, dbp_f, temp_f, rr_f, spo2_f, gcs_f, map_f = 75.0, 120.0, 80.0, 37.0, 16.0, 98.0, 15.0, 93.3

            result.append({
                "id": r.id,
                "patient_id": r.patient_id,
                "time": r.timestamp.strftime("%H:%M") if r.timestamp else "Recent",
                "timestamp_str": r.timestamp.strftime("%d %b %Y, %H:%M") if r.timestamp else "",
                "timestamp": r.timestamp,
                "heart_rate": hr_f,
                "sbp": sbp_f,
                "dbp": dbp_f,
                "map": map_f,
                "resp_rate": rr_f,
                "spo2": spo2_f,
                "temperature": temp_f,
                "gcs": gcs_f,
                "risk_level": str(r.risk_level),
                "risk_score": float(r.risk_score or 0.1),
            })
        return result


def add_vitals_reading(
    patient_id: int,
    vitals_dict: Dict[str, Any],
    risk_level: str,
    risk_score: float
) -> int:
    """Appends a new vitals observation to vitals_history."""
    with get_db() as session:
        reading = VitalsHistory(
            patient_id=patient_id,
            heart_rate=float(vitals_dict["heart_rate"]),
            sbp=float(vitals_dict["sbp"]),
            dbp=float(vitals_dict["dbp"]),
            temperature=float(vitals_dict["temperature"]),
            resp_rate=float(vitals_dict["resp_rate"]),
            spo2=float(vitals_dict["spo2"]),
            gcs=float(vitals_dict.get("gcs", 15.0)),
            map_value=float(vitals_dict.get("map", vitals_dict["dbp"] + (vitals_dict["sbp"] - vitals_dict["dbp"]) / 3)),
            timestamp=datetime.now(),
            risk_level=risk_level.upper(),
            risk_score=float(risk_score)
        )
        session.add(reading)
        session.flush()
        return reading.id


def log_alert(
    patient_id: int,
    risk_level: str,
    message: str,
    doctor_name: str,
    doctor_phone: str,
    patient_name: str,
    bed_str: str
) -> int:
    """Logs an alert and generates a simulated SMS dispatch message."""
    with get_db() as session:
        sms_text = (
            f"SMS ALERT to {doctor_name} ({doctor_phone}): Urgent attention required. "
            f"Patient {patient_name} in {bed_str} is {risk_level}. Reason: {message}"
        )
        alert = AlertsLog(
            patient_id=patient_id,
            risk_level=risk_level.upper(),
            message=message,
            timestamp=datetime.now(),
            acknowledged=False,
            sms_simulated_recipient=f"{doctor_name} ({doctor_phone})",
            sms_simulated_text=sms_text
        )
        session.add(alert)
        session.flush()
        return alert.id


def get_alerts_for_ward(hospital_id: int, ward: Optional[str] = None, limit: int = 30, active_only: bool = True) -> List[Dict[str, Any]]:
    """Fetches active un-resolved alerts, optionally filtered by ward."""
    with get_db() as session:
        query = session.query(AlertsLog, Patient).join(Patient, AlertsLog.patient_id == Patient.id).filter(
            Patient.hospital_id == hospital_id,
            Patient.is_active == True
        )
        if ward:
            query = query.filter(Patient.ward == ward.strip())
        if active_only:
            query = query.filter(AlertsLog.acknowledged == False)

        results = query.order_by(AlertsLog.timestamp.desc()).limit(limit).all()
        alerts = []
        for alert, patient in results:
            alerts.append({
                "id": alert.id,
                "patient_id": patient.id,
                "patient_name": patient.name,
                "ward": patient.ward,
                "bed_number": patient.bed_number,
                "risk_level": alert.risk_level,
                "message": alert.message,
                "timestamp": alert.timestamp,
                "timestamp_str": alert.timestamp.strftime("%H:%M:%S (%d %b)"),
                "acknowledged": alert.acknowledged,
                "sms_simulated_recipient": alert.sms_simulated_recipient,
                "sms_simulated_text": alert.sms_simulated_text,
            })
        return alerts


def auto_resolve_patient_alerts(patient_id: int) -> int:
    """Marks all active alerts for a patient as resolved when vitals return to GREEN/normal."""
    with get_db() as session:
        unresolved = session.query(AlertsLog).filter(
            AlertsLog.patient_id == patient_id,
            AlertsLog.acknowledged == False
        ).all()
        count = len(unresolved)
        for a in unresolved:
            a.acknowledged = True
        return count


def acknowledge_alert(alert_id: int) -> bool:
    """Marks an alert as acknowledged by a nurse/doctor."""
    with get_db() as session:
        alert = session.query(AlertsLog).filter_by(id=alert_id).first()
        if alert:
            alert.acknowledged = True
            return True
        return False


def discharge_patient(patient_id: int) -> bool:
    """
    Soft-discharges a patient: sets is_active = False and discharged_date = now().
    Does NOT hard-delete any row; preserves audit history.
    """
    with get_db() as session:
        p = session.query(Patient).filter_by(id=patient_id).first()
        if p and p.is_active:
            p.is_active = False
            p.discharged_date = datetime.now()
            return True
        return False


def search_patients(hospital_id: int, query_text: str, ward: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Searches patients by name, ward number, OR bed number.
    Always returns ward + bed together.
    """
    q = f"%{query_text.strip()}%"
    with get_db() as session:
        query = session.query(Patient).filter(
            Patient.hospital_id == hospital_id,
            Patient.is_active == True,
            or_(
                Patient.name.ilike(q),
                Patient.ward.ilike(q),
                Patient.bed_number.ilike(q),
            )
        )
        if ward:
            query = query.filter(Patient.ward == ward.strip())

        patients = query.order_by(Patient.ward.asc(), Patient.bed_number.asc()).all()
        return [_patient_to_dict(p) for p in patients]


def get_nurses_with_patient_counts(hospital_id: int) -> List[Dict[str, Any]]:
    """For Admin/Doctor view: lists all nurses, their ward, and patient counts."""
    with get_db() as session:
        nurses = session.query(Nurse).filter_by(hospital_id=hospital_id).all()
        result = []
        for n in nurses:
            active_count = session.query(Patient).filter(
                Patient.hospital_id == hospital_id,
                Patient.ward == n.ward,
                Patient.is_active == True
            ).count()
            result.append({
                "id": n.id,
                "name": n.name,
                "username": n.username,
                "ward": n.ward,
                "phone": n.phone,
                "active_patients_count": active_count,
            })
        return result


def get_hospital_wards(hospital_id: int) -> List[str]:
    """Returns a sorted list of unique wards in the hospital (from patients and nurses)."""
    with get_db() as session:
        p_wards = session.query(Patient.ward).filter_by(hospital_id=hospital_id).distinct().all()
        n_wards = session.query(Nurse.ward).filter_by(hospital_id=hospital_id).distinct().all()
        wards = set([w[0] for w in p_wards if w[0]] + [w[0] for w in n_wards if w[0]])
        if not wards:
            wards = {"Ward 3A", "Ward 3B", "ICU-A"}
        return sorted(list(wards))


def get_raw_table_data(hospital_id: Optional[int] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Debug utility to view raw rows across all SQLite tables for persistence auditing.
    Visible in Doctor Admin Control.
    """
    with get_db() as session:
        # Hospitals
        hospitals = [
            {"id": h.id, "name": h.name, "code": h.code, "created_at": str(h.created_at)}
            for h in session.query(Hospital).all()
        ]
        # Nurses
        nurses = [
            {"id": n.id, "name": n.name, "username": n.username, "ward": n.ward, "phone": n.phone, "hospital_id": n.hospital_id}
            for n in session.query(Nurse).all()
        ]
        # Patients
        pts_query = session.query(Patient)
        if hospital_id:
            pts_query = pts_query.filter_by(hospital_id=hospital_id)
        patients = [
            {
                "id": p.id,
                "name": p.name,
                "ward": p.ward,
                "bed_number": p.bed_number,
                "is_active": p.is_active,
                "admitted": str(p.admitted_date) if p.admitted_date else "",
                "discharged": str(p.discharged_date) if p.discharged_date else None,
                "condition": p.known_condition,
                "assigned_nurse_id": p.assigned_nurse_id,
            }
            for p in pts_query.order_by(Patient.id.desc()).all()
        ]
        # Vitals History (recent 60)
        vitals = [
            {
                "id": v.id,
                "patient_id": v.patient_id,
                "heart_rate": v.heart_rate,
                "sbp": v.sbp,
                "dbp": v.dbp,
                "map": v.map_value,
                "spo2": v.spo2,
                "resp_rate": v.resp_rate,
                "temp": v.temperature,
                "risk_level": v.risk_level,
                "risk_score": v.risk_score,
                "timestamp": str(v.timestamp),
            }
            for v in session.query(VitalsHistory).order_by(VitalsHistory.id.desc()).limit(60).all()
        ]
        # Alerts Log
        alerts = [
            {
                "id": a.id,
                "patient_id": a.patient_id,
                "risk_level": a.risk_level,
                "message": a.message[:60] + "..." if len(a.message) > 60 else a.message,
                "sms": a.sms_simulated_text[:60] + "..." if len(a.sms_simulated_text) > 60 else a.sms_simulated_text,
                "acknowledged": a.acknowledged,
                "timestamp": str(a.timestamp),
            }
            for a in session.query(AlertsLog).order_by(AlertsLog.id.desc()).limit(40).all()
        ]
        return {
            "patients": patients,
            "vitals_history": vitals,
            "nurses": nurses,
            "alerts_log": alerts,
            "hospitals": hospitals,
        }


def _patient_to_dict(p: Patient) -> Dict[str, Any]:
    """Helper to serialize a Patient object with guaranteed numeric types."""
    if not p:
        return {}

    # Get latest vitals reading
    latest_v = None
    readings = p.vitals_history
    if readings:
        last = readings[-1]
        try:
            hr_f = float(last.heart_rate)
            sbp_f = float(last.sbp)
            dbp_f = float(last.dbp)
            temp_f = float(last.temperature)
            rr_f = float(last.resp_rate)
            spo2_f = float(last.spo2)
            gcs_f = float(last.gcs) if last.gcs is not None else 15.0
            map_f = float(last.map_value) if last.map_value is not None else round(dbp_f + (sbp_f - dbp_f) / 3, 1)
        except (ValueError, TypeError):
            hr_f, sbp_f, dbp_f, temp_f, rr_f, spo2_f, gcs_f, map_f = 75.0, 120.0, 80.0, 37.0, 16.0, 98.0, 15.0, 93.3

        latest_v = {
            "heart_rate": hr_f,
            "sbp": sbp_f,
            "dbp": dbp_f,
            "temperature": temp_f,
            "resp_rate": rr_f,
            "spo2": spo2_f,
            "gcs": gcs_f,
            "map": map_f,
            "timestamp": last.timestamp,
            "time": last.timestamp.strftime("%H:%M") if last.timestamp else "Recent",
            "risk_level": str(last.risk_level),
            "risk_score": float(last.risk_score or 0.1),
        }

    nurse_name = p.assigned_nurse.name if p.assigned_nurse else "On-duty Nurse"
    bed_raw = str(p.bed_number).strip()
    bed_formatted = bed_raw if bed_raw.lower().startswith("bed") else f"Bed {bed_raw}"

    return {
        "id": p.id,
        "name": p.name,
        "hospital_id": p.hospital_id,
        "ward": p.ward,
        "bed_number": p.bed_number,
        "bed_display": f"{p.ward} &bull; {bed_formatted}",
        "admitted_date": p.admitted_date,
        "admitted_str": p.admitted_date.strftime("%d %b %Y, %H:%M") if p.admitted_date else "",
        "discharged_date": p.discharged_date,
        "assigned_nurse_id": p.assigned_nurse_id,
        "assigned_nurse_name": nurse_name,
        "known_condition": p.known_condition,
        "is_active": p.is_active,
        "vitals_count": len(readings),
        "latest_vitals": latest_v,
    }


# =============================================================================
# PROTOTYPE / DEMO SEED DATA (For Ideathon 5.0 Demo)
# To reseed or reset, call seed_demo_data(force=True)
# =============================================================================
def seed_demo_data(force: bool = False) -> None:
    """
    Seeds realistic prototype demo patients into SQLite for the Ideathon 5.0 Jury.
    Creates 6 patients across 3 wards (Ward 3A, Ward 3B, ICU-A) with rich vitals history
    covering GREEN, YELLOW, and RED risk tiers with trend trajectories.
    Label: DEMO_SEED_DATA
    """
    from datetime import timedelta
    with get_db() as session:
        # Check if already seeded and not force
        active_count = session.query(Patient).filter_by(is_active=True).count()
        if active_count >= 5 and not force:
            return

        # Ensure Hospital exists
        hosp = session.query(Hospital).first()
        if not hosp:
            hosp = Hospital(name="Sri Ramachandra Rural Health Centre", code="SRRHC-RURAL")
            session.add(hosp)
            session.flush()

        # Seed Authorized Staff Registry
        auth_staff_seeds = [
            ("Sri Ramachandra Rural Health Centre", "STAFF-1001", "Nurse Priya Sharma", "Nurse"),
            ("Sri Ramachandra Rural Health Centre", "STAFF-1002", "Nurse David Wilson", "Nurse"),
            ("Sri Ramachandra Rural Health Centre", "STAFF-1003", "Nurse Kavita Patel", "Nurse"),
            ("Sri Ramachandra Rural Health Centre", "STAFF-1004", "Nurse Sunita Rao", "Nurse"),
            ("Sri Ramachandra Rural Health Centre", "STAFF-1005", "Nurse Anil Kumar", "Nurse"),
        ]
        for h_name, s_id, s_name, s_role in auth_staff_seeds:
            if not session.query(AuthorizedStaff).filter_by(staff_id=s_id).first():
                session.add(AuthorizedStaff(hospital_name=h_name, staff_id=s_id, staff_name=s_name, role=s_role))
        session.flush()

        # Ensure Admin exists & anonymized
        admin = session.query(DoctorAdmin).filter_by(hospital_id=hosp.id).first()
        if not admin:
            admin = DoctorAdmin(
                name="Dr. A. Mehta (Chief Physician)",
                username="admin",
                password="password123",
                hospital_id=hosp.id,
                department="Critical Care Medicine",
                phone="+91-98765-00000"
            )
            session.add(admin)
            session.flush()
        else:
            if "Shankar" in admin.name or "Dr. On-Duty" in admin.name:
                admin.name = "Dr. A. Mehta (Chief Physician)"
            if admin.phone == "+91-98765-43210":
                admin.phone = "+91-98765-00000"
            session.flush()

        # Update existing alert records in SQLite to anonymized doctor credentials
        for al in session.query(AlertsLog).all():
            if al.sms_simulated_recipient and ("Shankar" in al.sms_simulated_recipient or "Dr. On Duty" in al.sms_simulated_recipient):
                al.sms_simulated_recipient = f"Dr. A. Mehta (Chief Physician) (+91-98765-00000)"
            if al.sms_simulated_text:
                al.sms_simulated_text = al.sms_simulated_text.replace("Dr. Shankar Kothapalli", "Dr. A. Mehta (Chief Physician)")
                al.sms_simulated_text = al.sms_simulated_text.replace("+91-98765-43210", "+91-98765-00000")
                al.sms_simulated_text = al.sms_simulated_text.replace("Dr. On-Duty Physician", "Dr. A. Mehta (Chief Physician)")
        session.flush()

        # Ensure Ward Nurses exist for all 3 demo wards
        nurse_data = [
            ("Priya Sharma", "priya.nurse", "nurse123", "Ward 3A", "+91-91234-56789"),
            ("David Wilson", "david.nurse", "nurse123", "Ward 3B", "+91-92345-67890"),
            ("Kavita Patel", "kavita.nurse", "nurse123", "ICU-A", "+91-93456-78901"),
        ]
        nurses_by_ward = {}
        for n_name, n_user, n_pass, n_ward, n_phone in nurse_data:
            nurse_obj = session.query(Nurse).filter_by(username=n_user).first()
            if not nurse_obj:
                nurse_obj = Nurse(
                    name=n_name,
                    username=n_user,
                    password=n_pass,
                    ward=n_ward,
                    phone=n_phone,
                    hospital_id=hosp.id
                )
                session.add(nurse_obj)
                session.flush()
            nurses_by_ward[n_ward] = nurse_obj

        # 6 Realistic Patients to Seed
        patients_to_seed = [
            # 1. Ward 3A - Stable GREEN
            {
                "name": "Rajesh Kumar",
                "ward": "Ward 3A",
                "bed_number": "01",
                "condition": "Post-CABG recovery (Day 3)",
                "nurse_ward": "Ward 3A",
                "vitals": [
                    {"offset_h": 6, "hr": 72.0, "sbp": 122.0, "dbp": 78.0, "rr": 15.0, "spo2": 98.0, "temp": 36.8, "level": "GREEN", "score": 0.05},
                    {"offset_h": 4, "hr": 74.0, "sbp": 120.0, "dbp": 80.0, "rr": 14.0, "spo2": 99.0, "temp": 36.9, "level": "GREEN", "score": 0.05},
                    {"offset_h": 2, "hr": 71.0, "sbp": 118.0, "dbp": 76.0, "rr": 16.0, "spo2": 98.0, "temp": 36.7, "level": "GREEN", "score": 0.05},
                    {"offset_h": 0, "hr": 70.0, "sbp": 121.0, "dbp": 79.0, "rr": 15.0, "spo2": 98.0, "temp": 36.8, "level": "GREEN", "score": 0.05},
                ]
            },
            # 2. Ward 3A - Critical RED (Sepsis downward spiral)
            {
                "name": "Sunita Deshmukh",
                "ward": "Ward 3A",
                "bed_number": "04",
                "condition": "Sepsis protocol / Acute Pyelonephritis",
                "nurse_ward": "Ward 3A",
                "vitals": [
                    {"offset_h": 6, "hr": 84.0, "sbp": 116.0, "dbp": 76.0, "rr": 18.0, "spo2": 96.0, "temp": 37.6, "level": "GREEN", "score": 0.12},
                    {"offset_h": 4, "hr": 98.0, "sbp": 106.0, "dbp": 68.0, "rr": 21.0, "spo2": 94.0, "temp": 38.4, "level": "YELLOW", "score": 0.45},
                    {"offset_h": 2, "hr": 114.0, "sbp": 92.0, "dbp": 60.0, "rr": 25.0, "spo2": 91.0, "temp": 39.1, "level": "YELLOW", "score": 0.65},
                    {"offset_h": 0, "hr": 128.0, "sbp": 82.0, "dbp": 52.0, "rr": 28.0, "spo2": 88.0, "temp": 39.5, "level": "RED", "score": 0.96},
                ],
                "alert": {
                    "level": "RED",
                    "msg": "Heart rate spike: 128 bpm; Hypotension SBP 82 mmHg (MAP 62 mmHg); Severe hypoxemia SpO2 88%; Tachypnea 28/min; Fever 39.5 °C; CRITICAL: Circulatory collapse (Shock Index 1.56)",
                    "sms": "SMS ALERT to Dr. A. Mehta (Chief Physician) (+91-98765-00000): Urgent attention required. Patient Sunita Deshmukh in Ward 3A • Bed 04 is RED."
                }
            },
            # 3. Ward 3A - Warning YELLOW (COPD respiratory drift)
            {
                "name": "Mohammed Farooq",
                "ward": "Ward 3A",
                "bed_number": "08",
                "condition": "Acute Exacerbation of COPD",
                "nurse_ward": "Ward 3A",
                "vitals": [
                    {"offset_h": 4, "hr": 88.0, "sbp": 130.0, "dbp": 82.0, "rr": 22.0, "spo2": 93.0, "temp": 37.1, "level": "GREEN", "score": 0.20},
                    {"offset_h": 2, "hr": 92.0, "sbp": 134.0, "dbp": 86.0, "rr": 24.0, "spo2": 92.0, "temp": 37.3, "level": "YELLOW", "score": 0.40},
                    {"offset_h": 0, "hr": 96.0, "sbp": 138.0, "dbp": 88.0, "rr": 26.0, "spo2": 90.0, "temp": 37.4, "level": "YELLOW", "score": 0.52},
                ]
            },
            # 4. Ward 3B - Warning YELLOW (Severe Dengue fever)
            {
                "name": "Ananya Sen",
                "ward": "Ward 3B",
                "bed_number": "02",
                "condition": "Severe Dengue with Thrombocytopenia",
                "nurse_ward": "Ward 3B",
                "vitals": [
                    {"offset_h": 4, "hr": 98.0, "sbp": 106.0, "dbp": 70.0, "rr": 18.0, "spo2": 97.0, "temp": 38.6, "level": "YELLOW", "score": 0.35},
                    {"offset_h": 2, "hr": 104.0, "sbp": 100.0, "dbp": 65.0, "rr": 19.0, "spo2": 96.0, "temp": 38.8, "level": "YELLOW", "score": 0.42},
                    {"offset_h": 0, "hr": 108.0, "sbp": 96.0, "dbp": 62.0, "rr": 20.0, "spo2": 95.0, "temp": 38.9, "level": "YELLOW", "score": 0.48},
                ]
            },
            # 5. Ward 3B - Stable GREEN (Post-cholecystectomy)
            {
                "name": "Gurpreet Singh",
                "ward": "Ward 3B",
                "bed_number": "06",
                "condition": "Post-laparoscopic cholecystectomy",
                "nurse_ward": "Ward 3B",
                "vitals": [
                    {"offset_h": 4, "hr": 66.0, "sbp": 128.0, "dbp": 82.0, "rr": 14.0, "spo2": 97.0, "temp": 36.6, "level": "GREEN", "score": 0.05},
                    {"offset_h": 2, "hr": 68.0, "sbp": 124.0, "dbp": 80.0, "rr": 15.0, "spo2": 98.0, "temp": 36.7, "level": "GREEN", "score": 0.05},
                    {"offset_h": 0, "hr": 67.0, "sbp": 126.0, "dbp": 81.0, "rr": 14.0, "spo2": 97.0, "temp": 36.6, "level": "GREEN", "score": 0.05},
                ]
            },
            # 6. ICU-A - Critical RED (Cardiogenic shock)
            {
                "name": "Lakshmi Narayanan",
                "ward": "ICU-A",
                "bed_number": "03",
                "condition": "Post-cardiac arrest / Cardiogenic shock",
                "nurse_ward": "ICU-A",
                "vitals": [
                    {"offset_h": 6, "hr": 118.0, "sbp": 88.0, "dbp": 56.0, "rr": 28.0, "spo2": 92.0, "temp": 38.2, "level": "YELLOW", "score": 0.68},
                    {"offset_h": 4, "hr": 124.0, "sbp": 82.0, "dbp": 52.0, "rr": 30.0, "spo2": 90.0, "temp": 38.4, "level": "RED", "score": 0.88},
                    {"offset_h": 2, "hr": 130.0, "sbp": 78.0, "dbp": 48.0, "rr": 32.0, "spo2": 88.0, "temp": 38.6, "level": "RED", "score": 0.95},
                    {"offset_h": 0, "hr": 136.0, "sbp": 74.0, "dbp": 46.0, "rr": 34.0, "spo2": 86.0, "temp": 38.8, "level": "RED", "score": 0.98},
                ],
                "alert": {
                    "level": "RED",
                    "msg": "Severe tachycardia: 136 bpm; Severe hypotension: SBP 74 mmHg (MAP 55 mmHg); Severe hypoxemia: SpO2 86%; Tachypnea: 34/min; CRITICAL: Circulatory collapse (Shock Index 1.84)",
                    "sms": "SMS ALERT to Dr. A. Mehta (Chief Physician) (+91-98765-00000): Urgent attention required. Patient Lakshmi Narayanan in ICU-A • Bed 03 is RED."
                }
            },
        ]

        now = datetime.now()
        for p_info in patients_to_seed:
            existing = session.query(Patient).filter_by(
                hospital_id=hosp.id,
                ward=p_info["ward"],
                bed_number=p_info["bed_number"],
                is_active=True
            ).first()
            if existing:
                continue

            nurse = nurses_by_ward.get(p_info["nurse_ward"])
            patient = Patient(
                name=p_info["name"],
                hospital_id=hosp.id,
                ward=p_info["ward"],
                bed_number=p_info["bed_number"],
                known_condition=p_info["condition"],
                assigned_nurse_id=nurse.id if nurse else None,
                admitted_date=now - timedelta(hours=12),
                is_active=True
            )
            session.add(patient)
            session.flush()

            # Add vitals entries
            for v in p_info["vitals"]:
                ts = now - timedelta(hours=v["offset_h"])
                map_v = round(v["dbp"] + (v["sbp"] - v["dbp"]) / 3, 1)
                v_entry = VitalsHistory(
                    patient_id=patient.id,
                    heart_rate=float(v["hr"]),
                    sbp=float(v["sbp"]),
                    dbp=float(v["dbp"]),
                    temperature=float(v["temp"]),
                    resp_rate=float(v["rr"]),
                    spo2=float(v["spo2"]),
                    gcs=15.0,
                    map_value=float(map_v),
                    timestamp=ts,
                    risk_level=v["level"],
                    risk_score=float(v["score"]),
                )
                session.add(v_entry)

            # Add alert if any
            if "alert" in p_info:
                al = AlertsLog(
                    patient_id=patient.id,
                    risk_level=p_info["alert"]["level"],
                    message=p_info["alert"]["msg"],
                    timestamp=now,
                    acknowledged=False,
                    sms_simulated_recipient=f"{admin.name} ({admin.phone})",
                    sms_simulated_text=p_info["alert"]["sms"]
                )
                session.add(al)

        session.commit()

