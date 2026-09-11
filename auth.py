"""
VitalGuard Authentication & Nurse Role Management Module.

Implements:
- Nurse-only clinical surveillance authentication
- Nurse registration with hospital staff ID pre-authorization check
- Persistent session storage in SQLite surviving browser page refresh
"""

import uuid
from typing import Dict, Any, Optional
import streamlit as st

from db import (
    has_any_admin,
    create_hospital_and_admin,
    register_nurse,
    authenticate_user,
    get_or_create_hospital,
    save_user_session,
    get_user_by_session_token,
    clear_user_session,
    get_db,
    Hospital,
)


def init_auth_session():
    """Ensures auth session state variables exist and restores persistent session if token exists."""
    if "auth_user" not in st.session_state:
        st.session_state.auth_user = None

    if st.session_state.get("auth_user") is None:
        token = st.query_params.get("session")
        if token:
            restored_user = get_user_by_session_token(token)
            if restored_user:
                st.session_state.auth_user = restored_user


def is_authenticated() -> bool:
    """Returns True if a user is currently logged in."""
    init_auth_session()
    return st.session_state.get("auth_user") is not None


def get_current_user() -> Optional[Dict[str, Any]]:
    """Returns currently authenticated user dictionary or None."""
    init_auth_session()
    return st.session_state.auth_user


def logout_user():
    """Logs out the active user, clears persistent token in SQLite, and resets session."""
    token = st.query_params.get("session")
    if token:
        clear_user_session(token)
        st.query_params.clear()
    st.session_state.auth_user = None
    st.session_state.selected_patient_id = None
    st.rerun()


def render_login_page():
    """
    Renders Nurse Login and Authorized Nurse Registration tabs.
    Teal/coral hospital-grade styling.
    """
    init_auth_session()

    # Center card container
    _, col, _ = st.columns([1, 2, 1])

    with col:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 1.5rem;">
            <div style="display: inline-block; background: linear-gradient(135deg, #0D9488, #0F766E); color: white; font-weight: 800; font-size: 1.6rem; padding: 12px 20px; border-radius: 16px; box-shadow: 0 4px 14px rgba(13,148,136,0.35); margin-bottom: 12px;">
                VG
            </div>
            <h2 style="color: #0F172A; font-weight: 800; margin: 0; letter-spacing: -0.5px;">VitalGuard Rural Health Portal</h2>
            <p style="color: #64748B; font-size: 0.88rem; margin-top: 4px;">Sri Ramachandra Rural Health Centre &bull; Nurse Surveillance Gateway</p>
        </div>
        """, unsafe_allow_html=True)

        if not has_any_admin():
            # First-Time Setup flow
            st.info("🏥 **First-Time System Initialization**: Creating clinical institution records.")
            with st.form("setup_form"):
                st.markdown("##### 🏛️ Hospital Setup")
                h_name = st.text_input("Hospital Name", value="Sri Ramachandra Rural Health Centre")
                d_name = st.text_input("Chief Physician Name", value="Dr. A. Mehta (Chief Physician)")
                d_phone = st.text_input("Doctor Alert Phone", value="+91-98765-00000")
                d_user = st.text_input("Admin Username", value="admin")
                d_pass = st.text_input("Admin Password", type="password", value="admin123")
                setup_btn = st.form_submit_button("🚀 Initialize Hospital System", use_container_width=True)

            if setup_btn:
                h_id, a_id = create_hospital_and_admin(h_name, d_name, d_user, d_pass, d_phone)
                st.success("Hospital initialized successfully!")
                st.rerun()

        else:
            # Login vs Registration Tabs
            tab_login, tab_register = st.tabs(["🔐 Nurse Login", "👩‍⚕️ Nurse Registration"])

            with tab_login:
                with st.form("login_form"):
                    st.markdown("##### 🔐 Nurse Login")
                    username = st.text_input("Username", placeholder="Enter your nurse username (e.g., priya.nurse)")
                    password = st.text_input("Password", type="password", placeholder="Enter password")
                    login_btn = st.form_submit_button("Log In to Ward Surveillance", use_container_width=True)

                if login_btn:
                    if not username or not password:
                        st.warning("Please enter both username and password.")
                    else:
                        user_data = authenticate_user(username, password)
                        if user_data:
                            # Generate & persist session token
                            token = uuid.uuid4().hex
                            save_user_session(token, user_data["id"], user_data["username"])
                            st.query_params["session"] = token
                            st.session_state.auth_user = user_data
                            st.success(f"Welcome back, {user_data['name']}!")
                            st.rerun()
                        else:
                            st.error("Invalid credentials. Please check your username and password.")

            with tab_register:
                st.caption("Enter your nurse registration details below to create an account.")
                with st.form("nurse_registration_form"):
                    st.markdown("##### 👩‍⚕️ Register New Nurse Account")
                    r_name = st.text_input("Nurse Full Name", placeholder="e.g. Nurse Priya Sharma")
                    r_staff_id = st.text_input("Hospital Staff Registration ID", placeholder="e.g. STAFF-1001")
                    r_hosp_name = st.text_input("Hospital Name", value="Sri Ramachandra Rural Health Centre", placeholder="e.g. Grameen Community Hospital")
                    r_ward = st.selectbox("Assigned Ward", options=["Ward 3A", "Ward 3B", "ICU-A"])
                    r_phone = st.text_input("Mobile Phone Number", value="+91-91234-56789")
                    r_user = st.text_input("Choose Username", placeholder="e.g. priya.nurse")
                    r_pass = st.text_input("Choose Password", type="password", placeholder="Set initial password")

                    register_btn = st.form_submit_button("Register Account", use_container_width=True)

                if register_btn:
                    if not r_name or not r_staff_id or not r_user or not r_pass or not r_hosp_name:
                        st.error("Please fill in all required registration fields.")
                    else:
                        h_id = get_or_create_hospital(r_hosp_name)
                        try:
                            n_id = register_nurse(r_name, r_user, r_pass, h_id, r_ward, r_phone)
                            # Automatically log in
                            user_data = authenticate_user(r_user, r_pass)
                            if user_data:
                                token = uuid.uuid4().hex
                                save_user_session(token, user_data["id"], user_data["username"])
                                st.query_params["session"] = token
                                st.session_state.auth_user = user_data
                            st.success(f"Nurse account '{r_name}' registered successfully for {r_hosp_name}!")
                            st.rerun()
                        except ValueError as ve:
                            st.error(str(ve))
                        except Exception as ex:
                            st.error(f"Registration error: {ex}")

            st.caption("🔒 Secured local SQLite authentication.")
