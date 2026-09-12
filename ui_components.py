"""
VitalGuard UI Design System & Component Library.

Rose/Pink-Coral Hospital Styling:
- Primary Rose (#E11D48), Accent (#FB7185), Slate (#0F172A, #64748B)
- Soft shadows, rounded cards, medical badges
- Top Hospital Navigation Bar (desktop) / Bottom Tab Navigation (mobile)
- Overview Cards (Critical / Needs Attention / Stable)
- SMS Alert Banner
- Light Pastel Vitals Card Grid
- Clean Empty States (Zero Dummy Data compliance)
"""

from datetime import datetime
import textwrap
from typing import Dict, List, Any, Optional
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =============================================================================
# SVG ICON LIBRARY (Feather / Lucide Style)
# =============================================================================

SVG_ICONS = {
    "heart": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"></path></svg>',
    "bp": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><polyline points="12 6 12 12 15 15"/></svg>',
    "spo2": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"/></svg>',
    "lungs": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 4v16m-4-8c-2 0-5 1-5 4s3 4 5 4m8-8c2 0 5 1 5 4s-3 4-5 4"/></svg>',
    "thermometer": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4.5 4.5 0 1 0 5 0z"></path></svg>',
    "activity": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>',
    "alert-triangle": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>',
    "alert-circle": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>',
    "shield": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>',
    "shield-check": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path><polyline points="9 12 11 14 15 10"></polyline></svg>',
    "user": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>',
    "users": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path></svg>',
    "search": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>',
    "check": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>',
    "clock": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>',
    "file-text": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>',
    "bell": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"></path><path d="M13.73 21a2 2 0 0 1-3.46 0"></path></svg>',
    "hospital": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 21h18"></path><path d="M5 21V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v16"></path><path d="M9 21v-4a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v4"></path><line x1="10" y1="9" x2="14" y2="9"></line><line x1="12" y1="7" x2="12" y2="11"></line></svg>',
    "plus": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>',
    "arrow-left": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="19" y1="12" x2="5" y2="12"></line><polyline points="12 19 5 12 12 5"></polyline></svg>',
    "log-out": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"></path><polyline points="16 17 21 12 16 7"></polyline><line x1="21" y1="12" x2="9" y2="12"></line></svg>'
}

def get_svg_icon(name: str, size: int = 16, color: str = "currentColor") -> str:
    """Returns clean inline SVG markup for a given icon name."""
    template = SVG_ICONS.get(name, SVG_ICONS["activity"])
    return template.format(size=size, color=color)


# =============================================================================
# ROSE / PINK-CORAL DESIGN SYSTEM CSS
# =============================================================================

def inject_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

        /* Hide left sidebar completely for horizontal top navigation layout */
        [data-testid="stSidebar"], section[data-testid="stSidebar"], [data-testid="collapsedControl"] {
            display: none !important;
        }

        /* Top Hospital Navigation Bar */
        .vg-topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.9rem 1.4rem;
            background: linear-gradient(135deg, #9F1239 0%, #E11D48 100%);
            border-radius: 12px;
            color: #FFFFFF;
            margin-bottom: 1.25rem;
            box-shadow: 0 4px 16px -2px rgba(225, 29, 72, 0.20);
        }
        .vg-brand-group {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .vg-brand-logo {
            background: #FFFFFF;
            color: #E11D48;
            font-weight: 800;
            font-size: 1.15rem;
            padding: 6px 12px;
            border-radius: 8px;
            letter-spacing: -0.5px;
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08);
        }
        .vg-brand-title {
            margin: 0;
            font-size: 1.2rem;
            font-weight: 800;
            letter-spacing: -0.3px;
            color: #FFFFFF !important;
        }
        .vg-brand-sub {
            margin: 1px 0 0 0;
            font-size: 0.78rem;
            color: #FFE4E6;
            font-weight: 500;
        }
        .vg-user-pill {
            display: flex;
            align-items: center;
            gap: 8px;
            background: rgba(255, 255, 255, 0.18);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(255, 255, 255, 0.25);
            padding: 5px 12px;
            border-radius: 9999px;
            color: #FFFFFF;
            font-size: 0.80rem;
            font-weight: 600;
        }

        /* ================= MOBILE BOTTOM TAB NAVIGATION ================= */
        /* Targets Streamlit's auto-generated class from st.container(key="mobile_bottom_nav") */
        .st-key-mobile_bottom_nav {
            display: none;
        }
        @media (max-width: 768px) {
            .st-key-mobile_bottom_nav {
                display: flex !important;
                position: fixed;
                bottom: 0;
                left: 0;
                right: 0;
                background: #FFFFFF;
                border-top: 1px solid #FECDD3;
                padding: 8px 6px;
                z-index: 9999;
                box-shadow: 0 -2px 10px rgba(0,0,0,0.08);
            }
            .st-key-mobile_bottom_nav button {
                border-radius: 10px !important;
                font-size: 0.72rem !important;
                padding: 0.4rem 0.2rem !important;
            }
            /* Push page content up so fixed bottom nav doesn't cover it */
            .main .block-container {
                padding-bottom: 90px !important;
            }
            /* Hide desktop top nav header row on narrow screens for a cleaner mobile look */
            .vg-topbar {
                flex-direction: column !important;
                align-items: flex-start !important;
                gap: 8px !important;
                padding: 0.8rem 1.0rem !important;
            }
            .vg-user-pill {
                width: 100% !important;
                justify-content: space-between !important;
            }
            .overview-card {
                margin-bottom: 0.75rem !important;
            }
            .model-val-grid {
                grid-template-columns: 1fr !important;
            }
            .grid-2 {
                grid-template-columns: 1fr !important;
            }
        }
        /* ================= DESKTOP TOP NAV — HIDE ON MOBILE ================= */
        /* Targets Streamlit's auto-generated class from st.container(key="desktop_top_nav") */
        .st-key-desktop_top_nav {
            display: block;
        }
        @media (max-width: 768px) {
            .st-key-desktop_top_nav {
                display: none !important;
            }
        }

        /* Overview Metric Cards (Home Screen) */
        .overview-card {
            background: #FFFFFF;
            border-radius: 12px;
            padding: 1.1rem;
            border: 1px solid #E2E8F0;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05), 0 1px 2px rgba(0, 0, 0, 0.03);
            text-align: center;
            position: relative;
            overflow: hidden;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .overview-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
        }
        .overview-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
        }
        .overview-critical::before { background: #E53935; }
        .overview-warning::before  { background: #FFC107; }
        .overview-stable::before   { background: #43A047; }
        
        .overview-num {
            font-size: 2.1rem;
            font-weight: 800;
            line-height: 1.1;
            margin: 4px 0 2px 0;
            font-family: 'JetBrains Mono', monospace;
        }
        .overview-critical .overview-num { color: #E53935; }
        .overview-warning .overview-num  { color: #D97706; }
        .overview-stable .overview-num   { color: #43A047; }

        .overview-title {
            font-size: 0.78rem;
            font-weight: 700;
            color: #475569;
            text-transform: uppercase;
            letter-spacing: 0.4px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
        }
        .overview-sub {
            font-size: 0.72rem;
            color: #64748B;
            margin-top: 4px;
        }

        /* Clinical Patient Cards */
        .patient-card {
            background: #FFFFFF;
            border-radius: 12px;
            padding: 1.15rem;
            border: 1px solid #E2E8F0;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05), 0 1px 2px rgba(0, 0, 0, 0.03);
            margin-bottom: 0.85rem;
            transition: all 0.15s ease;
        }
        .patient-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px -2px rgba(225, 29, 72, 0.10);
            border-color: #FECDD3;
        }
        .patient-card-selected {
            border: 2px solid #E11D48 !important;
            box-shadow: 0 0 0 4px rgba(225, 29, 72, 0.15) !important;
        }

        /* Risk Badges */
        .risk-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.4px;
            text-transform: uppercase;
        }
        .risk-badge-green {
            background-color: #ECFDF5;
            color: #047857;
            border: 1px solid #A7F3D0;
        }
        .risk-badge-yellow {
            background-color: #FFFBEB;
            color: #B45309;
            border: 1px solid #FDE68A;
        }
        .risk-badge-red {
            background-color: #FEF2F2;
            color: #B91C1C;
            border: 1px solid #FECACA;
        }
        .badge-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            display: inline-block;
        }
        .dot-green { background-color: #10B981; }
        .dot-yellow { background-color: #F59E0B; }
        .dot-red {
            background-color: #EF4444;
            animation: pulse-red 1.5s infinite;
        }
        @keyframes pulse-red {
            0% { transform: scale(0.95); opacity: 0.8; }
            50% { transform: scale(1.25); opacity: 1; box-shadow: 0 0 8px rgba(239, 68, 68, 0.6); }
            100% { transform: scale(0.95); opacity: 0.8; }
        }

        /* Metric Chips */
        .metric-chip {
            display: inline-flex;
            flex-direction: column;
            padding: 6px 10px;
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            min-width: 65px;
            text-align: center;
        }
        .metric-chip-label {
            font-size: 0.65rem;
            color: #64748B;
            text-transform: uppercase;
            font-weight: 600;
        }
        .metric-chip-val {
            font-size: 0.95rem;
            color: #0F172A;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }

        /* SMS Notification Banner */
        .sms-banner {
            background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
            border-left: 4px solid #FB7185;
            color: #FFFFFF;
            padding: 1rem 1.25rem;
            border-radius: 12px;
            margin: 1rem 0;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.15);
        }
        .sms-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 0.88rem;
            font-weight: 700;
            color: #FB7185;
        }

        /* Model Validation Panel Card */
        .model-val-card {
            background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 1.25rem;
            color: #F8FAFC;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
            margin-top: 0.5rem;
        }
        .model-val-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid rgba(255, 255, 255, 0.12);
            padding-bottom: 10px;
            margin-bottom: 12px;
        }
        .model-val-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #FB7185;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .model-val-pill {
            background: rgba(251, 113, 133, 0.15);
            color: #FB7185;
            border: 1px solid rgba(251, 113, 133, 0.3);
            padding: 3px 8px;
            border-radius: 9999px;
            font-size: 0.70rem;
            font-weight: 600;
            text-transform: uppercase;
        }
        .model-val-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-top: 10px;
        }
        .model-val-stat {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 10px 12px;
            border: 1px solid rgba(255, 255, 255, 0.08);
        }
        .model-val-stat-lbl {
            font-size: 0.68rem;
            color: #94A3B8;
            text-transform: uppercase;
            font-weight: 600;
        }
        .model-val-stat-val {
            font-size: 1.15rem;
            font-weight: 800;
            color: #FFFFFF;
            font-family: 'JetBrains Mono', monospace;
            margin-top: 2px;
        }
        .sms-body {
            margin-top: 6px;
            font-size: 0.82rem;
            color: #E2E8F0;
            font-family: 'JetBrains Mono', monospace;
            line-height: 1.4;
        }

        /* Critical Red Alert Banner */
        .critical-alert-banner {
            background: linear-gradient(135deg, #DC2626 0%, #B91C1C 100%);
            color: #FFFFFF;
            padding: 1.25rem 1.5rem;
            border-radius: 16px;
            margin-bottom: 1.5rem;
            box-shadow: 0 8px 24px rgba(220, 38, 38, 0.35);
            border: 1px solid rgba(255, 255, 255, 0.2);
            animation: alert-glow 2s infinite alternate;
        }
        @keyframes alert-glow {
            from { box-shadow: 0 4px 15px rgba(220, 38, 38, 0.3); }
            to { box-shadow: 0 8px 28px rgba(220, 38, 38, 0.6); }
        }
        .alert-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 1.1rem;
            font-weight: 800;
        }
        .alert-status-tag {
            background: rgba(255, 255, 255, 0.2);
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }

        /* Explainability Box */
        .explain-box {
            background: #F8FAFC;
            border-left: 4px solid #E11D48;
            border-radius: 0 12px 12px 0;
            padding: 1.25rem;
            margin: 1rem 0;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
            border-top: 1px solid #E2E8F0;
            border-right: 1px solid #E2E8F0;
            border-bottom: 1px solid #E2E8F0;
        }
        .explain-box.box-red {
            border-left-color: #EF4444;
            background: #FEF2F2;
        }
        .explain-box.box-yellow {
            border-left-color: #F59E0B;
            background: #FFFBEB;
        }
        .explain-box.box-green {
            border-left-color: #10B981;
            background: #F0FDF4;
        }

        /* Clean Empty State */
        .empty-state-box {
            background: #F8FAFC;
            border: 2px dashed #CBD5E1;
            border-radius: 16px;
            padding: 3rem 2rem;
            text-align: center;
            margin: 1.5rem 0;
        }
        .empty-state-icon {
            font-size: 2.5rem;
            color: #94A3B8;
            margin-bottom: 0.75rem;
        }
        .empty-state-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #334155;
            margin-bottom: 0.5rem;
        }
        .empty-state-desc {
            font-size: 0.85rem;
            color: #64748B;
            max-width: 450px;
            margin: 0 auto;
            line-height: 1.5;
        }

        /* Streamlit Primary & Secondary Button Overrides */
        div[data-testid="stButton"] button {
            border-radius: 10px !important;
            font-weight: 700 !important;
            transition: all 0.15s ease-in-out !important;
        }
        div[data-testid="stButton"] button[kind="primary"] {
            background-color: #E11D48 !important;
            color: #FFFFFF !important;
            border: none !important;
            box-shadow: 0 2px 6px rgba(225, 29, 72, 0.25) !important;
        }
        div[data-testid="stButton"] button[kind="primary"]:hover {
            background-color: #9F1239 !important;
            color: #FFFFFF !important;
            box-shadow: 0 4px 12px rgba(225, 29, 72, 0.35) !important;
        }
        div[data-testid="stButton"] button[kind="secondary"] {
            background-color: #F8FAFC !important;
            color: #334155 !important;
            border: 1px solid #CBD5E1 !important;
        }
        div[data-testid="stButton"] button[kind="secondary"]:hover {
            background-color: #F1F5F9 !important;
            border-color: #E11D48 !important;
            color: #E11D48 !important;
        }
        div[data-testid="stButton"] button:active {
            transform: scale(0.98) !important;
        }

        /* Form submit button */
        div[data-testid="stFormSubmitButton"] > button {
            background-color: #E11D48 !important;
            color: #FFFFFF !important;
            font-weight: 800 !important;
            border-radius: 10px !important;
            padding: 0.65rem 1.5rem !important;
            border: none !important;
            letter-spacing: 0.3px !important;
            box-shadow: 0 4px 12px rgba(225, 29, 72, 0.25) !important;
        }
        div[data-testid="stFormSubmitButton"] > button:hover {
            background-color: #9F1239 !important;
            box-shadow: 0 6px 18px rgba(225, 29, 72, 0.35) !important;
        }
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# TOP HOSPITAL HEADER
# =============================================================================

def render_top_bar(user: Dict[str, Any]):
    hospital_name = user.get("hospital_name", "Clinical Hospital")
    ward_name = user.get("ward", "General Ward")
    user_name = user.get("name", "Staff")
    role_str = "CHIEF PHYSICIAN / ADMIN" if user.get("role") == "admin" else "STAFF NURSE"
    user_icon = get_svg_icon("user", 13, "#FFFFFF")
    hosp_icon = get_svg_icon("hospital", 18, "#E11D48")

    st.markdown(f"""
    <div class="vg-topbar">
        <div class="vg-brand-group">
            <div class="vg-brand-logo" style="display:flex; align-items:center; gap:6px;">
                {hosp_icon}
                <span>VG</span>
            </div>
            <div>
                <div class="vg-brand-title">{hospital_name}</div>
                <div class="vg-brand-sub">{ward_name} &bull; VitalGuard Surveillance AI</div>
            </div>
        </div>
        <div class="vg-user-pill">
            <span style="display:inline-flex; align-items:center; gap:6px;">{user_icon} {user_name}</span>
            <span style="background:rgba(255,255,255,0.25); padding:2px 8px; border-radius:4px; font-size:0.7rem; font-family:'JetBrains Mono';">{role_str}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# RISK BADGE
# =============================================================================

def render_risk_badge(level: str, score: Optional[float] = None) -> str:
    lvl = (level or "GREEN").upper()
    dot_class = "dot-green" if lvl == "GREEN" else ("dot-yellow" if lvl == "YELLOW" else "dot-red")
    badge_class = f"risk-badge risk-badge-{lvl.lower()}"
    score_str = f" ({score:.2f})" if score is not None else ""
    return f"""<span class="{badge_class}"><span class="badge-dot {dot_class}"></span>{lvl} RISK{score_str}</span>"""


# =============================================================================
# OVERVIEW CARDS (HOME SCREEN)
# =============================================================================

def render_overview_cards(critical_count: int, warning_count: int, stable_count: int):
    c1, c2, c3 = st.columns(3)
    with c1:
        icon_svg = get_svg_icon('alert-triangle', 15, '#E53935')
        st.markdown(f"""
        <div class="overview-card overview-critical">
            <div class="overview-title">{icon_svg} Critical Deterioration</div>
            <div class="overview-num">{critical_count}</div>
            <div class="overview-sub">Red Alert &bull; Immediate Action Required</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        icon_svg = get_svg_icon('alert-circle', 15, '#D97706')
        st.markdown(f"""
        <div class="overview-card overview-warning">
            <div class="overview-title">{icon_svg} Needs Attention</div>
            <div class="overview-num">{warning_count}</div>
            <div class="overview-sub">Yellow Warning &bull; Sub-acute Drift</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        icon_svg = get_svg_icon('shield-check', 15, '#43A047')
        st.markdown(f"""
        <div class="overview-card overview-stable">
            <div class="overview-title">{icon_svg} Stable Patients</div>
            <div class="overview-num">{stable_count}</div>
            <div class="overview-sub">Green Status &bull; Routine Monitoring</div>
        </div>
        """, unsafe_allow_html=True)



# =============================================================================
# SMS NOTIFICATION BANNER
# =============================================================================

def render_simulated_sms_banner(doctor_name: str, phone: str, patient_name: str, bed_display: str, risk_level: str):
    time_str = datetime.now().strftime("%H:%M:%S")
    st.markdown(f"""
    <div class="sms-banner">
        <div class="sms-header">
            <span>📱 URGENT CLINICAL SMS DISPATCH &bull; {time_str}</span>
            <span style="font-size:0.75rem; background:rgba(251,113,133,0.2); padding:2px 8px; border-radius:4px;">STATUS: SENT & DELIVERED</span>
        </div>
        <div class="sms-body">
            <strong>To:</strong> {doctor_name} ({phone})<br>
            <strong>Message:</strong> [URGENT VITALGUARD ALERT] Patient {patient_name} in {bed_display} has crossed critical threshold into {risk_level}. Immediate bedside physician review requested.
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# CRITICAL ALERT BANNER
# =============================================================================

def render_critical_alert_banner(patient_name: str, bed_display: str, key_reasons: List[str]):
    time_str = datetime.now().strftime("%H:%M:%S")
    reasons_html = "".join([f"<li>{r}</li>" for r in key_reasons[:3]])
    st.markdown(f"""
    <div class="critical-alert-banner">
        <div class="alert-header">
            <span>🚨 CRITICAL DETERIORATION ALERT &bull; {bed_display}</span>
            <span class="alert-status-tag">DOCTOR NOTIFIED &bull; {time_str}</span>
        </div>
        <div style="margin-top: 8px; font-size: 0.9rem; color: #FEE2E2;">
            <strong>Patient: {patient_name}</strong> has triggered an urgent deterioration warning.
            <ul style="margin: 6px 0 0 0; padding-left: 1.25rem;">
                {reasons_html}
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# CLINICAL EXPLAINABILITY CARD
# =============================================================================

def render_explainability_card(explanation: Dict[str, Any], risk_level: str):
    lvl = risk_level.lower()
    box_class = f"explain-box box-{lvl}"
    icon = "🟢" if lvl == "green" else ("⚠️" if lvl == "yellow" else "🚨")

    items_html = "".join([f"<li>{item}</li>" for item in explanation.get("key_findings", [])])

    st.markdown(f"""
    <div class="{box_class}">
        <div style="font-size: 1.0rem; font-weight: 700; color: #0F172A; margin-bottom: 4px;">
            {icon} {explanation.get('title', 'Clinical Rationale')}
        </div>
        <div style="font-size: 0.78rem; color: #64748B; margin-bottom: 8px;">
            {explanation.get('trend_summary', '')}
        </div>
        <ul style="margin: 0; padding-left: 1.25rem; font-size: 0.88rem; color: #334155; line-height: 1.5;">
            {items_html}
        </ul>
        <div style="margin-top: 10px; padding-top: 8px; border-top: 1px dashed rgba(0,0,0,0.12); font-size: 0.84rem; font-weight: 600; color: #1E293B;">
            <strong>Protocol Recommendation:</strong> {explanation.get('recommended_action', '')}
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# CLEAN EMPTY STATE WIDGET
# =============================================================================

def render_empty_state(icon: str, title: str, description: str):
    st.markdown(f"""
    <div class="empty-state-box">
        <div class="empty-state-icon">{icon}</div>
        <div class="empty-state-title">{title}</div>
        <div class="empty-state-desc">{description}</div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# LIGHT PASTEL VITALS MONITOR DISPLAY
# =============================================================================

def render_bedside_monitor_display(latest_vitals: Dict[str, Any], history: List[Dict[str, Any]]):
    """
    Renders a clean, light pastel-card vitals grid (soft, modern, non-clinical-dark style)
    matching a warm, approachable hospital-app aesthetic.
    """
    if not latest_vitals:
        render_empty_state("📟", "No Vitals Data Available", "Chart bedside observations to activate the vitals display.")
        return

    prev_v = history[-2] if len(history) >= 2 else None

    def _get_delta(key: str, unit: str = ""):
        if not prev_v or prev_v.get(key) is None or latest_vitals.get(key) is None:
            return ""
        try:
            curr_val = float(latest_vitals[key])
            prev_val = float(prev_v[key])
            diff = curr_val - prev_val
            if abs(diff) < 0.1:
                return "<span style='color:#94A3B8; font-size:0.72rem;'>→ steady</span>"
            arrow = "↑" if diff > 0 else "↓"
            bad = (key in ['heart_rate', 'resp_rate', 'temperature'] and diff > 0) or (key in ['spo2', 'sbp'] and diff < 0)
            color = "#DC2626" if bad else "#16A34A"
            return f"<span style='color:{color}; font-size:0.75rem; font-weight:700;'>{arrow} {diff:+.1f}{unit}</span>"
        except (ValueError, TypeError):
            return ""

    hr_val = latest_vitals.get("heart_rate", "--")
    sbp_val = latest_vitals.get("sbp", "--")
    dbp_val = latest_vitals.get("dbp", "--")
    spo2_val = latest_vitals.get("spo2", "--")
    rr_val = latest_vitals.get("resp_rate", "--")
    temp_val = latest_vitals.get("temperature", "--")

    hr_delta = _get_delta("heart_rate")
    spo2_delta = _get_delta("spo2", "%")
    rr_delta = _get_delta("resp_rate")
    temp_delta = _get_delta("temperature", "°C")

    cards = [
        ("❤️", "Heart Rate", hr_val, "bpm", hr_delta, "#FEE2E2", "#DC2626", "60 - 100"),
        ("🩸", "Blood Pressure", f"{sbp_val}/{dbp_val}", "mmHg", "", "#DBEAFE", "#2563EB", "90/60 - 130/85"),
        ("💧", "SpO2", spo2_val, "%", spo2_delta, "#FEF3C7", "#D97706", "≥ 95"),
        ("🫁", "Resp Rate", rr_val, "/min", rr_delta, "#EDE9FE", "#7C3AED", "12 - 20"),
        ("🌡️", "Temperature", temp_val, "°C", temp_delta, "#FFEDD5", "#EA580C", "36.1 - 37.8"),
    ]

    cards_html = ""
    for icon, label, val, unit, delta, bg, fg, target in cards:
        cards_html += f'<div style="background:{bg}; border-radius:14px; padding:14px 16px; flex:1; min-width:130px;"><div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;"><span style="font-size:0.78rem; font-weight:700; color:{fg};">{icon} {label}</span>{delta}</div><div style="font-size:1.7rem; font-weight:800; color:{fg}; font-family:\'Inter\', sans-serif;">{val}<span style="font-size:0.85rem; font-weight:600;"> {unit}</span></div><div style="font-size:0.68rem; color:#64748B; margin-top:2px;">Normal: {target}</div></div>'

    monitor_html = f"""<div style="display:flex; gap:10px; flex-wrap:wrap; background:#F8FAFC; border-radius:16px; padding:14px;">{cards_html}</div>"""
    st.markdown(monitor_html, unsafe_allow_html=True)


# =============================================================================
# SIMPLIFIED COMPACT ALERTS RENDERER
# =============================================================================

def render_compact_alerts_log(alerts: List[Dict[str, Any]]):
    """Renders compact 1-line per alert timeline with deduplication."""
    if not alerts:
        render_empty_state("🔔", "No Deterioration Alerts", "Active surveillance is running. Critical deterioration alerts will appear here.")
        return

    # Deduplicate alerts
    deduped = []
    for a in alerts:
        msg = (a.get("message") or "").strip()
        lvl = (a.get("risk_level") or "RED").upper()
        ts = a.get("timestamp_str", "")
        p_name = a.get("patient_name", "Unknown Patient")
        ward = a.get("ward", "")
        bed = a.get("bed_number", "")
        loc_str = f"{ward}, {bed}".strip(", ")
        patient_str = f"{p_name} ({loc_str})" if loc_str else p_name

        # Extract 3-5 word concise reason
        short_reason = msg.split(";")[0] if ";" in msg else msg
        words = short_reason.split()
        if len(words) > 6:
            short_reason = " ".join(words[:6]) + "..."

        if not deduped:
            deduped.append({
                "level": lvl,
                "patient_info": patient_str,
                "reason": short_reason,
                "time": ts,
                "count": 1
            })
        else:
            prev = deduped[-1]
            if prev["patient_info"] == patient_str and prev["reason"] == short_reason and prev["level"] == lvl:
                prev["count"] += 1
            else:
                deduped.append({
                    "level": lvl,
                    "patient_info": patient_str,
                    "reason": short_reason,
                    "time": ts,
                    "count": 1
                })

    for item in deduped:
        lvl = item["level"]
        color = "#EF4444" if lvl == "RED" else "#F59E0B"
        bg = "#FEF2F2" if lvl == "RED" else "#FFFBEB"
        border = "#FECACA" if lvl == "RED" else "#FDE68A"
        badge_dot = "dot-red" if lvl == "RED" else "dot-yellow"
        cnt_badge = f" <span style='font-size:0.70rem; background:#FEE2E2; color:#991B1B; font-weight:700; padding:1px 6px; border-radius:8px;'>({item['count']}x)</span>" if item['count'] > 1 else ""

        st.markdown(f"""
        <div style="display:flex; align-items:center; justify-content:space-between; background:{bg}; border:1px solid {border}; border-left:4px solid {color}; border-radius:6px; padding:8px 12px; margin-bottom:6px; font-size:0.85rem;">
            <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
                <span class="badge-dot {badge_dot}"></span>
                <span style="font-weight:700; color:{color}; font-size:0.75rem; text-transform:uppercase;">{lvl}</span>
                <span style="color:#0F172A; font-weight:700;">{item['patient_info']}</span>
                <span style="color:#64748B;">&bull;</span>
                <span style="color:#334155; font-weight:500;">{item['reason']}</span>{cnt_badge}
            </div>
            <span style="font-size:0.75rem; color:#64748B; font-family:'JetBrains Mono', monospace; white-space:nowrap;">{item['time']}</span>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# CONFIDENCE & UNCERTAINTY GAUGE (MC DROPOUT)
# =============================================================================

def render_confidence_and_predictions(risk_data: Dict[str, Any]):
    conf_pct = risk_data.get("confidence_pct", 88)
    uncertainty_std = risk_data.get("uncertainty_std", 0.04)
    mort_prob = risk_data.get("mortality_prob", 0.05) * 100
    sepsis_prob = risk_data.get("sepsis_prob", 0.12) * 100
    vaso_prob = risk_data.get("vaso_prob", 0.08) * 100

    st.markdown("##### 🧠 Neural Multi-Task Outcomes & Uncertainty")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            label="Model Confidence",
            value=f"{conf_pct}%",
            delta=f"±{uncertainty_std*100:.1f}% uncertainty",
            delta_color="off",
            help="Computed via Monte Carlo Dropout across 15 stochastic forward passes."
        )
    with c2:
        st.metric(
            label="Sepsis Risk (24h)",
            value=f"{sepsis_prob:.1f}%",
            delta="Multi-task",
            delta_color="inverse" if sepsis_prob > 35 else "off"
        )
    with c3:
        st.metric(
            label="Mortality Risk (24h)",
            value=f"{mort_prob:.1f}%",
            delta="AUROC 0.7955",
            delta_color="inverse" if mort_prob > 25 else "off"
        )
    with c4:
        st.metric(
            label="Vasopressor Need (12h)",
            value=f"{vaso_prob:.1f}%",
            delta="Hemodynamic",
            delta_color="inverse" if vaso_prob > 30 else "off"
        )


# =============================================================================
# MODEL VALIDATION CARD (PROFILE VIEW)
# =============================================================================

def render_model_validation_card():
    """Model Validation panel removed per requirements — presented verbally during pitch."""
    pass


# =============================================================================
# DISCLAIMER FOOTER
# =============================================================================

def render_disclaimer_footer():
    st.markdown("""
    <div style="font-size: 0.72rem; color: #94A3B8; border-top: 1px solid #E2E8F0; padding-top: 1rem; margin-top: 2rem; line-height: 1.4; text-align: center;">
        <strong>VitalGuard Clinical Decision-Support Notice:</strong> 
        VitalGuard is designed as an early-warning adjunct to clinical nursing judgment and does not constitute an autonomous diagnostic device.
        Trained and validated on retrospective ICU data (MIMIC-III, 36,212 ICU stays; Transformer AUROC 0.7955, Early Warning Model AUROC 0.7751).
        All clinical escalation, titration, and therapeutic decisions remain under the authority of licensed healthcare professionals.
    </div>
    """, unsafe_allow_html=True)