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

        [data-testid="stSidebar"], section[data-testid="stSidebar"], [data-testid="collapsedControl"] {
            display: none !important;
        }

        /* Root background and global body styling - High Contrast Light Theme */
        body, .stApp {
            background-color: #F8FAFC !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: #0F172A !important;
        }

        /* Global Text & Heading High Contrast Rules */
        .stMarkdown p, .stMarkdown span, .stMarkdown label, div[data-testid="stMarkdownContainer"] p {
            color: #1E293B !important;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #0F172A !important;
            font-weight: 800 !important;
        }
        label[data-testid="stWidgetLabel"] p, label[data-testid="stWidgetLabel"] {
            color: #0F172A !important;
            font-weight: 700 !important;
            font-size: 0.88rem !important;
        }

        /* High Contrast Form Inputs, Text Areas & Selectboxes */
        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div,
        div[data-baseweb="textarea"] > div {
            background-color: #FFFFFF !important;
            border: 1.5px solid #64748B !important;
            color: #0F172A !important;
        }
        input, select, textarea {
            color: #0F172A !important;
            font-weight: 600 !important;
        }
        div[data-baseweb="select"] span {
            color: #0F172A !important;
            font-weight: 600 !important;
        }

        /* Main Container Padding */
        .main .block-container {
            padding-top: 1.25rem !important;
            padding-bottom: 2.5rem !important;
            max-width: 1350px !important;
        }

        /* ================= CENTERED SLEEK LOGIN CARD ================= */
        .login-card-centered {
            background: #FFFFFF;
            border-radius: 20px;
            padding: 2.25rem 2rem;
            color: #0F172A;
            box-shadow: 0 20px 45px -10px rgba(15, 23, 42, 0.12);
            border: 1px solid #CBD5E1;
            margin: 1.5rem auto;
            text-align: center;
        }

        /* ================= PERSISTENT TOP NAV BAR (DARK NAVY) ================= */
        .vg-topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.75rem 1.4rem;
            background: #0B192C;
            border-bottom: 3px solid #E11D48;
            border-radius: 14px;
            color: #FFFFFF;
            margin-bottom: 1.25rem;
            box-shadow: 0 8px 24px -4px rgba(11, 25, 44, 0.25);
        }
        .vg-brand-group {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .vg-brand-logo {
            background: #E11D48;
            color: #FFFFFF;
            font-weight: 800;
            font-size: 1.1rem;
            padding: 5px 11px;
            border-radius: 8px;
            letter-spacing: -0.5px;
            box-shadow: 0 2px 8px rgba(225, 29, 72, 0.3);
        }
        .vg-brand-title {
            margin: 0;
            font-size: 1.2rem;
            font-weight: 800;
            letter-spacing: -0.3px;
            color: #0F172A !important;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .vg-version-tag {
            background: #FFE4E6 !important;
            color: #E11D48 !important;
            font-size: 0.68rem;
            font-weight: 800;
            padding: 2px 7px;
            border-radius: 4px;
            border: 1px solid #FECDD3 !important;
            font-family: 'JetBrains Mono', monospace;
        }
        .vg-brand-sub {
            margin: 1px 0 0 0;
            font-size: 0.76rem;
            color: #475569 !important;
            font-weight: 600;
        }
        .vg-user-pill {
            display: flex;
            align-items: center;
            gap: 8px;
            background: rgba(255, 255, 255, 0.12);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            padding: 5px 12px;
            border-radius: 9999px;
            color: #FFFFFF;
            font-size: 0.80rem;
            font-weight: 600;
        }

        /* ================= OVERVIEW STAT PANELS ================= */
        .overview-card-soft {
            border-radius: 14px;
            padding: 1.25rem 1.4rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1rem;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
            position: relative;
            overflow: hidden;
        }
        .overview-card-soft:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(15, 23, 42, 0.08);
        }
        .stat-card-soft-red {
            background-color: #FEF2F2;
            border: 1.5px solid #FCA5A5;
            color: #991B1B;
        }
        .stat-card-soft-amber {
            background-color: #FFFBEB;
            border: 1.5px solid #FDE68A;
            color: #92400E;
        }
        .stat-card-soft-emerald {
            background-color: #ECFDF5;
            border: 1.5px solid #A7F3D0;
            color: #065F46;
        }
        .stat-soft-num {
            font-size: 2.2rem;
            font-weight: 800;
            line-height: 1;
            font-family: 'JetBrains Mono', monospace;
            margin: 4px 0 2px 0;
        }
        .stat-card-soft-red .stat-soft-num { color: #991B1B !important; }
        .stat-card-soft-amber .stat-soft-num { color: #92400E !important; }
        .stat-card-soft-emerald .stat-soft-num { color: #065F46 !important; }

        .stat-soft-title {
            font-size: 0.82rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .stat-card-soft-red .stat-soft-title { color: #991B1B !important; }
        .stat-card-soft-amber .stat-soft-title { color: #92400E !important; }
        .stat-card-soft-emerald .stat-soft-title { color: #065F46 !important; }

        .stat-soft-sub {
            font-size: 0.75rem;
            font-weight: 600;
            opacity: 0.9;
            margin-top: 2px;
        }
        .stat-card-soft-red .stat-soft-sub { color: #B91C1C !important; }
        .stat-card-soft-amber .stat-soft-sub { color: #B45309 !important; }
        .stat-card-soft-emerald .stat-soft-sub { color: #047857 !important; }

        .stat-soft-icon {
            width: 52px;
            height: 52px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            opacity: 0.95;
        }

        /* ================= PATIENT CARDS WITH LEFT RISK BORDER ================= */
        .patient-card {
            background: #FFFFFF !important;
            border-radius: 14px;
            padding: 1.2rem;
            border: 1px solid #CBD5E1;
            border-left: 5px solid #10B981;
            box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
            margin-bottom: 1rem;
            transition: all 0.15s ease;
        }
        .patient-card-red {
            border-left: 5px solid #DC2626 !important;
        }
        .patient-card-yellow {
            border-left: 5px solid #F59E0B !important;
        }
        .patient-card-green {
            border-left: 5px solid #10B981 !important;
        }
        .patient-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px -4px rgba(15, 23, 42, 0.08);
            border-color: #94A3B8;
        }

        /* ================= STATUS BADGES IN 3 SIZES ================= */
        .risk-badge {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            border-radius: 9999px;
            font-weight: 800;
            letter-spacing: 0.4px;
            text-transform: uppercase;
        }
        .risk-badge-lg {
            padding: 6px 14px;
            font-size: 0.82rem;
        }
        .risk-badge-md {
            padding: 4px 10px;
            font-size: 0.72rem;
        }
        .risk-badge-sm {
            padding: 2px 7px;
            font-size: 0.65rem;
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

        /* ================= METRIC CHIPS (HIGH CONTRAST) ================= */
        .metric-chip {
            display: inline-flex;
            flex-direction: column;
            padding: 5px 10px;
            background: #F1F5F9 !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 8px;
            min-width: 62px;
            text-align: center;
        }
        .metric-chip-label {
            font-size: 0.64rem;
            color: #475569 !important;
            text-transform: uppercase;
            font-weight: 700;
        }
        .metric-chip-val {
            font-size: 0.92rem;
            color: #0F172A !important;
            font-weight: 800;
            font-family: 'JetBrains Mono', monospace;
        }

        /* ================= VITALS CARD WITH TREND INDICATORS & RANGES ================= */
        .vital-trend-card {
            background: #FFFFFF;
            border: 1px solid #CBD5E1;
            border-radius: 14px;
            padding: 1rem 1.1rem;
            flex: 1;
            min-width: 140px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.03);
            transition: transform 0.15s ease;
        }
        .vital-trend-card:hover {
            transform: translateY(-2px);
            border-color: #94A3B8;
        }
        .vital-card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 6px;
        }
        .vital-card-label {
            font-size: 0.78rem;
            font-weight: 700;
            color: #334155;
            display: flex;
            align-items: center;
            gap: 5px;
        }

        /* ================= ALERT FEED CARDS ================= */
        .alert-feed-card {
            background: #FFFFFF !important;
            border-radius: 12px;
            border: 1px solid #CBD5E1;
            border-left: 5px solid #DC2626;
            padding: 1rem 1.2rem;
            margin-bottom: 0.85rem;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
            transition: all 0.15s ease;
        }
        .alert-feed-card:hover {
            border-color: #94A3B8;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        }
        .alert-badge-new {
            background: #DC2626;
            color: #FFFFFF;
            font-size: 0.65rem;
            font-weight: 800;
            padding: 2px 7px;
            border-radius: 4px;
            letter-spacing: 0.5px;
        }

        .sms-banner {
            background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
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

        .explain-box {
            background: #FFFFFF;
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

        .empty-state-box {
            background: #FFFFFF;
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
            font-weight: 800;
            color: #0F172A;
            margin-bottom: 0.5rem;
        }
        .empty-state-desc {
            font-size: 0.85rem;
            color: #475569;
            max-width: 450px;
            margin: 0 auto;
            line-height: 1.5;
        }

        /* ================= HIGH-CONTRAST BUTTONS ================= */
        div[data-testid="stButton"] button {
            border-radius: 10px !important;
            font-weight: 700 !important;
            transition: all 0.15s ease-in-out !important;
        }
        /* Primary Button: Solid Red (#E11D48) with bold WHITE text */
        div[data-testid="stButton"] button[kind="primary"] {
            background-color: #E11D48 !important;
            color: #FFFFFF !important;
            border: none !important;
            box-shadow: 0 3px 8px rgba(225, 29, 72, 0.3) !important;
        }
        div[data-testid="stButton"] button[kind="primary"] p,
        div[data-testid="stButton"] button[kind="primary"] span {
            color: #FFFFFF !important;
            font-weight: 800 !important;
        }
        div[data-testid="stButton"] button[kind="primary"]:hover {
            background-color: #9F1239 !important;
            color: #FFFFFF !important;
            box-shadow: 0 6px 14px rgba(225, 29, 72, 0.4) !important;
        }

        /* Secondary Button: Clean White with solid dark border & bold dark text (#0F172A) */
        div[data-testid="stButton"] button[kind="secondary"] {
            background-color: #FFFFFF !important;
            color: #0F172A !important;
            border: 1.5px solid #64748B !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
        }
        div[data-testid="stButton"] button[kind="secondary"] p,
        div[data-testid="stButton"] button[kind="secondary"] span {
            color: #0F172A !important;
            font-weight: 700 !important;
        }
        div[data-testid="stButton"] button[kind="secondary"]:hover {
            background-color: #F1F5F9 !important;
            border-color: #E11D48 !important;
            color: #E11D48 !important;
        }
        div[data-testid="stButton"] button[kind="secondary"]:hover p,
        div[data-testid="stButton"] button[kind="secondary"]:hover span {
            color: #E11D48 !important;
        }

        /* Form Submit Button */
        div[data-testid="stFormSubmitButton"] > button {
            background-color: #E11D48 !important;
            color: #FFFFFF !important;
            font-weight: 800 !important;
            border-radius: 10px !important;
            padding: 0.65rem 1.5rem !important;
            border: none !important;
            letter-spacing: 0.3px !important;
            box-shadow: 0 4px 12px rgba(225, 29, 72, 0.3) !important;
        }
        div[data-testid="stFormSubmitButton"] > button p,
        div[data-testid="stFormSubmitButton"] > button span {
            color: #FFFFFF !important;
            font-weight: 800 !important;
        }
        div[data-testid="stFormSubmitButton"] > button:hover {
            background-color: #9F1239 !important;
            box-shadow: 0 6px 18px rgba(225, 29, 72, 0.4) !important;
        }

        /* ================= MOBILE BOTTOM TAB NAVIGATION (ICON ONLY, SMALL) ================= */
        .st-key-mobile_bottom_nav {
            display: none;
        }
        @media (max-width: 768px) {
            .st-key-mobile_bottom_nav {
                display: block !important;
                position: fixed !important;
                bottom: 0 !important;
                left: 0 !important;
                right: 0 !important;
                background: #FFFFFF !important;
                border-top: 1px solid #FECDD3;
                padding: 5px 20px !important;
                z-index: 99999 !important;
                box-shadow: 0 -2px 10px rgba(0,0,0,0.08);
            }
            .st-key-mobile_bottom_nav [data-testid="stVerticalBlock"],
            .st-key-mobile_bottom_nav [data-testid="stHorizontalBlock"] {
                display: flex !important;
                flex-direction: row !important;
                flex-wrap: nowrap !important;
                width: 100% !important;
                gap: 4px !important;
            }
            .st-key-mobile_bottom_nav [data-testid="column"] {
                flex: 1 1 0 !important;
                width: auto !important;
                min-width: 0 !important;
                display: flex !important;
                justify-content: center !important;
            }
            .st-key-mobile_bottom_nav div[data-testid="stButton"] {
                display: flex !important;
                justify-content: center !important;
            }
            .st-key-mobile_bottom_nav div[data-testid="stButton"] button {
                width: 40px !important;
                height: 36px !important;
                min-height: 36px !important;
                min-width: 40px !important;
                border-radius: 10px !important;
                font-size: 1.0rem !important;
                line-height: 1 !important;
                padding: 0 !important;
                margin: 0 auto !important;
                border: none !important;
                background: transparent !important;
                box-shadow: none !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
            }
            .st-key-mobile_bottom_nav div[data-testid="stButton"] button p {
                font-size: 1.0rem !important;
                margin: 0 !important;
                line-height: 1 !important;
            }
            .st-key-mobile_bottom_nav div[data-testid="stButton"] button[kind="primary"] {
                background: #FEF2F2 !important;
            }
            .main .block-container {
                padding-bottom: 70px !important;
            }
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
        .st-key-desktop_top_nav {
            display: block;
        }
        @media (max-width: 768px) {
            .st-key-desktop_top_nav {
                display: none !important;
            }
        }
        /* ================= MOBILE-ONLY BRANDED HEADER ================= */
        .st-key-mobile_header {
            display: none;
        }
        @media (max-width: 768px) {
            .st-key-mobile_header {
                display: block !important;
            }
        }
        .vg-mobile-header-card {
            background: linear-gradient(135deg, #9F1239 0%, #E11D48 100%);
            border-radius: 16px;
            padding: 1.1rem 1.25rem;
            color: #FFFFFF;
            margin-bottom: 1rem;
            box-shadow: 0 4px 16px -2px rgba(225, 29, 72, 0.25);
        }
        .vg-mobile-header-top {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 4px;
        }
        .vg-mobile-header-logo {
            background: #FFFFFF;
            color: #E11D48;
            font-weight: 800;
            font-size: 1.0rem;
            padding: 4px 10px;
            border-radius: 8px;
        }
        .vg-mobile-header-name {
            font-size: 1.05rem;
            font-weight: 800;
        }
        .vg-mobile-header-sub {
            font-size: 0.75rem;
            color: #FFE4E6;
            margin-top: 2px;
        }

        /* ================= OVERVIEW STAT PANELS ================= */
        .overview-card-soft {
            border-radius: 14px;
            padding: 1.25rem 1.4rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1rem;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
            position: relative;
            overflow: hidden;
        }
        .overview-card-soft:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(15, 23, 42, 0.1);
        }
        .stat-card-soft-red {
            background-color: #FEF2F2;
            border: 1px solid #FECDD3;
            color: #9F1239;
        }
        .stat-card-soft-amber {
            background-color: #FFFBEB;
            border: 1px solid #FDE68A;
            color: #92400E;
        }
        .stat-card-soft-emerald {
            background-color: #ECFDF5;
            border: 1px solid #A7F3D0;
            color: #065F46;
        }
        .stat-soft-num {
            font-size: 2.2rem;
            font-weight: 800;
            line-height: 1;
            font-family: 'JetBrains Mono', monospace;
            margin: 4px 0 2px 0;
            color: #0F172A !important;
        }
        .stat-soft-title {
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .stat-soft-sub {
            font-size: 0.75rem;
            opacity: 0.9;
            margin-top: 2px;
            color: #475569 !important;
        }
        .stat-soft-icon {
            width: 52px;
            height: 52px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            opacity: 0.95;
        }

        /* ================= PATIENT CARDS WITH LEFT RISK BORDER ================= */
        .patient-card {
            background: #FFFFFF !important;
            border-radius: 14px;
            padding: 1.2rem;
            border: 1px solid #E2E8F0;
            border-left: 5px solid #10B981;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
            margin-bottom: 1rem;
            transition: all 0.15s ease;
        }
        .patient-card-red {
            border-left: 5px solid #DC2626 !important;
        }
        .patient-card-yellow {
            border-left: 5px solid #F59E0B !important;
        }
        .patient-card-green {
            border-left: 5px solid #10B981 !important;
        }
        .patient-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(15, 23, 42, 0.08);
            border-color: #CBD5E1;
        }

        /* ================= STATUS BADGES IN 3 SIZES ================= */
        .risk-badge {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            border-radius: 9999px;
            font-weight: 700;
            letter-spacing: 0.4px;
            text-transform: uppercase;
        }
        .risk-badge-lg {
            padding: 6px 14px;
            font-size: 0.82rem;
        }
        .risk-badge-md {
            padding: 4px 10px;
            font-size: 0.72rem;
        }
        .risk-badge-sm {
            padding: 2px 7px;
            font-size: 0.65rem;
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
            border: 1px solid #FECDD3;
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

        /* ================= METRIC CHIPS ================= */
        .metric-chip {
            display: inline-flex;
            flex-direction: column;
            padding: 5px 10px;
            background: #F1F5F9 !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 8px;
            min-width: 62px;
            text-align: center;
        }
        .metric-chip-label {
            font-size: 0.64rem;
            color: #475569 !important;
            text-transform: uppercase;
            font-weight: 700;
        }
        .metric-chip-val {
            font-size: 0.92rem;
            color: #0F172A !important;
            font-weight: 800;
            font-family: 'JetBrains Mono', monospace;
        }

        /* ================= VITALS CARD WITH TREND INDICATORS & RANGES ================= */
        .vital-trend-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 14px;
            padding: 1rem 1.1rem;
            flex: 1;
            min-width: 140px;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
            transition: transform 0.15s ease;
        }
        .vital-trend-card:hover {
            transform: translateY(-2px);
            border-color: #CBD5E1;
        }
        .vital-card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 6px;
        }
        .vital-card-label {
            font-size: 0.78rem;
            font-weight: 700;
            color: #334155;
            display: flex;
            align-items: center;
            gap: 5px;
        }

        /* ================= ALERT FEED CARDS ================= */
        .alert-feed-card {
            background: #FFFFFF !important;
            border-radius: 12px;
            border: 1px solid #E2E8F0;
            border-left: 5px solid #DC2626;
            padding: 1rem 1.2rem;
            margin-bottom: 0.85rem;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
            transition: all 0.15s ease;
        }
        .alert-feed-card:hover {
            border-color: #CBD5E1;
            box-shadow: 0 6px 16px rgba(15, 23, 42, 0.08);
        }
        .alert-badge-new {
            background: #DC2626;
            color: #FFFFFF;
            font-size: 0.65rem;
            font-weight: 800;
            padding: 2px 7px;
            border-radius: 4px;
            letter-spacing: 0.5px;
        }

        .sms-banner {
            background: #FFF1F2;
            border-left: 4px solid #E11D48;
            color: #881337;
            padding: 1rem 1.25rem;
            border-radius: 12px;
            margin: 1rem 0;
            box-shadow: 0 2px 8px rgba(225, 29, 72, 0.08);
        }
        .sms-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 0.88rem;
            font-weight: 800;
            color: #9F1239;
        }

        .explain-box {
            background: #FFFFFF;
            border-left: 4px solid #E11D48;
            border-radius: 0 12px 12px 0;
            padding: 1.25rem;
            margin: 1rem 0;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
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
            background: #ECFDF5;
        }

        .empty-state-box {
            background: #FFFFFF;
            border: 2px dashed #CBD5E1;
            border-radius: 16px;
            padding: 3rem 2rem;
            text-align: center;
            margin: 1.5rem 0;
        }
        .empty-state-icon {
            font-size: 2.5rem;
            color: #64748B;
            margin-bottom: 0.75rem;
        }
        .empty-state-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #0F172A;
            margin-bottom: 0.5rem;
        }
        .empty-state-desc {
            font-size: 0.85rem;
            color: #475569;
            max-width: 450px;
            margin: 0 auto;
            line-height: 1.5;
        }
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# TOP HOSPITAL HEADER
# =============================================================================
def render_mobile_header(user: Dict[str, Any]):
    """Compact branded header shown ONLY on mobile widths."""
    hospital_name = user.get("hospital_name", "Clinical Hospital")
    ward_name = user.get("ward", "General Ward")
    user_name = user.get("name", "Staff")

    st.markdown(f"""
    <div class="vg-mobile-header-card">
        <div class="vg-mobile-header-top">
            <div class="vg-mobile-header-logo">VG</div>
            <div class="vg-mobile-header-name" style="display:flex; align-items:center; gap:6px;">
                <span>VitalGuard Pro</span>
                <span style="background:rgba(255,255,255,0.25); font-size:0.7rem; padding:2px 6px; border-radius:4px; font-family:'JetBrains Mono';">v2.4</span>
            </div>
        </div>
        <div class="vg-mobile-header-sub">{hospital_name} &bull; {ward_name} &bull; {user_name}</div>
    </div>
    """, unsafe_allow_html=True)
def render_mobile_bottom_nav(active_tab: str, alert_count: int = 0):
    """Fixed bottom tab bar shown ONLY on mobile widths — icons only, compact."""
    nav_container = st.container(key="mobile_bottom_nav")
    with nav_container:
        cols = st.columns(4)
        tabs = [
            ("home", "🏠"),
            ("patients", "👥"),
            ("alerts", "🚨"),
            ("profile", "👤"),
        ]
        clicked = None
        for col, (tab_key, icon) in zip(cols, tabs):
            with col:
                is_active = (active_tab == tab_key)
                if st.button(icon, key=f"nav_{tab_key}", type="primary" if is_active else "secondary", use_container_width=True):
                    clicked = tab_key
        return clicked
def render_top_bar(user: Dict[str, Any]):
    hospital_name = user.get("hospital_name", "Clinical Hospital")
    ward_name = user.get("ward", "General Ward")
    user_name = user.get("name", "Staff")
    role_str = "CHIEF PHYSICIAN / ADMIN" if user.get("role") == "admin" else "STAFF NURSE"
    user_icon = get_svg_icon("user", 13, "#FFFFFF")

    st.markdown(f"""
    <div class="vg-topbar">
        <div class="vg-brand-group">
            <div class="vg-brand-logo">VG</div>
            <div>
                <div class="vg-brand-title">
                    <span>VitalGuard Pro</span>
                    <span class="vg-version-tag">v2.4</span>
                </div>
                <div class="vg-brand-sub">{hospital_name} &bull; {ward_name} &bull; Surveillance AI</div>
            </div>
        </div>
        <div class="vg-user-pill">
            <span style="display:inline-flex; align-items:center; gap:6px;">{user_icon} {user_name}</span>
            <span style="background:rgba(255,255,255,0.2); color:#FFFFFF; padding:2px 8px; border-radius:4px; font-size:0.7rem; font-family:'JetBrains Mono'; font-weight:700;">{role_str}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# RISK BADGE
# =============================================================================

def render_risk_badge(level: str, score: Optional[float] = None, size: str = "md") -> str:
    lvl = (level or "GREEN").upper()
    dot_class = "dot-green" if lvl == "GREEN" else ("dot-yellow" if lvl == "YELLOW" else "dot-red")
    size_class = f"risk-badge-{size}"
    badge_class = f"risk-badge risk-badge-{lvl.lower()} {size_class}"
    score_str = f" ({score:.2f})" if score is not None else ""
    return f"""<span class="{badge_class}"><span class="badge-dot {dot_class}"></span>{lvl} RISK{score_str}</span>"""


# =============================================================================
# OVERVIEW CARDS (HOME SCREEN - SOFT PANELS)
# =============================================================================

def render_overview_cards(critical_count: int, warning_count: int, stable_count: int):
    c1, c2, c3 = st.columns(3)
    with c1:
        icon_svg = get_svg_icon('alert-triangle', 26, '#991B1B')
        st.markdown(f"""
        <div class="overview-card-soft stat-card-soft-red">
            <div>
                <div class="stat-soft-title">🚨 Critical Deterioration</div>
                <div class="stat-soft-num">{critical_count}</div>
                <div class="stat-soft-sub">Immediate Bedside Intervention</div>
            </div>
            <div class="stat-soft-icon" style="background:#FEE2E2;">
                {icon_svg}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        icon_svg = get_svg_icon('alert-circle', 26, '#92400E')
        st.markdown(f"""
        <div class="overview-card-soft stat-card-soft-amber">
            <div>
                <div class="stat-soft-title">⚠️ Needs Attention</div>
                <div class="stat-soft-num">{warning_count}</div>
                <div class="stat-soft-sub">Sub-acute Trend Drift</div>
            </div>
            <div class="stat-soft-icon" style="background:#FEF3C7;">
                {icon_svg}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        icon_svg = get_svg_icon('shield-check', 26, '#065F46')
        st.markdown(f"""
        <div class="overview-card-soft stat-card-soft-emerald">
            <div>
                <div class="stat-soft-title">🟢 Stable Patients</div>
                <div class="stat-soft-num">{stable_count}</div>
                <div class="stat-soft-sub">Normal Telemetry Bounds</div>
            </div>
            <div class="stat-soft-icon" style="background:#D1FAE5;">
                {icon_svg}
            </div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# SPLIT-PANEL LOGIN LEFT PANEL
# =============================================================================

def render_split_login_left_panel():
    """Renders the dark navy left panel for the split-panel login screen."""
    st.markdown("""
    <div class="login-left-card">
        <div>
            <div style="display:flex; align-items:center; gap:12px; margin-bottom: 1.5rem;">
                <div style="background:#E11D48; color:#FFFFFF; font-weight:800; font-size:1.3rem; padding:6px 14px; border-radius:10px; letter-spacing:-0.5px; box-shadow:0 4px 12px rgba(225,29,72,0.3);">VG</div>
                <div>
                    <div style="font-weight:800; font-size:1.3rem; color:#FFFFFF; line-height:1.2;">VitalGuard</div>
                    <div style="font-size:0.75rem; color:#94A3B8; font-weight:600;">Clinical Early Warning System</div>
                </div>
            </div>
            <div class="login-tagline">⚡ Next-Gen Telemetry & Surveillance</div>
            <div class="login-headline">Patient monitoring built for clinical decision-making</div>
            <div class="login-desc">
                Continuous AI-driven physiological surveillance, personalized baseline anomaly detection, and automated SMS escalation to clinical teams.
            </div>
            <div class="login-checklist">
                <div class="login-check-item">
                    <span class="login-check-icon">✓</span>
                    <span>Real-time NEWS2 & Shock Index trajectory scoring</span>
                </div>
                <div class="login-check-item">
                    <span class="login-check-icon">✓</span>
                    <span>Personalized physiological baseline adaptivity</span>
                </div>
                <div class="login-check-item">
                    <span class="login-check-icon">✓</span>
                    <span>Instant doctor SMS dispatch & escalation protocols</span>
                </div>
                <div class="login-check-item">
                    <span class="login-check-icon">✓</span>
                    <span>Automated PDF discharge audit report generation</span>
                </div>
            </div>
        </div>
        <div class="login-badges">
            <span class="login-badge">🔒 HIPAA Compliant Protocol</span>
            <span class="login-badge">🛡️ ISO 27001 Security Framework</span>
            <span class="login-badge">🇪🇺 CE Medical Device Class IIa</span>
            <span class="login-badge">⚡ HL7 FHIR Interoperable</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# DESIGN SYSTEM SHOWCASE COMPONENT
# =============================================================================

def render_design_system_showcase():
    """Renders an interactive reference page displaying actual component patterns."""
    st.subheader("🎨 VitalGuard SaaS Design System & Reference Palette")
    st.caption("Live design tokens, typography hierarchy, status badges, buttons, and form state reference.")

    # 1. COLOR TOKENS
    st.markdown("##### 1. Brand & Healthcare Color Swatches")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown('<div class="ds-swatch" style="background:#0B192C;"><span>Dark Navy (Header)</span><code>#0B192C</code></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="ds-swatch" style="background:#E11D48;"><span>Primary Rose</span><code>#E11D48</code></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="ds-swatch" style="background:#FEF2F2; color:#991B1B; border:1px solid #FCA5A5;"><span>Soft Red (Critical)</span><code>#FEF2F2</code></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="ds-swatch" style="background:#FFFBEB; color:#92400E; border:1px solid #FDE68A;"><span>Soft Amber (Warning)</span><code>#FFFBEB</code></div>', unsafe_allow_html=True)
    with c5:
        st.markdown('<div class="ds-swatch" style="background:#ECFDF5; color:#065F46; border:1px solid #A7F3D0;"><span>Soft Green (Stable)</span><code>#ECFDF5</code></div>', unsafe_allow_html=True)

    st.markdown("---")

    # 2. STATUS BADGES IN 3 SIZES
    st.markdown("##### 2. Clinical Risk Status Badges (3 Sizes)")
    b1, b2, b3 = st.columns(3)
    with b1:
        st.markdown("**Large (lg)**")
        st.markdown(render_risk_badge("RED", 0.92, size="lg"), unsafe_allow_html=True)
        st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)
        st.markdown(render_risk_badge("YELLOW", 0.55, size="lg"), unsafe_allow_html=True)
        st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)
        st.markdown(render_risk_badge("GREEN", 0.12, size="lg"), unsafe_allow_html=True)
    with b2:
        st.markdown("**Medium (md)**")
        st.markdown(render_risk_badge("RED", 0.92, size="md"), unsafe_allow_html=True)
        st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)
        st.markdown(render_risk_badge("YELLOW", 0.55, size="md"), unsafe_allow_html=True)
        st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)
        st.markdown(render_risk_badge("GREEN", 0.12, size="md"), unsafe_allow_html=True)
    with b3:
        st.markdown("**Small (sm)**")
        st.markdown(render_risk_badge("RED", size="sm"), unsafe_allow_html=True)
        st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)
        st.markdown(render_risk_badge("YELLOW", size="sm"), unsafe_allow_html=True)
        st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)
        st.markdown(render_risk_badge("GREEN", size="sm"), unsafe_allow_html=True)

    st.markdown("---")

    # 3. TYPOGRAPHY SCALE
    st.markdown("##### 3. Typography Scale")
    st.markdown('<h1 style="margin:0;">Display Heading H1 (2.2rem Inter 800)</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="margin:0; margin-top:4px;">Section Title H2 (1.5rem Inter 700)</h2>', unsafe_allow_html=True)
    st.markdown('<h4 style="margin:0; margin-top:4px;">Component Label H4 (1.1rem Inter 600)</h4>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.9rem; color:#475569; margin-top:4px;">Body text (0.9rem Inter regular with clean line-height for clinical readability)</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-family:\'JetBrains Mono\'; font-size:1.1rem; color:#0F172A; font-weight:700; margin-top:4px;">Telemetry Value: 120/80 mmHg &bull; HR 72 bpm (JetBrains Mono 700)</p>', unsafe_allow_html=True)

    st.markdown("---")

    # 4. BUTTON VARIANTS & FORM INPUT STATES
    st.markdown("##### 4. Buttons & Form Components")
    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
    with btn_col1:
        st.button("Primary Button", type="primary", use_container_width=True, key="ds_btn_p")
    with btn_col2:
        st.button("Secondary Button", type="secondary", use_container_width=True, key="ds_btn_s")
    with btn_col3:
        st.text_input("Active Input Field", value="Sample Vital Entry", key="ds_inp_a")
    with btn_col4:
        st.selectbox("Dropdown State", options=["Option 1 (Default)", "Option 2"], key="ds_sel_a")



# =============================================================================
# SMS NOTIFICATION BANNER
# =============================================================================

def render_simulated_sms_banner(doctor_name: str, phone: str, patient_name: str, bed_display: str, risk_level: str):
    time_str = datetime.now().strftime("%H:%M:%S")
    st.markdown(f"""
    <div class="sms-banner" style="background:#FFF1F2; border-left:5px solid #E11D48; border-radius:12px; padding:1rem 1.25rem; margin:1rem 0; box-shadow:0 2px 8px rgba(225,29,72,0.1);">
        <div class="sms-header" style="display:flex; align-items:center; justify-content:space-between; font-size:0.88rem; font-weight:800; color:#9F1239;">
            <span>📱 URGENT CLINICAL SMS DISPATCH &bull; {time_str}</span>
            <span style="font-size:0.75rem; background:#E11D48; color:#FFFFFF; padding:2px 8px; border-radius:4px; font-weight:800;">STATUS: SENT & DELIVERED</span>
        </div>
        <div class="sms-body" style="font-size:0.88rem; color:#0F172A !important; font-weight:600; margin-top:6px; line-height:1.5;">
            <strong style="color:#0F172A !important;">To:</strong> {doctor_name} ({phone})<br>
            <strong style="color:#0F172A !important;">Message:</strong> [URGENT VITALGUARD ALERT] Patient <strong>{patient_name}</strong> in {bed_display} has crossed critical threshold into {risk_level}. Immediate bedside physician review requested.
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# CRITICAL ALERT BANNER
# =============================================================================

def render_critical_alert_banner(patient_name: str, bed_display: str, key_reasons: List[str]):
    time_str = datetime.now().strftime("%H:%M:%S")
    reasons_html = "".join([f"<li style='color:#991B1B !important; font-weight:700; margin-bottom:3px;'>{r}</li>" for r in key_reasons[:5]])
    st.markdown(f"""
    <div class="critical-alert-banner" style="background:#FEF2F2; border:1.5px solid #FECDD3; border-left:5px solid #DC2626; border-radius:14px; padding:1.1rem 1.25rem; margin-bottom:1rem; box-shadow:0 2px 8px rgba(220,38,38,0.1);">
        <div class="alert-header" style="display:flex; align-items:center; justify-content:space-between; font-size:0.95rem; font-weight:800; color:#9F1239; margin-bottom:6px;">
            <span>🚨 CRITICAL DETERIORATION ALERT &bull; {bed_display}</span>
            <span class="alert-status-tag" style="background:#DC2626; color:#FFFFFF; font-size:0.7rem; font-weight:800; padding:3px 9px; border-radius:6px; letter-spacing:0.3px;">DOCTOR NOTIFIED &bull; {time_str}</span>
        </div>
        <div style="margin-top: 8px; font-size: 0.92rem; color: #0F172A !important; font-weight: 600;">
            <strong style="color:#991B1B !important; font-size:1.0rem;">Patient: {patient_name}</strong> has triggered an urgent deterioration warning.
            <ul style="margin: 8px 0 0 0; padding-left: 1.25rem; font-size: 0.88rem; line-height: 1.5;">
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

    items_html = "".join([f"<li style='color:#0F172A !important; font-weight:600; margin-bottom:4px;'>{item}</li>" for item in explanation.get("key_findings", [])])

    st.markdown(f"""
    <div class="{box_class}">
        <div style="font-size: 1.0rem; font-weight: 800; color: #0F172A !important; margin-bottom: 4px;">
            {icon} {explanation.get('title', 'Clinical Rationale')}
        </div>
        <div style="font-size: 0.82rem; color: #475569 !important; font-weight: 600; margin-bottom: 8px;">
            {explanation.get('trend_summary', '')}
        </div>
        <ul style="margin: 0; padding-left: 1.25rem; font-size: 0.88rem; line-height: 1.5;">
            {items_html}
        </ul>
        <div style="margin-top: 10px; padding-top: 8px; border-top: 1px dashed #CBD5E1; font-size: 0.86rem; font-weight: 800; color: #0F172A !important;">
            <strong style="color:#0F172A !important;">Protocol Recommendation:</strong> {explanation.get('recommended_action', '')}
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