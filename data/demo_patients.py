"""
Demo patient dataset for VitalGuard.
Curated from realistic MIMIC-III ICU test cohort profiles representing:
1. Stable recovery (Green)
2. Borderline / subtle compensation (Yellow)
3. Acute septic shock deterioration (Red)
4. Hypoxemic respiratory failure (Red)
5. Cardiogenic / borderline MAP (Yellow)
6. Newly admitted patient (Single reading - graceful fallback demonstration)
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any


def get_demo_patients() -> List[Dict[str, Any]]:
    now = datetime.now().replace(minute=0, second=0, microsecond=0)

    patients = [
        {
            "id": "MIMIC-41082",
            "bed": "Bed 101 (MICU)",
            "name": "Rajan Verma",
            "age": 58,
            "gender": "Male",
            "diagnosis": "Post-operative monitoring after elective hemicolectomy",
            "admission_hours_ago": 18,
            "vitals_history": [
                {
                    "time": (now - timedelta(hours=5)).strftime("%H:%M"),
                    "heart_rate": 72.0, "sbp": 122.0, "dbp": 78.0, "map": 92.7,
                    "spo2": 98.0, "resp_rate": 15.0, "temperature": 36.8, "gcs": 15.0
                },
                {
                    "time": (now - timedelta(hours=4)).strftime("%H:%M"),
                    "heart_rate": 74.0, "sbp": 120.0, "dbp": 76.0, "map": 90.7,
                    "spo2": 98.0, "resp_rate": 14.0, "temperature": 36.9, "gcs": 15.0
                },
                {
                    "time": (now - timedelta(hours=3)).strftime("%H:%M"),
                    "heart_rate": 71.0, "sbp": 124.0, "dbp": 80.0, "map": 94.7,
                    "spo2": 99.0, "resp_rate": 16.0, "temperature": 36.7, "gcs": 15.0
                },
                {
                    "time": (now - timedelta(hours=2)).strftime("%H:%M"),
                    "heart_rate": 73.0, "sbp": 121.0, "dbp": 77.0, "map": 91.7,
                    "spo2": 98.0, "resp_rate": 15.0, "temperature": 36.8, "gcs": 15.0
                },
                {
                    "time": (now - timedelta(hours=1)).strftime("%H:%M"),
                    "heart_rate": 70.0, "sbp": 123.0, "dbp": 79.0, "map": 93.7,
                    "spo2": 98.0, "resp_rate": 14.0, "temperature": 36.8, "gcs": 15.0
                },
                {
                    "time": now.strftime("%H:%M"),
                    "heart_rate": 72.0, "sbp": 122.0, "dbp": 78.0, "map": 92.7,
                    "spo2": 98.0, "resp_rate": 15.0, "temperature": 36.8, "gcs": 15.0
                }
            ],
            "baseline_note": "Patient resting comfortably. Surgical site clean, hemodynamics stable, zero escalation required.",
            "target_risk": "GREEN",
        },
        {
            "id": "MIMIC-52190",
            "bed": "Bed 104 (SICU)",
            "name": "Sunita Patel",
            "age": 64,
            "gender": "Female",
            "diagnosis": "Complicated pyelonephritis with borderline systemic response",
            "admission_hours_ago": 14,
            "vitals_history": [
                {
                    "time": (now - timedelta(hours=5)).strftime("%H:%M"),
                    "heart_rate": 78.0, "sbp": 128.0, "dbp": 82.0, "map": 97.3,
                    "spo2": 97.0, "resp_rate": 16.0, "temperature": 37.1, "gcs": 15.0
                },
                {
                    "time": (now - timedelta(hours=4)).strftime("%H:%M"),
                    "heart_rate": 82.0, "sbp": 124.0, "dbp": 78.0, "map": 93.3,
                    "spo2": 96.0, "resp_rate": 18.0, "temperature": 37.3, "gcs": 15.0
                },
                {
                    "time": (now - timedelta(hours=3)).strftime("%H:%M"),
                    "heart_rate": 86.0, "sbp": 118.0, "dbp": 74.0, "map": 88.7,
                    "spo2": 96.0, "resp_rate": 19.0, "temperature": 37.6, "gcs": 15.0
                },
                {
                    "time": (now - timedelta(hours=2)).strftime("%H:%M"),
                    "heart_rate": 92.0, "sbp": 114.0, "dbp": 72.0, "map": 86.0,
                    "spo2": 95.0, "resp_rate": 21.0, "temperature": 37.9, "gcs": 14.0
                },
                {
                    "time": (now - timedelta(hours=1)).strftime("%H:%M"),
                    "heart_rate": 95.0, "sbp": 110.0, "dbp": 70.0, "map": 83.3,
                    "spo2": 95.0, "resp_rate": 22.0, "temperature": 38.1, "gcs": 14.0
                },
                {
                    "time": now.strftime("%H:%M"),
                    "heart_rate": 98.0, "sbp": 108.0, "dbp": 68.0, "map": 81.3,
                    "spo2": 94.0, "resp_rate": 23.0, "temperature": 38.3, "gcs": 14.0
                }
            ],
            "baseline_note": "Early compensatory response: subtle progressive tachycardia (78 → 98 bpm) and tachypnea with low-grade fever.",
            "target_risk": "YELLOW",
        },
        {
            "id": "MIMIC-68421",
            "bed": "Bed 203 (MICU)",
            "name": "Anthony DeSouza",
            "age": 71,
            "gender": "Male",
            "diagnosis": "Severe intra-abdominal sepsis deteriorating toward septic shock",
            "admission_hours_ago": 26,
            "vitals_history": [
                {
                    "time": (now - timedelta(hours=5)).strftime("%H:%M"),
                    "heart_rate": 88.0, "sbp": 114.0, "dbp": 72.0, "map": 86.0,
                    "spo2": 96.0, "resp_rate": 18.0, "temperature": 37.8, "gcs": 14.0
                },
                {
                    "time": (now - timedelta(hours=4)).strftime("%H:%M"),
                    "heart_rate": 96.0, "sbp": 106.0, "dbp": 66.0, "map": 79.3,
                    "spo2": 94.0, "resp_rate": 21.0, "temperature": 38.2, "gcs": 14.0
                },
                {
                    "time": (now - timedelta(hours=3)).strftime("%H:%M"),
                    "heart_rate": 105.0, "sbp": 98.0, "dbp": 60.0, "map": 72.7,
                    "spo2": 93.0, "resp_rate": 23.0, "temperature": 38.7, "gcs": 13.0
                },
                {
                    "time": (now - timedelta(hours=2)).strftime("%H:%M"),
                    "heart_rate": 112.0, "sbp": 92.0, "dbp": 56.0, "map": 68.0,
                    "spo2": 92.0, "resp_rate": 25.0, "temperature": 38.9, "gcs": 13.0
                },
                {
                    "time": (now - timedelta(hours=1)).strftime("%H:%M"),
                    "heart_rate": 118.0, "sbp": 86.0, "dbp": 52.0, "map": 63.3,
                    "spo2": 91.0, "resp_rate": 27.0, "temperature": 39.2, "gcs": 12.0
                },
                {
                    "time": now.strftime("%H:%M"),
                    "heart_rate": 124.0, "sbp": 82.0, "dbp": 48.0, "map": 59.3,
                    "spo2": 90.0, "resp_rate": 28.0, "temperature": 39.4, "gcs": 11.0
                }
            ],
            "baseline_note": "CRITICAL: Classical septic shock triad (severe tachycardia > 120, hypotension MAP < 65, tachypnea, high fever).",
            "target_risk": "RED",
        },
        {
            "id": "MIMIC-73915",
            "bed": "Bed 208 (CCU)",
            "name": "Priya Sharma",
            "age": 52,
            "gender": "Female",
            "diagnosis": "Acute hypoxemic respiratory failure secondary to viral pneumonia",
            "admission_hours_ago": 8,
            "vitals_history": [
                {
                    "time": (now - timedelta(hours=5)).strftime("%H:%M"),
                    "heart_rate": 84.0, "sbp": 132.0, "dbp": 84.0, "map": 100.0,
                    "spo2": 95.0, "resp_rate": 20.0, "temperature": 37.4, "gcs": 15.0
                },
                {
                    "time": (now - timedelta(hours=4)).strftime("%H:%M"),
                    "heart_rate": 88.0, "sbp": 136.0, "dbp": 86.0, "map": 102.7,
                    "spo2": 93.0, "resp_rate": 22.0, "temperature": 37.6, "gcs": 15.0
                },
                {
                    "time": (now - timedelta(hours=3)).strftime("%H:%M"),
                    "heart_rate": 94.0, "sbp": 140.0, "dbp": 88.0, "map": 105.3,
                    "spo2": 91.0, "resp_rate": 25.0, "temperature": 37.8, "gcs": 14.0
                },
                {
                    "time": (now - timedelta(hours=2)).strftime("%H:%M"),
                    "heart_rate": 100.0, "sbp": 142.0, "dbp": 90.0, "map": 107.3,
                    "spo2": 89.0, "resp_rate": 28.0, "temperature": 38.0, "gcs": 14.0
                },
                {
                    "time": (now - timedelta(hours=1)).strftime("%H:%M"),
                    "heart_rate": 106.0, "sbp": 145.0, "dbp": 92.0, "map": 109.7,
                    "spo2": 88.0, "resp_rate": 30.0, "temperature": 38.1, "gcs": 13.0
                },
                {
                    "time": now.strftime("%H:%M"),
                    "heart_rate": 110.0, "sbp": 148.0, "dbp": 94.0, "map": 112.0,
                    "spo2": 87.0, "resp_rate": 32.0, "temperature": 38.2, "gcs": 13.0
                }
            ],
            "baseline_note": "Marked respiratory compromise: precipitous SpO2 drop (95% -> 87%), high work of breathing with severe tachypnea.",
            "target_risk": "RED",
        },
        {
            "id": "MIMIC-80234",
            "bed": "Bed 302 (MICU)",
            "name": "David Chen",
            "age": 68,
            "gender": "Male",
            "diagnosis": "Non-STEMI post-PCI, borderline vasopressor requirement",
            "admission_hours_ago": 32,
            "vitals_history": [
                {
                    "time": (now - timedelta(hours=5)).strftime("%H:%M"),
                    "heart_rate": 62.0, "sbp": 104.0, "dbp": 68.0, "map": 80.0,
                    "spo2": 97.0, "resp_rate": 16.0, "temperature": 36.6, "gcs": 15.0
                },
                {
                    "time": (now - timedelta(hours=4)).strftime("%H:%M"),
                    "heart_rate": 60.0, "sbp": 100.0, "dbp": 64.0, "map": 76.0,
                    "spo2": 97.0, "resp_rate": 17.0, "temperature": 36.6, "gcs": 15.0
                },
                {
                    "time": (now - timedelta(hours=3)).strftime("%H:%M"),
                    "heart_rate": 59.0, "sbp": 96.0, "dbp": 62.0, "map": 73.3,
                    "spo2": 96.0, "resp_rate": 18.0, "temperature": 36.5, "gcs": 15.0
                },
                {
                    "time": (now - timedelta(hours=2)).strftime("%H:%M"),
                    "heart_rate": 57.0, "sbp": 94.0, "dbp": 60.0, "map": 71.3,
                    "spo2": 96.0, "resp_rate": 18.0, "temperature": 36.5, "gcs": 14.0
                },
                {
                    "time": (now - timedelta(hours=1)).strftime("%H:%M"),
                    "heart_rate": 58.0, "sbp": 92.0, "dbp": 58.0, "map": 69.3,
                    "spo2": 95.0, "resp_rate": 19.0, "temperature": 36.4, "gcs": 14.0
                },
                {
                    "time": now.strftime("%H:%M"),
                    "heart_rate": 56.0, "sbp": 90.0, "dbp": 56.0, "map": 67.3,
                    "spo2": 95.0, "resp_rate": 19.0, "temperature": 36.5, "gcs": 14.0
                }
            ],
            "baseline_note": "Borderline cardiogenic hypotension (SBP 90, MAP 67). Requires close titration of fluid/inotropes.",
            "target_risk": "YELLOW",
        },
        {
            "id": "MIMIC-91402",
            "bed": "Bed 305 (Step-down)",
            "name": "Arjun Nair",
            "age": 45,
            "gender": "Male",
            "diagnosis": "Newly admitted: blunt trauma observation",
            "admission_hours_ago": 1,
            "vitals_history": [
                {
                    "time": now.strftime("%H:%M"),
                    "heart_rate": 84.0, "sbp": 126.0, "dbp": 80.0, "map": 95.3,
                    "spo2": 98.0, "resp_rate": 16.0, "temperature": 36.9, "gcs": 15.0
                }
            ],
            "baseline_note": "New admission (1 reading available). Single reading evaluated with fallback logic pending longitudinal trend data.",
            "target_risk": "GREEN",
        }
    ]
    return patients
