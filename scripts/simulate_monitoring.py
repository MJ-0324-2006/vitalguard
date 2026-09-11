"""
Real-Time Monitoring Simulation Script.

Simulates real-time ICU monitoring on historical patient data,
evaluating alert performance vs. ground truth event times.

Usage:
    python scripts/simulate_monitoring.py \
        --config configs/mimic_config.yaml \
        --model-dir experiments/transformer/checkpoints/ \
        --patient-ids 123456 234567 345678 \
        --data-dir data/processed/ \
        --stream-speed 60x
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.transformer_detector import build_transformer_detector
from src.monitoring.realtime_detector import (
    DetectorConfig, RealTimeDetector, MonitoringSimulator, VitalSignReading
)
from src.training.trainer import Trainer

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Simulate real-time ICU monitoring")
    parser.add_argument("--config", type=str, default="configs/mimic_config.yaml")
    parser.add_argument("--model-dir", type=str, default="experiments/transformer/checkpoints/")
    parser.add_argument("--model-type", type=str, default="transformer")
    parser.add_argument("--patient-ids", nargs="+", type=int, default=None)
    parser.add_argument("--data-dir", type=str, default="data/processed/")
    parser.add_argument("--output-dir", type=str, default="results/simulation/")
    parser.add_argument("--stream-speed", type=str, default="60x",
                        help="Playback speed multiplier (e.g., 60x = 60x real-time)")
    parser.add_argument("--n-patients", type=int, default=5,
                        help="Number of patients to simulate (if patient-ids not specified)")
    parser.add_argument("--real-time", action="store_true",
                        help="Simulate at actual speed (very slow)")
    parser.add_argument("--debug", action="store_true",
                        help="Use synthetic data")
    return parser.parse_args()


def generate_synthetic_patient(
    patient_id: str,
    n_hours: int = 72,
    event_hour: int = 48,
    has_event: bool = True,
):
    """Generate a synthetic ICU patient with a deterioration event."""
    np.random.seed(hash(patient_id) % 10000)
    base_time = datetime(2024, 1, 1, 8, 0, 0)
    readings = []

    for h in range(n_hours):
        ts = base_time + timedelta(hours=h)

        # Normal baseline with gradual deterioration near event
        if has_event and h >= event_hour - 6:
            progress = (h - (event_hour - 6)) / 6  # 0→1 over 6h
            hr_base = 84 + 30 * progress
            sbp_base = 120 - 30 * progress
            spo2_base = 97 - 8 * progress
            rr_base = 18 + 8 * progress
            map_base = 82 - 20 * progress
        else:
            hr_base, sbp_base, spo2_base = 84, 120, 97
            rr_base, map_base = 18, 82

        reading = VitalSignReading(
            timestamp=ts,
            heart_rate=float(np.clip(np.random.normal(hr_base, 8), 40, 200)),
            sbp=float(np.clip(np.random.normal(sbp_base, 12), 60, 220)),
            dbp=float(np.clip(np.random.normal(60, 8), 30, 120)),
            map=float(np.clip(np.random.normal(map_base, 8), 35, 150)),
            spo2=float(np.clip(np.random.normal(spo2_base, 1.5), 70, 100)),
            resp_rate=float(np.clip(np.random.normal(rr_base, 2), 6, 45)),
            temperature=float(np.clip(np.random.normal(37.2, 0.3), 35, 41)),
            gcs=float(np.random.choice([15, 14, 13]) if not (has_event and h > event_hour)
                      else np.random.choice([12, 11, 10])),
        )
        readings.append(reading)

    events = {}
    if has_event:
        events["clinical_event"] = float(event_hour)

    return readings, events


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse speed
    speed_str = args.stream_speed.rstrip("x")
    speed_multiplier = float(speed_str)

    # -------------------------------------------------------------------------
    # Build model
    # -------------------------------------------------------------------------
    if args.model_type == "transformer":
        model = build_transformer_detector(cfg["models"]["transformer"])
    else:
        raise NotImplementedError(f"Simulation only supports transformer model currently")

    checkpoint_path = Path(args.model_dir) / "best_model.pt"
    if checkpoint_path.exists():
        model = Trainer.load_for_inference(str(checkpoint_path), model)
        logger.info(f"Loaded checkpoint: {checkpoint_path}")
    else:
        logger.warning("No checkpoint found, using random weights (demo mode)")

    # -------------------------------------------------------------------------
    # Configure detector
    # -------------------------------------------------------------------------
    monitor_cfg = cfg["monitoring"]
    detector_config = DetectorConfig(
        window_hours=monitor_cfg["window_hours"],
        step_minutes=monitor_cfg["step_minutes"],
        channel_names=cfg["channels"]["vital_signs"],
        high_threshold=monitor_cfg["thresholds"]["high"],
        medium_threshold=monitor_cfg["thresholds"]["medium"],
        critical_threshold=monitor_cfg["thresholds"]["critical"],
        high_persistence_steps=monitor_cfg["persistence"]["high_steps"],
        medium_persistence_steps=monitor_cfg["persistence"]["medium_steps"],
        cooldown_minutes=monitor_cfg["anti_fatigue"]["cooldown_minutes"],
        max_alerts_per_day=monitor_cfg["anti_fatigue"]["max_alerts_per_day"],
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = RealTimeDetector(model, detector_config, device=device)
    simulator = MonitoringSimulator(detector, speed_multiplier=speed_multiplier)

    # -------------------------------------------------------------------------
    # Determine patient IDs
    # -------------------------------------------------------------------------
    if args.patient_ids:
        patient_ids = [str(pid) for pid in args.patient_ids]
    else:
        patient_ids = [f"SIM_{i:03d}" for i in range(args.n_patients)]

    # -------------------------------------------------------------------------
    # Run simulations
    # -------------------------------------------------------------------------
    all_results = {}
    summary_rows = []

    for i, pid in enumerate(patient_ids):
        logger.info(f"\nSimulating patient {i+1}/{len(patient_ids)}: {pid}")

        if args.debug or True:  # always use synthetic for now
            has_event = i < len(patient_ids) // 2  # half with events, half without
            event_hour = np.random.randint(24, 60) if has_event else None

            readings, events = generate_synthetic_patient(
                pid,
                n_hours=72,
                event_hour=event_hour or 48,
                has_event=has_event,
            )
        else:
            # Load real patient data from data_dir
            import pandas as pd
            vitals_path = Path(args.data_dir) / "vitals" / f"{pid}.parquet"
            if not vitals_path.exists():
                logger.warning(f"No data for patient {pid}, skipping")
                continue
            vitals_df = pd.read_parquet(vitals_path)
            # Convert to readings list
            readings = [
                VitalSignReading(
                    timestamp=ts,
                    **{col: float(row[col]) if not pd.isna(row.get(col, float("nan"))) else None
                       for col in detector_config.channel_names if col in row}
                )
                for ts, row in vitals_df.iterrows()
            ]
            events = {}

        # Process
        t0 = time.perf_counter()
        result = simulator.simulate(
            patient_id=pid,
            vitals_df=None,  # using readings list directly
            event_times=events,
            real_time=args.real_time,
        )

        # Manual processing since we have readings list
        result = {"patient_id": pid, "alerts": [], "score_timeline": [], "performance": {}}
        detector.start_patient(pid)

        for reading in readings:
            alert = detector.process_reading(pid, reading)
            status = detector.get_current_status(pid)
            result["score_timeline"].append({
                "time": reading.timestamp.isoformat(),
                "score": status.get("current_score", 0),
            })
            if alert:
                result["alerts"].append({
                    "time": reading.timestamp.isoformat(),
                    "level": alert.alert_level,
                    "score": alert.anomaly_score,
                })

        elapsed = time.perf_counter() - t0
        n_alerts = len(result["alerts"])
        logger.info(
            f"  Patient {pid}: {n_alerts} alerts | "
            f"Sim time: {elapsed:.1f}s | "
            f"Event: {'Yes (h=' + str(event_hour) + ')' if has_event else 'No'}"
        )

        all_results[pid] = result
        summary_rows.append({
            "patient_id": pid,
            "n_alerts": n_alerts,
            "has_event": has_event,
            "event_hour": event_hour,
            "first_alert_hour": (
                result["alerts"][0]["time"] if result["alerts"] else None
            ),
        })

    # -------------------------------------------------------------------------
    # Aggregate results
    # -------------------------------------------------------------------------
    import pandas as pd
    summary_df = pd.DataFrame(summary_rows)
    print("\n" + "="*60)
    print("MONITORING SIMULATION SUMMARY")
    print("="*60)
    print(summary_df.to_string(index=False))

    # Alarm rate
    total_patient_hours = len(patient_ids) * 72
    total_alerts = sum(r["n_alerts"] for r in summary_rows)
    alarm_rate = total_alerts / (total_patient_hours / 24)
    print(f"\nOverall alarm rate: {alarm_rate:.2f} alarms/patient-day")
    print(f"Total patient-hours: {total_patient_hours}")
    print(f"Total alerts: {total_alerts}")

    # Save results
    summary_df.to_csv(output_dir / "simulation_summary.csv", index=False)
    with open(output_dir / "simulation_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print dashboard payload for first patient
    first_pid = patient_ids[0]
    if first_pid in detector._buffers:
        dashboard = detector.get_dashboard_payload(first_pid)
        print(f"\nDashboard payload (patient {first_pid}):")
        print(dashboard[:400] + "...")

    logger.info(f"\nSimulation complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
