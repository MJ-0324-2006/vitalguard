"""
Integration test for VitalGuard pipeline:
data -> inference -> risk_logic -> explainability
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data.demo_patients import get_demo_patients
from inference import get_inference_engine
from risk_logic import classify_risk, calculate_news2, calculate_shock_index, calculate_map
from explainability import generate_explanation


def test_pipeline():
    print("=" * 70)
    print("VITALGUARD END-TO-END PIPELINE VALIDATION")
    print("=" * 70)

    engine = get_inference_engine()
    patients = get_demo_patients()

    all_passed = True

    for p in patients:
        history = p["vitals_history"]
        current = history[-1]

        # 1. Inference
        pred = engine.run_prediction(history)
        raw_score = pred["raw_score"]
        uncertainty = pred["uncertainty_std"]

        # 2. Risk Classification
        risk = classify_risk(raw_score, current, uncertainty_std=uncertainty)
        level = risk["level"]
        score = risk["score"]
        conf = risk["confidence_pct"]

        # 3. Explainability
        exp = generate_explanation(level, history, pred["channel_scores"], risk)

        expected = p["target_risk"]
        match = (level == expected)
        if not match:
            all_passed = False

        status = "PASS" if match else "FAIL"
        print(f"[{status}] {p['bed']:<20} | {p['name']:<18} | Expected: {expected:<6} | Got: {level:<6} (Score: {score:.3f}, Conf: {conf}%)")
        print(f"       Action: {exp['title']}")
        if exp['key_findings']:
            print(f"       Evidence: {exp['key_findings'][0]}")
        print("-" * 70)

    # Edge case 1: Single reading
    print("\nTesting Edge Case: Single Reading...")
    single_p = patients[-1]
    res_single = engine.run_prediction(single_p["vitals_history"])
    print(f" -> Single reading fallback used: {res_single.get('is_single_reading', False)}")

    # Edge case 2: Empty input
    print("\nTesting Edge Case: Empty Input...")
    res_empty = engine.run_prediction([])
    print(f" -> Empty input fallback used: {res_empty.get('fallback_used', False)}")

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL PATIENT CLASSIFICATIONS AND PIPELINE TESTS PASSED!")
    else:
        print("SOME PATIENT CLASSIFICATIONS DIFFERED FROM EXPECTED.")
    print("=" * 70)


if __name__ == "__main__":
    test_pipeline()
