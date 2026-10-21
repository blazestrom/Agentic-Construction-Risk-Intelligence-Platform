"""
Verification Script for Phase 4: Site Risk Agent
Tests explainable hazard candidate creation, proximity logic, and composite scoring.
"""

import sys
import json
from pathlib import Path

# Ensure root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.detector import ConstructionSafetyDetector
from src.risk_engine.evaluator import SiteRiskEvaluator

def test_risk_engine():
    print("=" * 60)
    print("TESTING PHASE 4: SITE RISK AGENT")
    print("=" * 60)
    
    detector = ConstructionSafetyDetector()
    evaluator = SiteRiskEvaluator()
    
    # Run detector on test image containing violations
    test_img = Path("test/images/000005_jpg.rf.c613c066b79795d219855aca269760e6.jpg")
    if not test_img.exists():
        # Fallback to any valid test image
        test_img = next(Path("test/images").glob("*.jpg"))
        
    print(f"\n[1] Running Detection & Risk Assessment on: {test_img.name}")
    det_result = detector.detect_image(test_img, conf_threshold=0.25)
    
    assessment = evaluator.assess_frame(det_result["detections"])
    
    print(f"\n[2] Risk Assessment Results:")
    print(f"  Site Risk Score : {assessment['site_risk_score']} / 100")
    print(f"  Risk Level      : {assessment['risk_level']}")
    print(f"  Total Hazards   : {assessment['total_hazards_count']}")
    print(f"  Severity Counts : {assessment['severity_breakdown']}")
    print(f"  Observed Facts  : {assessment['observed_facts_count']}")
    
    print("\n[3] Identified Hazard Candidates (Explainable Evidence):")
    for idx, h in enumerate(assessment['hazards']):
        print(f"  Hazard #{idx+1} [{h['severity']}]: {h['rule_name']}")
        print(f"    Evidence : {h['observed_evidence']}")
        print(f"    Rationale: {h['rationale']}")
        print(f"    Action   : {h['recommended_action']}")
        print(f"    Context  : {h['spatial_context']}")
        print()
        
    # Test Scenario 2: Synthetic Proximity Simulation (Worker near active Excavator without Hardhat)
    print("[4] Testing Synthetic Critical Scenario (Worker near Excavator + NO-Hardhat)...")
    synthetic_detections = [
        {
            "class_name": "Person",
            "category": "worker",
            "confidence": 0.88,
            "bbox_xyxy": [100.0, 100.0, 200.0, 300.0],
            "bbox_normalized": [0.25, 0.35, 0.16, 0.35]
        },
        {
            "class_name": "NO-Hardhat",
            "category": "ppe_violation",
            "confidence": 0.82,
            "bbox_xyxy": [120.0, 100.0, 180.0, 160.0],
            "bbox_normalized": [0.25, 0.22, 0.09, 0.1]
        },
        {
            "class_name": "Excavator",
            "category": "heavy_machinery",
            "confidence": 0.94,
            "bbox_xyxy": [180.0, 80.0, 500.0, 450.0],
            "bbox_normalized": [0.55, 0.45, 0.5, 0.6]
        }
    ]
    
    critical_assessment = evaluator.assess_frame(synthetic_detections)
    print(f"  Score: {critical_assessment['site_risk_score']} / 100, Level: {critical_assessment['risk_level']}")
    assert critical_assessment['risk_level'] in ["HIGH", "CRITICAL"], "Should trigger HIGH or CRITICAL risk"
    print(f"  Triggered Rules: {[h['rule_name'] for h in critical_assessment['hazards']]}")
    
    # Save output
    out_file = Path("reports/sample_risk_assessment.json")
    with open(out_file, "w") as f:
        json.dump({
            "real_image_assessment": assessment,
            "critical_scenario_assessment": critical_assessment
        }, f, indent=2)
        
    print(f"\n[5] Risk Assessment JSON saved to: {out_file}")
    print("\n" + "=" * 60)
    print("PHASE 4 SITE RISK AGENT VERIFICATION PASSED")
    print("=" * 60)

if __name__ == "__main__":
    test_risk_engine()
