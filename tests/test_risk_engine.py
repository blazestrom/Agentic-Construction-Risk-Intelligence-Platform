"""
Unit Tests for Site Risk Engine & Deterministic Rules
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.risk_engine.evaluator import (
    SiteRiskEvaluator,
    compute_normalized_center_distance,
    boxes_intersect_or_near
)

def test_spatial_proximity_computation():
    box1 = [0.2, 0.2, 0.1, 0.1]
    box2 = [0.2, 0.25, 0.1, 0.1]
    dist = compute_normalized_center_distance(box1, box2)
    assert round(dist, 3) == 0.05

def test_boxes_intersect_or_near():
    box_a = [10.0, 10.0, 50.0, 50.0]
    box_b = [45.0, 45.0, 90.0, 90.0]
    assert boxes_intersect_or_near(box_a, box_b, margin_ratio=0.1) is True
    
    distant_box = [200.0, 200.0, 300.0, 300.0]
    assert boxes_intersect_or_near(box_a, distant_box, margin_ratio=0.1) is False

def test_compliant_scene_scores_low():
    evaluator = SiteRiskEvaluator()
    detections = [
        {"class_name": "Person", "category": "worker", "confidence": 0.90, "bbox_xyxy": [10, 10, 50, 100], "bbox_normalized": [0.1, 0.2, 0.1, 0.3]},
        {"class_name": "Hardhat", "category": "ppe_compliant", "confidence": 0.85, "bbox_xyxy": [15, 10, 45, 30], "bbox_normalized": [0.1, 0.1, 0.05, 0.05]},
        {"class_name": "Safety Vest", "category": "ppe_compliant", "confidence": 0.88, "bbox_xyxy": [15, 35, 45, 80], "bbox_normalized": [0.1, 0.2, 0.06, 0.15]}
    ]
    res = evaluator.assess_frame(detections)
    assert res["site_risk_score"] == 0
    assert res["risk_level"] == "LOW"
    assert res["total_hazards_count"] == 0

def test_direct_violation_triggers_hazard():
    evaluator = SiteRiskEvaluator()
    detections = [
        {"class_name": "Person", "category": "worker", "confidence": 0.90, "bbox_xyxy": [10, 10, 50, 100], "bbox_normalized": [0.1, 0.2, 0.1, 0.3]},
        {"class_name": "NO-Hardhat", "category": "ppe_violation", "confidence": 0.85, "bbox_xyxy": [15, 10, 45, 30], "bbox_normalized": [0.1, 0.1, 0.05, 0.05]}
    ]
    res = evaluator.assess_frame(detections)
    assert res["site_risk_score"] > 0
    assert res["total_hazards_count"] >= 1
    hazard_rules = [h["rule_id"] for h in res["hazards"]]
    assert "R01_NO_HARDHAT" in hazard_rules

def test_compound_proximity_hazard():
    evaluator = SiteRiskEvaluator()
    # Unprotected worker directly next to Excavator
    detections = [
        {"class_name": "Person", "category": "worker", "confidence": 0.92, "bbox_xyxy": [100, 100, 160, 260], "bbox_normalized": [0.2, 0.3, 0.1, 0.2]},
        {"class_name": "NO-Safety Vest", "category": "ppe_violation", "confidence": 0.85, "bbox_xyxy": [110, 130, 150, 200], "bbox_normalized": [0.2, 0.28, 0.07, 0.1]},
        {"class_name": "Excavator", "category": "heavy_machinery", "confidence": 0.95, "bbox_xyxy": [150, 80, 400, 350], "bbox_normalized": [0.35, 0.35, 0.3, 0.4]}
    ]
    res = evaluator.assess_frame(detections)
    assert res["site_risk_score"] >= 45
    assert res["risk_level"] in ["HIGH", "CRITICAL"]
    rules = [h["rule_id"] for h in res["hazards"]]
    assert "R04_COMPOUND_PROXIMITY_NO_PPE" in rules or "R03_MACHINERY_PROXIMITY" in rules
