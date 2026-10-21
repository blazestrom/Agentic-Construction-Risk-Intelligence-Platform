"""
Explainable Site Risk Evaluator & Scoring Engine
Agentic Construction Risk Intelligence Platform
"""

import math
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from .rules import DEFAULT_RISK_RULES, RiskRuleConfig

def compute_box_center(box: List[float]) -> tuple:
    # box: [x1, y1, x2, y2]
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

def compute_normalized_center_distance(norm_box_a: List[float], norm_box_b: List[float]) -> float:
    # norm_box: [xc, yc, w, h] in [0, 1]
    dx = norm_box_a[0] - norm_box_b[0]
    dy = norm_box_a[1] - norm_box_b[1]
    return math.sqrt(dx * dx + dy * dy)

def boxes_intersect_or_near(box_a: List[float], box_b: List[float], margin_ratio: float = 0.2) -> bool:
    # Expand box_a by margin
    w_a = box_a[2] - box_a[0]
    h_a = box_a[3] - box_a[1]
    ax1 = box_a[0] - w_a * margin_ratio
    ay1 = box_a[1] - h_a * margin_ratio
    ax2 = box_a[2] + w_a * margin_ratio
    ay2 = box_a[3] + h_a * margin_ratio
    
    # Check overlap with box_b
    return not (ax2 < box_b[0] or ax1 > box_b[2] or ay2 < box_b[1] or ay1 > box_b[3])

class SiteRiskEvaluator:
    """
    Evaluates raw YOLO object detections and converts them into structured,
    explainable hazard assessments with deterministic risk scoring.
    """
    
    def __init__(self, rules: Optional[Dict[str, RiskRuleConfig]] = None):
        self.rules = rules or DEFAULT_RISK_RULES
        
    def assess_frame(
        self,
        detections: List[Dict[str, Any]],
        image_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Assess risk for a single frame / image.
        Returns explainable hazards and composite site risk score.
        """
        hazards: List[Dict[str, Any]] = []
        observed_facts: List[Dict[str, Any]] = []
        
        # Partition detections
        workers = []
        no_hardhats = []
        no_vests = []
        machinery = []
        cones = []
        ladders = []
        
        for det in detections:
            cname = det["class_name"]
            conf = det["confidence"]
            cat = det["category"]
            
            observed_facts.append({
                "class_name": cname,
                "confidence": conf,
                "category": cat,
                "bbox": det["bbox_xyxy"]
            })
            
            if cname == "Person":
                workers.append(det)
            elif cname == "NO-Hardhat":
                no_hardhats.append(det)
            elif cname == "NO-Safety Vest":
                no_vests.append(det)
            elif cat == "heavy_machinery":
                machinery.append(det)
            elif cname == "Safety Cone":
                cones.append(det)
            elif cname == "Ladder":
                ladders.append(det)

        # ---------------------------------------------------------------------
        # RULE 1: Unprotected Head (NO-Hardhat)
        # ---------------------------------------------------------------------
        r01 = self.rules.get("R01_NO_HARDHAT")
        if r01:
            for nh in no_hardhats:
                if nh["confidence"] >= r01.min_confidence:
                    hazards.append({
                        "hazard_id": f"HAZ-{uuid.uuid4().hex[:8].upper()}",
                        "rule_id": r01.rule_id,
                        "rule_name": r01.rule_name,
                        "severity": r01.severity,
                        "confidence": nh["confidence"],
                        "observed_evidence": f"Detected '{nh['class_name']}' with {nh['confidence']*100:.1f}% confidence.",
                        "rationale": r01.description,
                        "recommended_action": r01.action,
                        "primary_box": nh["bbox_xyxy"],
                        "affected_entity": "Personnel without headgear",
                        "spatial_context": "Direct visual detection of missing hardhat"
                    })

        # ---------------------------------------------------------------------
        # RULE 2: Low Visibility (NO-Safety Vest)
        # ---------------------------------------------------------------------
        r02 = self.rules.get("R02_NO_VEST")
        if r02:
            for nv in no_vests:
                if nv["confidence"] >= r02.min_confidence:
                    hazards.append({
                        "hazard_id": f"HAZ-{uuid.uuid4().hex[:8].upper()}",
                        "rule_id": r02.rule_id,
                        "rule_name": r02.rule_name,
                        "severity": r02.severity,
                        "confidence": nv["confidence"],
                        "observed_evidence": f"Detected '{nv['class_name']}' with {nv['confidence']*100:.1f}% confidence.",
                        "rationale": r02.description,
                        "recommended_action": r02.action,
                        "primary_box": nv["bbox_xyxy"],
                        "affected_entity": "Personnel without safety vest",
                        "spatial_context": "Direct visual detection of missing vest"
                    })

        # ---------------------------------------------------------------------
        # RULE 3 & 4: Proximity Between Worker & Heavy Machinery
        # ---------------------------------------------------------------------
        r03 = self.rules.get("R03_MACHINERY_PROXIMITY")
        r04 = self.rules.get("R04_COMPOUND_PROXIMITY_NO_PPE")
        
        for w in workers:
            for m in machinery:
                # Proximity check: normalized distance <= 0.28 (in normalized image frame)
                norm_dist = compute_normalized_center_distance(w["bbox_normalized"], m["bbox_normalized"])
                is_near = norm_dist < 0.28 or boxes_intersect_or_near(w["bbox_xyxy"], m["bbox_xyxy"], margin_ratio=0.15)
                
                if is_near:
                    # Check if this worker overlaps with any NO-Hardhat or NO-Safety Vest
                    worker_has_ppe_violation = any(
                        boxes_intersect_or_near(w["bbox_xyxy"], nh["bbox_xyxy"], margin_ratio=0.1)
                        for nh in (no_hardhats + no_vests)
                    )
                    
                    combined_conf = round((w["confidence"] + m["confidence"]) / 2.0, 4)
                    
                    if worker_has_ppe_violation and r04 and combined_conf >= r04.min_confidence:
                        hazards.append({
                            "hazard_id": f"HAZ-{uuid.uuid4().hex[:8].upper()}",
                            "rule_id": r04.rule_id,
                            "rule_name": r04.rule_name,
                            "severity": r04.severity,
                            "confidence": combined_conf,
                            "observed_evidence": f"Worker (conf: {w['confidence']:.2f}) observed adjacent to {m['class_name']} (conf: {m['confidence']:.2f}) with concurrent PPE non-compliance.",
                            "rationale": r04.description,
                            "recommended_action": r04.action,
                            "primary_box": w["bbox_xyxy"],
                            "associated_box": m["bbox_xyxy"],
                            "affected_entity": f"Worker near {m['class_name']}",
                            "spatial_context": f"Proximity index: {norm_dist:.3f} (spatial proximity threshold triggered)"
                        })
                    elif r03 and combined_conf >= r03.min_confidence:
                        hazards.append({
                            "hazard_id": f"HAZ-{uuid.uuid4().hex[:8].upper()}",
                            "rule_id": r03.rule_id,
                            "rule_name": r03.rule_name,
                            "severity": r03.severity,
                            "confidence": combined_conf,
                            "observed_evidence": f"Worker (conf: {w['confidence']:.2f}) observed in operational sweep of {m['class_name']} (conf: {m['confidence']:.2f}).",
                            "rationale": r03.description,
                            "recommended_action": r03.action,
                            "primary_box": w["bbox_xyxy"],
                            "associated_box": m["bbox_xyxy"],
                            "affected_entity": f"Worker near {m['class_name']}",
                            "spatial_context": f"Proximity index: {norm_dist:.3f}"
                        })

        # ---------------------------------------------------------------------
        # RULE 5: Ladder Fall Risk Zone
        # ---------------------------------------------------------------------
        r05 = self.rules.get("R05_LADDER_ZONE")
        if r05:
            for l in ladders:
                for w in workers:
                    if boxes_intersect_or_near(w["bbox_xyxy"], l["bbox_xyxy"], margin_ratio=0.1):
                        combined_conf = round((w["confidence"] + l["confidence"]) / 2.0, 4)
                        if combined_conf >= r05.min_confidence:
                            hazards.append({
                                "hazard_id": f"HAZ-{uuid.uuid4().hex[:8].upper()}",
                                "rule_id": r05.rule_id,
                                "rule_name": r05.rule_name,
                                "severity": r05.severity,
                                "confidence": combined_conf,
                                "observed_evidence": f"Person (conf: {w['confidence']:.2f}) on or adjacent to Ladder (conf: {l['confidence']:.2f}).",
                                "rationale": r05.description,
                                "recommended_action": r05.action,
                                "primary_box": w["bbox_xyxy"],
                                "associated_box": l["bbox_xyxy"],
                                "affected_entity": "Worker on ladder",
                                "spatial_context": "Vertical elevation risk zone"
                            })

        # ---------------------------------------------------------------------
        # Calculate Explainable Overall Site Risk Score (0 - 100)
        # ---------------------------------------------------------------------
        # Formula:
        # Raw = sum(rule_weight * confidence)
        # Score = min(100, round(Raw))
        severity_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
        raw_score = 0.0
        
        for h in hazards:
            sev = h["severity"]
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
            rule_cfg = self.rules.get(h["rule_id"])
            base_wt = rule_cfg.base_weight if rule_cfg else 10.0
            raw_score += base_wt * h["confidence"]
            
        site_risk_score = min(100, int(round(raw_score)))
        
        if site_risk_score >= 75:
            risk_level = "CRITICAL"
        elif site_risk_score >= 50:
            risk_level = "HIGH"
        elif site_risk_score >= 25:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "site_risk_score": site_risk_score,
            "risk_level": risk_level,
            "total_hazards_count": len(hazards),
            "severity_breakdown": severity_counts,
            "hazards": hazards,
            "observed_facts_count": len(observed_facts),
            "formula_explanation": {
                "formula": "min(100, round(sum(RuleBaseWeight_i * Confidence_i)))",
                "scale": "0 to 100 (0-24: LOW, 25-49: MEDIUM, 50-74: HIGH, 75-100: CRITICAL)",
                "critical_weights": "NO-Hardhat: 25, NO-Vest: 15, Proximity: 30, Compound Unprotected Proximity: 45",
                "disclaimer": "This is an explainable decision-support indicator based on visual detection, not a certified structural or legal safety clearance."
            }
        }
