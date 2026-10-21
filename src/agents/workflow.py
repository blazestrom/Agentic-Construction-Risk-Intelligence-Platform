"""
Agentic Workflow Orchestrator
Agentic Construction Risk Intelligence Platform
"""

import time
from pathlib import Path
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

from src.inference.detector import ConstructionSafetyDetector
from src.risk_engine.evaluator import SiteRiskEvaluator
from src.database.models import RiskAssessmentRecord, HazardRecord, DetectionRecord, SiteRecord

class SiteRiskAgenticWorkflow:
    """
    Multi-step Agentic Pipeline:
    1. Receive monitoring input (Image/Video)
    2. Run YOLOv8 detection
    3. Evaluate safety risk rules
    4. Synthesize explainable evidence & executive summary
    5. Persist assessment & hazards to database
    6. Check alert thresholds (HIGH / CRITICAL)
    7. Generate structured report
    """
    
    def __init__(self, detector: Optional[ConstructionSafetyDetector] = None, evaluator: Optional[SiteRiskEvaluator] = None):
        self.detector = detector or ConstructionSafetyDetector()
        self.evaluator = evaluator or SiteRiskEvaluator()
        
    def generate_executive_explanation(self, assessment: Dict[str, Any]) -> str:
        """
        Generate grounded, explainable natural-language summary strictly from observed facts.
        Does NOT fabricate unseen conditions.
        """
        level = assessment["risk_level"]
        score = assessment["site_risk_score"]
        h_count = assessment["total_hazards_count"]
        sev = assessment["severity_breakdown"]
        hazards = assessment["hazards"]
        
        if h_count == 0:
            return (
                f"Site is currently rated at {level} Risk (Score: {score}/100). "
                "All detected personnel have compliant personal protective equipment (PPE), and no hazardous "
                "proximity conditions were observed in this frame."
            )
            
        summary_lines = [
            f"Site Risk Level is currently {level} with a Composite Risk Score of {score}/100.",
            f"Detected {h_count} active hazard condition(s): {sev.get('CRITICAL', 0)} Critical, "
            f"{sev.get('HIGH', 0)} High, {sev.get('MEDIUM', 0)} Medium, {sev.get('LOW', 0)} Low.",
            "\nPrimary Findings & Recommended Immediate Actions:"
        ]
        
        for idx, h in enumerate(hazards[:4]):
            summary_lines.append(f"- [{h['severity']}] {h['rule_name']}: {h['observed_evidence']} -> Action: {h['recommended_action']}")
            
        if level in ["HIGH", "CRITICAL"]:
            summary_lines.append(
                "\nSAFETY ADVISORY: Threshold exceeded for field intervention. Dispatch site supervisor immediately to enforce compliance."
            )
            
        return "\n".join(summary_lines)

    def process_image_assessment(
        self,
        image_input: Any,
        source_filename: str,
        db_session: Optional[Session] = None,
        conf_threshold: float = 0.25,
        save_annotated: bool = True
    ) -> Dict[str, Any]:
        """Execute complete agentic pipeline on an image."""
        t_start = time.time()
        
        # Step 2: Run Object Detection
        det_res = self.detector.detect_image(
            image_input=image_input,
            conf_threshold=conf_threshold,
            save_annotated=save_annotated
        )
        
        # Step 3: Evaluate Risk Rules
        assessment = self.evaluator.assess_frame(det_res["detections"])
        
        # Step 4: Synthesize Explainable Assessment
        explanation = self.generate_executive_explanation(assessment)
        
        # Step 6: Alert Check
        alert_triggered = assessment["site_risk_score"] >= 50
        alert_details = {
            "alert_active": alert_triggered,
            "alert_level": assessment["risk_level"],
            "urgency": "IMMEDIATE" if assessment["risk_level"] == "CRITICAL" else "HIGH" if alert_triggered else "NORMAL",
            "message": f"Alert: {assessment['risk_level']} safety hazard detected on site!" if alert_triggered else "Status normal."
        }
        
        relative_annotated_url = None
        if det_res.get("annotated_image_path"):
            relative_annotated_url = f"/static/outputs/{Path(det_res['annotated_image_path']).name}"
            
        # Step 5: Persist to Database if session provided
        record_id = None
        if db_session:
            site = db_session.query(SiteRecord).first()
            site_id = site.id if site else None
            
            assessment_record = RiskAssessmentRecord(
                site_id=site_id,
                source_filename=source_filename,
                media_type="image",
                site_risk_score=assessment["site_risk_score"],
                risk_level=assessment["risk_level"],
                total_hazards_count=assessment["total_hazards_count"],
                severity_breakdown=assessment["severity_breakdown"],
                annotated_media_url=relative_annotated_url,
                inference_time_ms=det_res["inference_time_ms"]
            )
            db_session.add(assessment_record)
            db_session.flush() # populate assessment_record.id
            record_id = assessment_record.id
            
            # Persist hazards
            for h in assessment["hazards"]:
                hazard_rec = HazardRecord(
                    assessment_id=record_id,
                    hazard_code=h["hazard_id"],
                    rule_id=h["rule_id"],
                    rule_name=h["rule_name"],
                    severity=h["severity"],
                    confidence=h["confidence"],
                    observed_evidence=h["observed_evidence"],
                    rationale=h["rationale"],
                    recommended_action=h["recommended_action"],
                    affected_entity=h.get("affected_entity"),
                    spatial_context=h.get("spatial_context"),
                    primary_box=h.get("primary_box")
                )
                db_session.add(hazard_rec)
                
            # Persist raw detections
            for d in det_res["detections"]:
                det_rec = DetectionRecord(
                    assessment_id=record_id,
                    class_name=d["class_name"],
                    category=d["category"],
                    confidence=d["confidence"],
                    bbox_xyxy=d["bbox_xyxy"]
                )
                db_session.add(det_rec)
                
            db_session.commit()
            
        total_time_ms = round((time.time() - t_start) * 1000.0, 2)
        
        # Step 7: Return comprehensive structured report
        return {
            "assessment_id": record_id,
            "source_filename": source_filename,
            "timestamp": assessment["timestamp"],
            "site_risk_score": assessment["site_risk_score"],
            "risk_level": assessment["risk_level"],
            "total_hazards_count": assessment["total_hazards_count"],
            "severity_breakdown": assessment["severity_breakdown"],
            "executive_summary": explanation,
            "alert": alert_details,
            "annotated_media_url": relative_annotated_url,
            "hazards": assessment["hazards"],
            "detections_count": det_res["detections_count"],
            "detections": det_res["detections"],
            "timing_ms": {
                "inference_time": det_res["inference_time_ms"],
                "total_pipeline_time": total_time_ms
            },
            "formula_explanation": assessment["formula_explanation"]
        }
