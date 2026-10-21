"""
FastAPI Application Backend
Agentic Construction Risk Intelligence Platform
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timezone

from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import desc

# Root path injection
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.database.connection import init_db, get_db
from src.database.models import RiskAssessmentRecord, HazardRecord, DetectionRecord, SiteRecord
from src.inference.detector import ConstructionSafetyDetector
from src.risk_engine.evaluator import SiteRiskEvaluator
from src.agents.workflow import SiteRiskAgenticWorkflow

# Initialize outputs directory
OUTPUTS_DIR = ROOT_DIR / "reports" / "inference_outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

from contextlib import asynccontextmanager

# Singleton instances
detector_instance = None
evaluator_instance = None
workflow_instance = None

def get_workflow():
    global detector_instance, evaluator_instance, workflow_instance
    if workflow_instance is None:
        init_db()
        detector_instance = ConstructionSafetyDetector()
        evaluator_instance = SiteRiskEvaluator()
        workflow_instance = SiteRiskAgenticWorkflow(
            detector=detector_instance,
            evaluator=evaluator_instance
        )
    return workflow_instance

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("[Startup] Initializing Database & Workflow Pipeline...")
    get_workflow()
    print("[Startup] Backend successfully loaded and ready.")
    yield
    # Shutdown logic if any

# Create FastAPI app
app = FastAPI(
    title="Agentic Construction Risk Intelligence Platform",
    description="Explainable site hazard monitoring, computer vision detection, and risk intelligence API",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for Frontend React/Vite development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Mount static files to serve annotated media to the dashboard
app.mount("/static/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

@app.get("/", tags=["Dashboard UI"])
def serve_dashboard():
    """Serve the React + Tailwind interactive risk intelligence dashboard."""
    dashboard_file = ROOT_DIR / "frontend" / "index.html"
    if dashboard_file.exists():
        return FileResponse(str(dashboard_file))
    return JSONResponse({"message": "Construction Risk Intelligence Platform API is running. Visit /docs for Swagger API."})

@app.get("/health", tags=["System"])
def health_check():
    """System health check and component status."""
    wf = get_workflow()
    return {
        "status": "online",
        "service": "Construction Risk Intelligence Platform",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_loaded": detector_instance is not None,
        "device": detector_instance.device if detector_instance else "unknown"
    }

@app.get("/model/info", tags=["Model"])
def model_info():
    """Information about the active YOLOv8 model, weights, and supported classes."""
    wf = get_workflow()
    return {
        "weights_path": str(detector_instance.weights_path),
        "device": detector_instance.device,
        "total_classes": len(detector_instance.class_names),
        "classes": detector_instance.class_names
    }

@app.post("/detect/image", tags=["Inference"])
async def detect_image(
    file: UploadFile = File(...),
    conf_threshold: float = Form(0.25),
    iou_threshold: float = Form(0.45)
):
    """Run object detection on an uploaded construction site image."""
    suffix = Path(file.filename).suffix.lower() if file.filename else ".jpg"
    is_image = (file.content_type and file.content_type.startswith("image/")) or (
        suffix in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
    )
    if not is_image:
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image format.")
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
        
    get_workflow()
    try:
        results = detector_instance.detect_image(
            image_input=tmp_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            save_annotated=True
        )
        
        annotated_url = None
        if results.get("annotated_image_path"):
            annotated_url = f"/static/outputs/{Path(results['annotated_image_path']).name}"
            
        results["annotated_media_url"] = annotated_url
        results["original_filename"] = file.filename
        return results
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

@app.post("/detect/video", tags=["Inference"])
async def detect_video(
    file: UploadFile = File(...),
    conf_threshold: float = Form(0.25),
    frame_stride: int = Form(2),
    max_frames: Optional[int] = Form(100)
):
    """Run object detection on an uploaded construction site video."""
    suffix = Path(file.filename).suffix.lower() if file.filename else ".mp4"
    is_video = (file.content_type and file.content_type.startswith("video/")) or (
        suffix in [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    )
    if not is_video:
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid video format.")
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
        
    get_workflow()
    try:
        results = detector_instance.detect_video(
            video_input=tmp_path,
            conf_threshold=conf_threshold,
            frame_stride=frame_stride,
            max_frames=max_frames
        )
        
        annotated_url = None
        if results.get("annotated_video_path"):
            annotated_url = f"/static/outputs/{Path(results['annotated_video_path']).name}"
            
        results["annotated_video_url"] = annotated_url
        results["original_filename"] = file.filename
        return results
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

@app.post("/risk/assess", tags=["Agentic Risk Engine"])
async def assess_risk(
    file: UploadFile = File(...),
    conf_threshold: float = Form(0.25),
    db: Session = Depends(get_db)
):
    """
    Execute full agentic workflow:
    Detection -> Risk Rules -> Explainable Evidence -> Database Storage -> Alert Evaluation.
    """
    suffix = Path(file.filename).suffix.lower() if file.filename else ".jpg"
    is_image = (file.content_type and file.content_type.startswith("image/")) or (
        suffix in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
    )
    if not is_image:
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image format.")
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
        
    wf = get_workflow()
    try:
        assessment_response = wf.process_image_assessment(
            image_input=tmp_path,
            source_filename=file.filename,
            db_session=db,
            conf_threshold=conf_threshold,
            save_annotated=True
        )
        return assessment_response
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

@app.get("/risks", tags=["Risk Management"])
def list_risk_assessments(
    limit: int = Query(20, ge=1, le=100),
    risk_level: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Retrieve historical risk assessments and hazards stored in the database."""
    init_db()
    query = db.query(RiskAssessmentRecord).order_by(desc(RiskAssessmentRecord.created_at))
    if risk_level:
        query = query.filter(RiskAssessmentRecord.risk_level == risk_level.upper())
        
    records = query.limit(limit).all()
    results = []
    for r in records:
        results.append({
            "id": r.id,
            "source_filename": r.source_filename,
            "media_type": r.media_type,
            "site_risk_score": r.site_risk_score,
            "risk_level": r.risk_level,
            "total_hazards_count": r.total_hazards_count,
            "severity_breakdown": r.severity_breakdown,
            "annotated_media_url": r.annotated_media_url,
            "inference_time_ms": r.inference_time_ms,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "hazards": [
                {
                    "hazard_code": h.hazard_code,
                    "rule_id": h.rule_id,
                    "rule_name": h.rule_name,
                    "severity": h.severity,
                    "confidence": h.confidence,
                    "observed_evidence": h.observed_evidence,
                    "recommended_action": h.recommended_action
                } for h in r.hazards
            ]
        })
    return {"count": len(results), "assessments": results}

@app.get("/dashboard/summary", tags=["Dashboard"])
def get_dashboard_summary(db: Session = Depends(get_db)):
    """Retrieve aggregate telemetry and metrics for the frontend risk monitoring dashboard."""
    init_db()
    total_assessments = db.query(RiskAssessmentRecord).count()
    recent = db.query(RiskAssessmentRecord).order_by(desc(RiskAssessmentRecord.created_at)).limit(10).all()
    
    # Severity distribution
    severity_totals = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    all_hazards = db.query(HazardRecord).all()
    for h in all_hazards:
        if h.severity in severity_totals:
            severity_totals[h.severity] += 1
            
    latest = recent[0] if recent else None
    
    # Class frequency
    detections = db.query(DetectionRecord).all()
    class_distribution = {}
    for d in detections:
        class_distribution[d.class_name] = class_distribution.get(d.class_name, 0) + 1
        
    return {
        "status": "active",
        "current_risk_score": latest.site_risk_score if latest else 0,
        "current_risk_level": latest.risk_level if latest else "LOW",
        "latest_source": latest.source_filename if latest else None,
        "latest_annotated_url": latest.annotated_media_url if latest else None,
        "total_monitored_events": total_assessments,
        "total_hazards_detected": len(all_hazards),
        "severity_distribution": severity_totals,
        "detection_class_distribution": class_distribution,
        "recent_assessments": [
            {
                "id": r.id,
                "source": r.source_filename,
                "score": r.site_risk_score,
                "level": r.risk_level,
                "hazards_count": r.total_hazards_count,
                "timestamp": r.created_at.strftime("%H:%M:%S") if r.created_at else ""
            } for r in recent
        ]
    }
