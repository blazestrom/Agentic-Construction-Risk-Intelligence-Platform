"""
SQLAlchemy Database Models
Agentic Construction Risk Intelligence Platform
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, Float, DateTime, JSON, ForeignKey, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

def generate_uuid() -> str:
    return str(uuid.uuid4())

class SiteRecord(Base):
    __tablename__ = "sites"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, nullable=False, default="Primary Construction Site")
    location = Column(String, nullable=True, default="Sector 4 - Operational Yard")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    assessments = relationship("RiskAssessmentRecord", back_populates="site", cascade="all, delete-orphan")

class RiskAssessmentRecord(Base):
    __tablename__ = "risk_assessments"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    site_id = Column(String, ForeignKey("sites.id"), nullable=True)
    source_filename = Column(String, nullable=False)
    media_type = Column(String, nullable=False) # "image" or "video"
    site_risk_score = Column(Integer, nullable=False)
    risk_level = Column(String, nullable=False) # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    total_hazards_count = Column(Integer, default=0)
    severity_breakdown = Column(JSON, default=dict)
    annotated_media_url = Column(String, nullable=True)
    inference_time_ms = Column(Float, default=0.0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    site = relationship("SiteRecord", back_populates="assessments")
    hazards = relationship("HazardRecord", back_populates="assessment", cascade="all, delete-orphan")
    detections = relationship("DetectionRecord", back_populates="assessment", cascade="all, delete-orphan")

class HazardRecord(Base):
    __tablename__ = "hazards"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    assessment_id = Column(String, ForeignKey("risk_assessments.id"), nullable=False)
    hazard_code = Column(String, nullable=False)
    rule_id = Column(String, nullable=False)
    rule_name = Column(String, nullable=False)
    severity = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    observed_evidence = Column(Text, nullable=False)
    rationale = Column(Text, nullable=False)
    recommended_action = Column(Text, nullable=False)
    affected_entity = Column(String, nullable=True)
    spatial_context = Column(String, nullable=True)
    primary_box = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    assessment = relationship("RiskAssessmentRecord", back_populates="hazards")

class DetectionRecord(Base):
    __tablename__ = "detections"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    assessment_id = Column(String, ForeignKey("risk_assessments.id"), nullable=False)
    class_name = Column(String, nullable=False)
    category = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    bbox_xyxy = Column(JSON, nullable=False)
    timestamp_ms = Column(Float, nullable=True)
    
    assessment = relationship("RiskAssessmentRecord", back_populates="detections")
