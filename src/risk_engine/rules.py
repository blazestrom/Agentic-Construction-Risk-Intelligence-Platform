"""
Rule definitions and configuration for the Site Risk Engine.
Agentic Construction Risk Intelligence Platform
"""

from typing import Dict, Any, List
from pydantic import BaseModel, Field

class RiskRuleConfig(BaseModel):
    rule_id: str
    rule_name: str
    severity: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    base_weight: float
    min_confidence: float = 0.30
    description: str
    action: str

# Documented deterministic risk rules based strictly on verified dataset classes
DEFAULT_RISK_RULES = {
    "R01_NO_HARDHAT": RiskRuleConfig(
        rule_id="R01_NO_HARDHAT",
        rule_name="Unprotected Head (Missing Hardhat)",
        severity="HIGH",
        base_weight=25.0,
        min_confidence=0.30,
        description="Worker observed on site without mandatory hardhat / head protection.",
        action="Instruct worker to immediately don an OSHA/ANSI-compliant hardhat before proceeding."
    ),
    "R02_NO_VEST": RiskRuleConfig(
        rule_id="R02_NO_VEST",
        rule_name="Low Visibility (Missing Safety Vest)",
        severity="MEDIUM",
        base_weight=15.0,
        min_confidence=0.30,
        description="Worker detected without high-visibility safety vest in operational area.",
        action="Supply high-visibility reflective vest to personnel to prevent vehicular strike hazards."
    ),
    "R03_MACHINERY_PROXIMITY": RiskRuleConfig(
        rule_id="R03_MACHINERY_PROXIMITY",
        rule_name="Dangerous Machinery Proximity",
        severity="HIGH",
        base_weight=30.0,
        min_confidence=0.35,
        description="Worker located within close proximity of heavy machinery (Excavator/Loader/Truck).",
        action="Signal operator, sound warning beacon, and enforce minimum safe approach boundary (5 meters)."
    ),
    "R04_COMPOUND_PROXIMITY_NO_PPE": RiskRuleConfig(
        rule_id="R04_COMPOUND_PROXIMITY_NO_PPE",
        rule_name="Critical Proximity with PPE Non-Compliance",
        severity="CRITICAL",
        base_weight=45.0,
        min_confidence=0.35,
        description="Worker lacking required PPE (hardhat or vest) in close proximity to active heavy equipment.",
        action="IMMEDIATE STOP WORK: Clear worker from machinery blind spot immediately and conduct safety briefing."
    ),
    "R05_LADDER_ZONE": RiskRuleConfig(
        rule_id="R05_LADDER_ZONE",
        rule_name="Elevated Fall Risk Zone",
        severity="MEDIUM",
        base_weight=15.0,
        min_confidence=0.30,
        description="Person operating near ladder structure without confirmed fall protection.",
        action="Verify 3-point contact, ladder footing stability, and tethering if working above 6 feet."
    ),
    "R06_UNCONED_EQUIPMENT": RiskRuleConfig(
        rule_id="R06_UNCONED_EQUIPMENT",
        rule_name="Demarcation Absence Around Heavy Equipment",
        severity="LOW",
        base_weight=10.0,
        min_confidence=0.30,
        description="Heavy machinery stationed without protective safety cones or exclusion barrier.",
        action="Place safety cones around machinery swing radius to designate active swing zone."
    )
}

SEVERITY_LEVELS = {
    "LOW": {"weight_multiplier": 1.0, "color": "green"},
    "MEDIUM": {"weight_multiplier": 1.2, "color": "yellow"},
    "HIGH": {"weight_multiplier": 1.5, "color": "orange"},
    "CRITICAL": {"weight_multiplier": 2.0, "color": "red"}
}
