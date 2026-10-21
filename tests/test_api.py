"""
Test Suite for FastAPI Endpoints and Risk Agent Workflow
"""

import sys
from pathlib import Path
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.api.main import app

client = TestClient(app)

def test_serve_dashboard():
    res = client.get("/")
    assert res.status_code == 200
    assert "Construction Risk Intelligence Platform" in res.text

def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert data["model_loaded"] is True

def test_model_info():
    res = client.get("/model/info")
    assert res.status_code == 200
    data = res.json()
    assert data["total_classes"] == 25
    assert "Person" in data["classes"].values() or "Person" in data["classes"]

def test_risk_assess_image():
    # Pick a sample image from test set
    test_img = next(Path("test/images").glob("*.jpg"))
    with open(test_img, "rb") as f:
        res = client.post(
            "/risk/assess",
            files={"file": (test_img.name, f, "image/jpeg")},
            data={"conf_threshold": "0.25"}
        )
    assert res.status_code == 200
    data = res.json()
    assert "site_risk_score" in data
    assert "risk_level" in data
    assert "executive_summary" in data
    assert "hazards" in data
    print(f"Test Image Risk Score: {data['site_risk_score']} ({data['risk_level']})")

def test_dashboard_summary():
    res = client.get("/dashboard/summary")
    assert res.status_code == 200
    data = res.json()
    assert "current_risk_score" in data
    assert "severity_distribution" in data

def test_list_risks():
    res = client.get("/risks?limit=5")
    assert res.status_code == 200
    data = res.json()
    assert "assessments" in data
