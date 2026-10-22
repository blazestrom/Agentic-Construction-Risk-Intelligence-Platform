"""
Terminal CLI Tool for Construction Risk Assessment
Allows running detection and explainable risk analysis directly in terminal.
"""

import sys
import argparse
from pathlib import Path

# Ensure root in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.inference.detector import ConstructionSafetyDetector
from src.risk_engine.evaluator import SiteRiskEvaluator
from src.agents.workflow import SiteRiskAgenticWorkflow

def run_terminal_inspection(image_path: str = None, conf: float = 0.25):
    print("=" * 65)
    print(" CONSTRUCTION RISK INTELLIGENCE PLATFORM (TERMINAL CLI)")
    print("=" * 65)
    
    detector = ConstructionSafetyDetector()
    evaluator = SiteRiskEvaluator()
    workflow = SiteRiskAgenticWorkflow(detector=detector, evaluator=evaluator)
    
    if image_path is None:
        # Default to first test sample
        sample_images = list(Path("test/images").glob("*.jpg"))
        if not sample_images:
            print("No test images found in test/images.")
            return
        img_target = sample_images[0]
        print(f"\n[Info] No image specified. Using sample: {img_target.name}")
    else:
        img_target = Path(image_path)
        if not img_target.exists():
            print(f"Error: Image not found at {img_target}")
            return
            
    print(f"\nRunning AI detection and safety rules on: {img_target}...")
    result = workflow.process_image_assessment(
        image_input=img_target,
        source_filename=img_target.name,
        conf_threshold=conf,
        save_annotated=True
    )
    
    print("\n" + "-" * 65)
    print("INSPECTION RESULTS")
    print("-" * 65)
    print(f"  Source File      : {result['source_filename']}")
    print(f"  Inference Latency: {result['timing_ms']['inference_time']} ms")
    print(f"  Total Detections : {result['detections_count']}")
    print(f"  Site Risk Score  : {result['site_risk_score']} / 100")
    print(f"  Risk Level       : {result['risk_level']}")
    print(f"  Alert Status     : {'⚠️ ACTIVE (DISPATCH SUPERVISOR)' if result['alert']['alert_active'] else '✓ NORMAL'}")
    
    print("\n[Detected Objects in Scene]:")
    if result['detections']:
        for d in result['detections']:
            print(f"  - {d['class_name']:<18} ({d['category']:<15}) | Conf: {d['confidence']*100:.1f}% | Box: {d['bbox_xyxy']}")
    else:
        print("  - No objects detected at current confidence threshold.")
        
    print("\n[Identified Safety Hazards]:")
    if result['hazards']:
        for idx, h in enumerate(result['hazards'], 1):
            print(f"  #{idx} [{h['severity']}] {h['rule_name']}")
            print(f"      Evidence : {h['observed_evidence']}")
            print(f"      Action   : {h['recommended_action']}")
    else:
        print("  ✓ Compliant operations. Zero hazardous conditions detected.")
        
    print("\n[Executive Explanation]:")
    print("  " + result['executive_summary'].replace("\n", "\n  "))
    
    if result.get('annotated_media_url'):
        print(f"\n[Annotated Image Saved]: reports/inference_outputs/{Path(result['annotated_media_url']).name}")
        
    print("=" * 65 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Terminal CLI for Construction Risk Assessment")
    parser.add_argument("--image", type=str, default=None, help="Path to site image")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default 0.25)")
    args = parser.parse_args()
    
    run_terminal_inspection(image_path=args.image, conf=args.conf)
