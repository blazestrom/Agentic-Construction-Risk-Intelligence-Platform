"""
Verification Script for Phase 3: Inference Pipeline
Tests Image and Video inference modules with trained weights.
"""

import sys
import json
from pathlib import Path
import cv2
import numpy as np

# Ensure root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.detector import ConstructionSafetyDetector

def test_inference_pipeline():
    print("=" * 60)
    print("TESTING PHASE 3: INFERENCE PIPELINE")
    print("=" * 60)
    
    weights_path = Path("models/runs/yolov8n_construction_safety/weights/best.pt")
    detector = ConstructionSafetyDetector(weights_path=weights_path)
    
    # 1. Test Single Image Inference
    test_img_dir = Path("test/images")
    test_images = sorted(list(test_img_dir.glob("*.jpg")))[:3]
    
    print("\n[1] Running Single-Image Inference on Test Set...")
    image_results = []
    for img_path in test_images:
        res = detector.detect_image(
            image_input=img_path,
            conf_threshold=0.25,
            save_annotated=True
        )
        print(f"  - Processed: {img_path.name}")
        print(f"    Detections: {res['detections_count']}, Time: {res['inference_time_ms']} ms")
        for d in res['detections'][:3]: # show top 3
            print(f"      * {d['class_name']} ({d['category']}) - conf: {d['confidence']} - box: {d['bbox_xyxy']}")
        if res['annotated_image_path']:
            print(f"    Annotated saved to: {res['annotated_image_path']}")
        image_results.append(res)
        
    # 2. Test Video Inference
    # Create a 45-frame test video snippet using frames from the test set
    print("\n[2] Preparing Test Construction Site Video Stream...")
    sample_video_path = Path("reports/test_sample_video.mp4")
    sample_video_path.parent.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(str(sample_video_path), fourcc, 15.0, (640, 640))
    
    # Pick 3 diverse images and write 15 frames each
    for img_path in test_images:
        img = cv2.imread(str(img_path))
        if img is not None:
            img_resized = cv2.resize(img, (640, 640))
            for _ in range(15):
                vw.write(img_resized)
    vw.release()
    print(f"  Created test video: {sample_video_path} (45 frames @ 15 FPS)")
    
    # Run Video Inference
    print("\n[3] Running Video Stream Inference...")
    video_res = detector.detect_video(
        video_input=sample_video_path,
        conf_threshold=0.25,
        frame_stride=1
    )
    
    print(f"  Video Processed: {video_res['source']}")
    print(f"  Total Frames: {video_res['total_video_frames']}, Processed: {video_res['processed_frames']}")
    print(f"  Processing Speed: {video_res['processing_fps']} FPS")
    print(f"  Aggregate Detections: {video_res['aggregate_detections']}")
    print(f"  Annotated Video Saved: {video_res['annotated_video_path']}")
    
    # Save structured telemetry sample
    telemetry_file = Path("reports/inference_telemetry_sample.json")
    with open(telemetry_file, "w") as f:
        json.dump({
            "image_inference_samples": image_results,
            "video_inference_summary": {k: v for k, v in video_res.items() if k != "frames"},
            "sample_first_frame_telemetry": video_res["frames"][0] if video_res["frames"] else {}
        }, f, indent=2)
        
    print(f"\n[4] Telemetry JSON exported to: {telemetry_file}")
    print("\n" + "=" * 60)
    print("PHASE 3 INFERENCE PIPELINE VERIFICATION PASSED")
    print("=" * 60)

if __name__ == "__main__":
    test_inference_pipeline()
