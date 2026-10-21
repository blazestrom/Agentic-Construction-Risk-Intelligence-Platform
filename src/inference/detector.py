"""
Inference Pipeline: Image and Video Object Detection
Agentic Construction Risk Intelligence Platform
"""

import os
import sys
import time
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Tuple
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Category taxonomy for downstream Site Risk Agent
CATEGORY_TAXONOMY = {
    "Hardhat": "ppe_compliant",
    "Safety Vest": "ppe_compliant",
    "Gloves": "ppe_compliant",
    "Mask": "ppe_compliant",
    "NO-Hardhat": "ppe_violation",
    "NO-Safety Vest": "ppe_violation",
    "NO-Mask": "ppe_violation",
    "Person": "worker",
    "Excavator": "heavy_machinery",
    "wheel loader": "heavy_machinery",
    "dump truck": "heavy_machinery",
    "machinery": "heavy_machinery",
    "Safety Cone": "hazard_marker",
    "Ladder": "equipment",
    "fire hydrant": "infrastructure",
    "SUV": "vehicle",
    "bus": "vehicle",
    "mini-van": "vehicle",
    "sedan": "vehicle",
    "semi": "vehicle",
    "trailer": "vehicle",
    "truck and trailer": "vehicle",
    "truck": "vehicle",
    "van": "vehicle",
    "vehicle": "vehicle"
}

# Color palette (BGR for OpenCV)
CATEGORY_COLORS = {
    "ppe_violation": (0, 0, 230),     # Bright Red
    "ppe_compliant": (0, 180, 0),     # Green
    "worker": (255, 180, 0),          # Bright Cyan/Yellow
    "heavy_machinery": (0, 120, 255), # Orange
    "hazard_marker": (0, 220, 220),   # Yellow
    "equipment": (180, 100, 255),     # Purple
    "vehicle": (150, 150, 150),       # Gray
    "infrastructure": (100, 200, 100)
}

class ConstructionSafetyDetector:
    """
    Reusable detector module for image and video inference using fine-tuned YOLOv8 weights.
    """
    
    DEFAULT_WEIGHTS = Path("models/runs/yolov8n_construction_safety/weights/best.pt")
    
    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        device: str = "auto"
    ):
        if weights_path is None:
            weights_path = self.DEFAULT_WEIGHTS
            
        self.weights_path = Path(weights_path).resolve()
        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at: {self.weights_path}. "
                "Ensure training completed or provide a valid checkpoint path."
            )
            
        if device == "auto":
            self.device = "0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"[Detector] Loading YOLOv8 model from {self.weights_path} on device '{self.device}'...")
        self.model = YOLO(str(self.weights_path))
        self.class_names = self.model.names if hasattr(self.model, "names") else {}
        print(f"[Detector] Successfully initialized. Loaded {len(self.class_names)} classes.")

    def _format_detection(self, box, class_id: int, conf: float, img_w: int, img_h: int) -> Dict[str, Any]:
        """Convert a single raw bounding box into structured JSON format."""
        x1, y1, x2, y2 = [float(v) for v in box]
        class_name = self.class_names.get(class_id, f"class_{class_id}")
        category = CATEGORY_TAXONOMY.get(class_name, "general_object")
        
        # Normalized coordinates [0, 1]
        xc = round(((x1 + x2) / 2.0) / img_w, 4)
        yc = round(((y1 + y2) / 2.0) / img_h, 4)
        bw = round((x2 - x1) / img_w, 4)
        bh = round((y2 - y1) / img_h, 4)
        
        return {
            "class_id": class_id,
            "class_name": class_name,
            "category": category,
            "confidence": round(conf, 4),
            "bbox_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "bbox_normalized": [xc, yc, bw, bh],
            "area_pixels": round((x2 - x1) * (y2 - y1), 1)
        }

    def annotate_frame(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Render clean, color-coded bounding boxes and labels onto a cv2 frame."""
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
            cname = det["class_name"]
            conf = det["confidence"]
            cat = det["category"]
            color = CATEGORY_COLORS.get(cat, (200, 200, 200))
            
            # Draw bounding box
            thickness = 2 if cat in ["ppe_violation", "heavy_machinery"] else 1
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            # Label banner
            label = f"{cname} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, max(th + 6, y1)), color, -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 2, max(th, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255) if cat in ["ppe_violation", "heavy_machinery"] else (0, 0, 0),
                1,
                cv2.LINE_AA
            )
        return annotated

    def detect_image(
        self,
        image_input: Union[str, Path, np.ndarray],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        save_annotated: bool = False,
        output_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Run inference on a single image.
        Returns structured detections and execution metrics.
        """
        t0 = time.time()
        
        # Load image
        if isinstance(image_input, (str, Path)):
            img_path = Path(image_input)
            if not img_path.exists():
                raise FileNotFoundError(f"Input image not found: {img_path}")
            frame = cv2.imread(str(img_path))
            if frame is None:
                raise ValueError(f"Could not decode image at: {img_path}")
            source_name = img_path.name
        elif isinstance(image_input, np.ndarray):
            frame = image_input
            source_name = "in_memory_array"
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")
            
        h, w = frame.shape[:2]
        
        # Run YOLO inference
        results = self.model.predict(
            source=frame,
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device,
            verbose=False
        )
        
        detections: List[Dict[str, Any]] = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cid = int(boxes.cls[i].cpu().numpy())
                det = self._format_detection(xyxy, cid, conf, w, h)
                detections.append(det)
                
        inference_time_ms = round((time.time() - t0) * 1000.0, 2)
        
        annotated_path_str = None
        if save_annotated:
            annotated_frame = self.annotate_frame(frame, detections)
            if output_path is None:
                out_dir = Path("reports/inference_outputs")
                out_dir.mkdir(parents=True, exist_ok=True)
                out_file = out_dir / f"detected_{source_name}"
            else:
                out_file = Path(output_path)
                out_file.parent.mkdir(parents=True, exist_ok=True)
                
            cv2.imwrite(str(out_file), annotated_frame)
            annotated_path_str = str(out_file)
            
        return {
            "source": source_name,
            "image_width": w,
            "image_height": h,
            "detections_count": len(detections),
            "detections": detections,
            "inference_time_ms": inference_time_ms,
            "annotated_image_path": annotated_path_str
        }

    def detect_video(
        self,
        video_input: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        frame_stride: int = 1,
        max_frames: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run inference on video input, writing an annotated video and returning frame-by-frame JSON telemetry.
        """
        video_path = Path(video_input)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video stream: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer
        if output_path is None:
            out_dir = Path("reports/inference_outputs")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_video_file = out_dir / f"annotated_{video_path.name}"
        else:
            out_video_file = Path(output_path)
            out_video_file.parent.mkdir(parents=True, exist_ok=True)
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(str(out_video_file), fourcc, fps / max(1, frame_stride), (w, h))
        
        frames_telemetry = []
        aggregate_counts: Dict[str, int] = {}
        frame_idx = 0
        processed_count = 0
        t_start = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_idx % frame_stride == 0:
                    timestamp_ms = round((frame_idx / fps) * 1000.0, 1)
                    
                    # Predict
                    results = self.model.predict(
                        source=frame,
                        conf=conf_threshold,
                        iou=iou_threshold,
                        device=self.device,
                        verbose=False
                    )
                    
                    frame_detections = []
                    if len(results) > 0 and results[0].boxes is not None:
                        boxes = results[0].boxes
                        for i in range(len(boxes)):
                            xyxy = boxes.xyxy[i].cpu().numpy()
                            conf = float(boxes.conf[i].cpu().numpy())
                            cid = int(boxes.cls[i].cpu().numpy())
                            det = self._format_detection(xyxy, cid, conf, w, h)
                            frame_detections.append(det)
                            aggregate_counts[det["class_name"]] = aggregate_counts.get(det["class_name"], 0) + 1
                            
                    annotated_frame = self.annotate_frame(frame, frame_detections)
                    out_writer.write(annotated_frame)
                    
                    frames_telemetry.append({
                        "frame_index": frame_idx,
                        "timestamp_ms": timestamp_ms,
                        "detections_count": len(frame_detections),
                        "detections": frame_detections
                    })
                    
                    processed_count += 1
                    if max_frames and processed_count >= max_frames:
                        break
                        
                frame_idx += 1
        finally:
            cap.release()
            out_writer.release()
            
        elapsed_total = round(time.time() - t_start, 2)
        fps_real = round(processed_count / elapsed_total, 2) if elapsed_total > 0 else 0
        
        return {
            "source": video_path.name,
            "total_video_frames": total_video_frames,
            "processed_frames": processed_count,
            "frame_stride": frame_stride,
            "video_fps": round(fps, 2),
            "processing_fps": fps_real,
            "total_processing_seconds": elapsed_total,
            "aggregate_detections": aggregate_counts,
            "annotated_video_path": str(out_video_file),
            "frames": frames_telemetry
        }
