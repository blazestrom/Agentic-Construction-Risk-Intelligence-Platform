"""
Phase 2: YOLOv8 Training & Evaluation Pipeline
Agentic Construction Risk Intelligence Platform
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

def detect_device(requested_device: str = None) -> str:
    """
    Selects GPU if CUDA is available and requested, otherwise falls back to CPU.
    """
    if requested_device is not None and requested_device != 'auto':
        return requested_device
        
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"[Device Selection] CUDA GPU detected: {device_name}. Using device='0'")
        return '0'
    else:
        print("[Device Selection] No compatible CUDA PyTorch installation detected. Graceful fallback to device='cpu'.")
        return 'cpu'

def run_training(
    model_name: str = "yolov8n.pt",
    data_yaml: str = "configs/dataset_verified.yaml",
    epochs: int = 25,
    imgsz: int = 640,
    batch: int = 16,
    workers: int = 2,
    device: str = "auto",
    seed: int = 42,
    project: str = "models/runs",
    run_name: str = "yolov8n_construction_safety"
):
    start_time = time.time()
    workspace_root = Path.cwd().resolve()
    data_path = (workspace_root / data_yaml).resolve()
    project_path = (workspace_root / project).resolve()
    
    selected_device = detect_device(device)
    
    print("\n" + "="*60)
    print("STARTING YOLOV8 MODEL TRAINING")
    print("="*60)
    print(f"  Base Model     : {model_name}")
    print(f"  Dataset Config : {data_path}")
    print(f"  Epochs         : {epochs}")
    print(f"  Image Size     : {imgsz}")
    print(f"  Batch Size     : {batch}")
    print(f"  Workers        : {workers}")
    print(f"  Device         : {selected_device}")
    print(f"  Random Seed    : {seed}")
    print(f"  Output Project : {project_path}")
    print(f"  Run Name       : {run_name}")
    print("="*60 + "\n")
    
    # 1. Initialize YOLO baseline
    model = YOLO(model_name)
    
    # 2. Train model
    train_results = model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        device=selected_device,
        seed=seed,
        project=str(project_path),
        name=run_name,
        exist_ok=True,
        save=True,
        verbose=True
    )
    
    train_duration = time.time() - start_time
    print(f"\n[Training Finished] Total time elapsed: {train_duration/60:.2f} minutes")
    
    # Check save paths
    run_dir = project_path / run_name
    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"
    
    print(f"  Best checkpoint : {best_pt} (Exists: {best_pt.exists()})")
    print(f"  Last checkpoint : {last_pt} (Exists: {last_pt.exists()})")
    
    if not best_pt.exists():
        print("[WARNING] best.pt not found, evaluating with current model in memory")
        eval_model = model
    else:
        eval_model = YOLO(str(best_pt))
        
    # 3. Evaluate on Validation Split
    print("\n" + "-"*60)
    print("EVALUATING BEST CHECKPOINT ON VALIDATION SPLIT")
    print("-"*60)
    val_metrics = eval_model.val(
        data=str(data_path),
        split="val",
        imgsz=imgsz,
        batch=batch,
        device=selected_device,
        project=str(project_path),
        name=f"{run_name}_val",
        exist_ok=True
    )
    
    # 4. Evaluate on Held-out Test Split
    print("\n" + "-"*60)
    print("EVALUATING BEST CHECKPOINT ON TEST SPLIT")
    print("-"*60)
    test_metrics = eval_model.val(
        data=str(data_path),
        split="test",
        imgsz=imgsz,
        batch=batch,
        device=selected_device,
        project=str(project_path),
        name=f"{run_name}_test",
        exist_ok=True
    )
    
    # 5. Extract structured metrics report
    def extract_summary(metrics_obj, split_name):
        box = metrics_obj.box
        p = float(box.mp) if hasattr(box, 'mp') else 0.0
        r = float(box.mr) if hasattr(box, 'mr') else 0.0
        map50 = float(box.map50) if hasattr(box, 'map50') else 0.0
        map50_95 = float(box.map) if hasattr(box, 'map') else 0.0
        
        per_class = {}
        if hasattr(metrics_obj, 'names') and hasattr(box, 'maps') and box.maps is not None:
            for cid, cname in metrics_obj.names.items():
                if cid < len(box.maps):
                    per_class[cname] = {
                        "mAP50_95": round(float(box.maps[cid]), 4)
                    }
                    
        return {
            "split": split_name,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "mAP50": round(map50, 4),
            "mAP50_95": round(map50_95, 4),
            "per_class": per_class
        }
        
    val_summary = extract_summary(val_metrics, "val")
    test_summary = extract_summary(test_metrics, "test")
    
    full_report = {
        "model": model_name,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": selected_device,
        "train_duration_minutes": round(train_duration / 60.0, 2),
        "checkpoints": {
            "best": str(best_pt),
            "last": str(last_pt)
        },
        "validation_metrics": val_summary,
        "test_metrics": test_summary
    }
    
    # Save structured metrics
    report_file = run_dir / "evaluation_report.json"
    with open(report_file, "w") as f:
        json.dump(full_report, f, indent=2)
        
    print("\n" + "="*60)
    print("TRAINING & EVALUATION SUMMARY")
    print("="*60)
    print(f"Validation Precision : {val_summary['precision']:.4f}")
    print(f"Validation Recall    : {val_summary['recall']:.4f}")
    print(f"Validation mAP@50    : {val_summary['mAP50']:.4f}")
    print(f"Validation mAP@50-95 : {val_summary['mAP50_95']:.4f}")
    print("-" * 60)
    print(f"Test Precision       : {test_summary['precision']:.4f}")
    print(f"Test Recall          : {test_summary['recall']:.4f}")
    print(f"Test mAP@50          : {test_summary['mAP50']:.4f}")
    print(f"Test mAP@50-95       : {test_summary['mAP50_95']:.4f}")
    print(f"\nFull report saved to : {report_file}")
    print("="*60)
    
    return full_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Construction Safety Dataset")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model weights")
    parser.add_argument("--data", type=str, default="configs/dataset_verified.yaml", help="Path to dataset yaml")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--workers", type=int, default=2, help="Dataloader workers")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--project", type=str, default="models/runs", help="Project output folder")
    parser.add_argument("--name", type=str, default="yolov8n_construction_safety", help="Run name")
    
    args = parser.parse_args()
    run_training(
        model_name=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        seed=args.seed,
        project=args.project,
        run_name=args.name
    )
