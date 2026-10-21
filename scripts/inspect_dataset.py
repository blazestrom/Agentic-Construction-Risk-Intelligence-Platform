"""
Phase 1: Dataset Inspection & Validation Script
Agentic Construction Risk Intelligence Platform
"""

import os
import sys
import yaml
import hashlib
from pathlib import Path
from collections import defaultdict, Counter
import cv2
import numpy as np
from PIL import Image

def compute_file_hash(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def inspect_dataset(root_dir: str):
    root_path = Path(root_dir).resolve()
    yaml_path = root_path / "data.yaml"
    
    print(f"==================================================")
    print(f"DATASET INSPECTION & VALIDATION REPORT")
    print(f"Root Directory: {root_path}")
    print(f"YAML Path: {yaml_path}")
    print(f"==================================================\n")
    
    if not yaml_path.exists():
        print(f"ERROR: data.yaml not found at {yaml_path}")
        return
        
    with open(yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
        
    print("[1] YAML Configuration:")
    for k, v in yaml_data.items():
        if k != 'names':
            print(f"  {k}: {v}")
    
    classes = yaml_data.get('names', [])
    num_classes = yaml_data.get('nc', len(classes))
    print(f"  nc (declared): {num_classes}")
    print(f"  names count: {len(classes)}")
    print(f"  Class list:")
    for idx, name in enumerate(classes):
        print(f"    [{idx:2d}] {name}")
    print()
    
    # Check directory paths
    splits = ['train', 'valid', 'test']
    split_stats = {}
    image_hashes = defaultdict(list) # hash -> list of (split, filename)
    class_counts_by_split = {split: Counter() for split in splits}
    total_boxes_by_split = Counter()
    
    corrupt_images = []
    malformed_labels = []
    out_of_bounds_boxes = []
    missing_labels = []
    unannotated_images = []
    image_resolutions = []
    
    for split in splits:
        img_dir = root_path / split / "images"
        lbl_dir = root_path / split / "labels"
        
        if not img_dir.exists():
            print(f"WARNING: Image directory not found: {img_dir}")
            continue
            
        img_files = sorted(list(img_dir.glob("*.*")))
        # Filter for standard image extensions
        img_files = [f for f in img_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']]
        lbl_files = sorted(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else []
        
        lbl_dict = {f.stem: f for f in lbl_files}
        
        split_stats[split] = {
            "image_count": len(img_files),
            "label_file_count": len(lbl_files),
            "img_dir": img_dir,
            "lbl_dir": lbl_dir
        }
        
        for img_path in img_files:
            # Check corrupt image
            try:
                with Image.open(img_path) as img:
                    img.verify()
                # Read dimensions with cv2
                img_cv = cv2.imread(str(img_path))
                if img_cv is None:
                    corrupt_images.append(str(img_path))
                    continue
                h, w, c = img_cv.shape
                image_resolutions.append((w, h))
            except Exception as e:
                corrupt_images.append(f"{img_path}: {e}")
                continue
                
            # Check duplicates/leakage
            f_hash = compute_file_hash(img_path)
            image_hashes[f_hash].append((split, img_path.name))
            
            # Check label correspondence
            lbl_path = lbl_dict.get(img_path.stem)
            if lbl_path is None:
                missing_labels.append((split, img_path.name))
                continue
                
            # Parse label file
            try:
                with open(lbl_path, 'r') as lf:
                    lines = [line.strip() for line in lf.readlines() if line.strip()]
                    
                if len(lines) == 0:
                    unannotated_images.append((split, img_path.name))
                    continue
                    
                for line_idx, line in enumerate(lines):
                    parts = line.split()
                    if len(parts) != 5:
                        malformed_labels.append((split, lbl_path.name, line_idx+1, f"Expected 5 tokens, got {len(parts)}: '{line}'"))
                        continue
                    try:
                        cls_id = int(parts[0])
                        xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    except ValueError as ve:
                        malformed_labels.append((split, lbl_path.name, line_idx+1, f"Parse error: {ve}"))
                        continue
                        
                    if cls_id < 0 or cls_id >= num_classes:
                        malformed_labels.append((split, lbl_path.name, line_idx+1, f"Class ID {cls_id} out of range [0, {num_classes-1}]"))
                        continue
                        
                    if bw <= 0 or bh <= 0:
                        malformed_labels.append((split, lbl_path.name, line_idx+1, f"Zero or negative box dimensions: w={bw}, h={bh}"))
                        continue
                        
                    # Check bounding box bounds
                    x1 = xc - bw / 2.0
                    y1 = yc - bh / 2.0
                    x2 = xc + bw / 2.0
                    y2 = yc + bh / 2.0
                    
                    if x1 < -0.05 or y1 < -0.05 or x2 > 1.05 or y2 > 1.05:
                        out_of_bounds_boxes.append((split, lbl_path.name, line_idx+1, f"Box severely out of bounds: [{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}]"))
                        
                    class_counts_by_split[split][cls_id] += 1
                    total_boxes_by_split[split] += 1
            except Exception as e:
                malformed_labels.append((split, lbl_path.name, 0, f"Read error: {e}"))

    print("[2] Split Summary:")
    total_imgs = sum(s["image_count"] for s in split_stats.values())
    total_boxes = sum(total_boxes_by_split.values())
    for split, s in split_stats.items():
        pct = (s["image_count"] / total_imgs * 100) if total_imgs > 0 else 0
        boxes = total_boxes_by_split[split]
        avg_b = (boxes / s["image_count"]) if s["image_count"] > 0 else 0
        print(f"  {split.upper():6s}: {s['image_count']:5d} images ({pct:5.1f}%), {s['label_file_count']:5d} label files, {boxes:6d} bounding boxes (avg {avg_b:.2f}/img)")
    print(f"  TOTAL : {total_imgs:5d} images, {total_boxes:6d} bounding boxes across all splits.\n")
    
    print("[3] Data Integrity & Anomaly Checks:")
    print(f"  - Corrupt or unreadable images: {len(corrupt_images)}")
    if corrupt_images:
        for c in corrupt_images[:5]:
            print(f"      * {c}")
    print(f"  - Missing label files (image exists, no .txt): {len(missing_labels)}")
    if missing_labels:
        for m in missing_labels[:5]:
            print(f"      * {m}")
    print(f"  - Empty label files (background images / zero objects): {len(unannotated_images)}")
    print(f"  - Malformed label lines: {len(malformed_labels)}")
    if malformed_labels:
        for m in malformed_labels[:5]:
            print(f"      * {m}")
    print(f"  - Severely out-of-bounds boxes (> 5% overflow): {len(out_of_bounds_boxes)}")
    if out_of_bounds_boxes:
        for o in out_of_bounds_boxes[:5]:
            print(f"      * {o}")
    print()
    
    # Check duplicate images and data leakage
    duplicates_within_split = 0
    leakage_across_splits = []
    for f_hash, occurrences in image_hashes.items():
        if len(occurrences) > 1:
            split_set = set(occ[0] for occ in occurrences)
            if len(split_set) > 1:
                leakage_across_splits.append(occurrences)
            else:
                duplicates_within_split += len(occurrences) - 1
                
    print("[4] Leakage & Duplicate Analysis:")
    print(f"  - Exact duplicate images within same split: {duplicates_within_split}")
    print(f"  - Data leakage across splits (identical image in multiple splits): {len(leakage_across_splits)}")
    if leakage_across_splits:
        for leak in leakage_across_splits[:5]:
            print(f"      * Leaked across: {leak}")
    print()
    
    # Image resolutions
    if image_resolutions:
        widths, heights = zip(*image_resolutions)
        print("[5] Image Resolution Characteristics:")
        print(f"  - Min resolution: {min(widths)}x{min(heights)}")
        print(f"  - Max resolution: {max(widths)}x{max(heights)}")
        print(f"  - Median/Most Common: {Counter(image_resolutions).most_common(3)}")
        print()
        
    # Class Distribution Table
    print("[6] Class Distribution by Split:")
    header = f"  {'ID':2s} | {'Class Name':20s} | {'Train':6s} | {'Valid':6s} | {'Test':6s} | {'Total':6s} | {'Train %':7s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    
    overall_class_counts = Counter()
    for split in splits:
        for cid, count in class_counts_by_split[split].items():
            overall_class_counts[cid] += count
            
    for cid in range(num_classes):
        cname = classes[cid] if cid < len(classes) else f"Unknown-{cid}"
        tr_c = class_counts_by_split['train'][cid]
        va_c = class_counts_by_split['valid'][cid]
        te_c = class_counts_by_split['test'][cid]
        tot_c = overall_class_counts[cid]
        pct = (tr_c / total_boxes_by_split['train'] * 100) if total_boxes_by_split['train'] > 0 else 0
        print(f"  {cid:2d} | {cname:20s} | {tr_c:6d} | {va_c:6d} | {te_c:6d} | {tot_c:6d} | {pct:6.2f}%")
        
    print()
    print("==================================================")
    print("END OF INSPECTION REPORT")
    print("==================================================")

if __name__ == "__main__":
    inspect_dataset(".")
