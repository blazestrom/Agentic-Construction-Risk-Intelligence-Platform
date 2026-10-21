"""
Visualize representative annotated images with YOLO ground truth bounding boxes.
"""

import os
import cv2
import yaml
import random
from pathlib import Path

def draw_annotations(root_dir: str, output_dir: str, num_samples: int = 10):
    root_path = Path(root_dir).resolve()
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    
    with open(root_path / "data.yaml", 'r') as f:
        yaml_data = yaml.safe_load(f)
    classes = yaml_data.get('names', [])
    
    # Palette of colors for classes
    random.seed(42)
    colors = {}
    for i, name in enumerate(classes):
        if "NO-" in name:
            colors[i] = (0, 0, 230)  # Bright Red for violations
        elif "Hardhat" in name or "Vest" in name or "Gloves" in name or "Mask" in name:
            colors[i] = (0, 200, 0)  # Green for compliant PPE
        elif name in ["Excavator", "wheel loader", "dump truck", "machinery"]:
            colors[i] = (0, 140, 255) # Orange for heavy machinery
        elif name == "Person":
            colors[i] = (255, 200, 0) # Cyan/Yellow for Person
        else:
            colors[i] = (200, 200, 200) # Gray
            
    # Select images that contain interesting classes (e.g. NO-Hardhat, NO-Safety Vest, Excavator, Person)
    splits = ['train', 'valid']
    candidate_images = []
    
    for split in splits:
        img_dir = root_path / split / "images"
        lbl_dir = root_path / split / "labels"
        
        for lbl_file in lbl_dir.glob("*.txt"):
            with open(lbl_file, 'r') as lf:
                lines = [line.strip() for line in lf.readlines() if line.strip()]
            if not lines:
                continue
            # Check classes in this file
            cls_ids = [int(line.split()[0]) for line in lines if len(line.split()) == 5]
            # Prioritize images with violations or heavy machinery
            has_violation = any(classes[c].startswith("NO-") for c in cls_ids if c < len(classes))
            has_machinery = any(classes[c] in ["Excavator", "wheel loader", "dump truck"] for c in cls_ids if c < len(classes))
            
            img_file = img_dir / f"{lbl_file.stem}.jpg"
            if not img_file.exists():
                # Try png
                img_file = img_dir / f"{lbl_file.stem}.png"
            if img_file.exists():
                priority = (2 if has_violation and has_machinery else 1 if has_violation else 0)
                candidate_images.append((priority, split, img_file, lbl_file))
                
    # Sort by priority and pick representative samples
    candidate_images.sort(key=lambda x: x[0], reverse=True)
    selected = candidate_images[:num_samples]
    
    print(f"Generating {len(selected)} visual ground-truth annotations...")
    for idx, (_, split, img_path, lbl_path) in enumerate(selected):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w, _ = img.shape
        
        with open(lbl_path, 'r') as lf:
            lines = [l.strip() for l in lf.readlines() if l.strip()]
            
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            cid, xc, yc, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            
            x1 = int((xc - bw / 2.0) * w)
            y1 = int((yc - bh / 2.0) * h)
            x2 = int((xc + bw / 2.0) * w)
            y2 = int((yc + bh / 2.0) * h)
            
            cname = classes[cid] if cid < len(classes) else str(cid)
            color = colors.get(cid, (255, 255, 255))
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label_text = f"{cname}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, max(th + 6, y1)), color, -1)
            cv2.putText(img, label_text, (x1 + 2, max(th, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            
        out_filename = out_path / f"sample_{idx+1}_{split}_{img_path.stem}.jpg"
        cv2.imwrite(str(out_filename), img)
        print(f"  [Saved] {out_filename.name} with {len(lines)} annotations")
        
    print(f"Visualizations successfully saved to {out_path}")

if __name__ == "__main__":
    draw_annotations(".", "reports/visualizations", num_samples=6)
