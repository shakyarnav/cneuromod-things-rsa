#!/usr/bin/env python
"""Generate a 2x4 Stimulus Grid comparing Low vs. High Clutter within identical categories."""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

ANN_PATH = "/scratch/arnav.shakya/cneuromod-things/THINGS/glmsingle/task-things_imgAnnotations.json"
IMG_DIR = "/scratch/arnav.shakya/cneuromod-things/THINGS/images" # Adjust if your image path differs, or fallback to placeholder check
OUT_FIG = "/scratch/arnav.shakya/rsa_work/figures"
os.makedirs(OUT_FIG, exist_ok=True)

def main():
    if not os.path.exists(ANN_PATH):
        print(f"Annotation file not found at {ANN_PATH}")
        return

    with open(ANN_PATH) as f:
        ann_data = json.load(f)

    # Group images by category and clutter status
    # Low clutter: manual_lone_object == 1 or manual_central_* == 1
    # High clutter: manual_rich_background == 1 or manual_scene == 1
    categories = {}
    for img_name, meta in ann_data.items():
        cat = meta.get("image_category")
        if not cat:
            continue
        if cat not in categories:
            categories[cat] = {"low": [], "high": []}
            
        is_lone = meta.get("manual_lone_object", 0) == 1
        is_rich = meta.get("manual_rich_background", 0) == 1 or meta.get("manual_scene", 0) == 1
        
        if is_lone:
            categories[cat]["low"].append(img_name)
        elif is_rich:
            categories[cat]["high"].append(img_name)

    # Find categories having at least 2 low and 2 high clutter images
    valid_cats = [cat for cat, d in categories.items() if len(d["low"]) >= 2 and len(d["high"]) >= 2]
    print(f"Found {len(valid_cats)} categories with both low and high clutter exemplars.")

    if not valid_cats:
        print("No valid categories found with strict criteria.")
        return

    # Select 2 distinct categories (e.g., animals or common objects)
    selected_cats = valid_cats[:2]
    print(f"Selected categories for grid: {selected_cats}")

    # Setup 2x4 grid: Row 1 = Category A (Low vs High), Row 2 = Category B (Low vs High)
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10, 5))
    
    for row_idx, cat in enumerate(selected_cats):
        low_imgs = categories[cat]["low"][:2]
        high_imgs = categories[cat]["high"][:2]
        
        # We want columns: [Low 1, Low 2, High 1, High 2]
        col_images = [(img, "Low Clutter") for img in low_imgs] + [(img, "High Clutter") for img in high_imgs]
        
        for col_idx, (img_name, clutter_type) in enumerate(col_images):
            ax = axes[row_idx, col_idx]
            
            # Try to find the image file in standard THINGS subfolders
            # THINGS image paths are usually structured under category or root
            found_path = None
            for root, dirs, files in os.walk("/scratch/arnav.shakya/cneuromod-things/"):
                match = [f for f in files if f.startswith(img_name)]
                if match:
                    found_path = os.path.join(root, match[0])
                    break
            
            if found_path and os.path.exists(found_path):
                img_obj = Image.open(found_path).convert("RGB")
                ax.imshow(img_obj)
            else:
                # Fallback blank placeholder if raw image files are stored elsewhere in DataLad annex
                ax.text(0.5, 0.5, f"{img_name}\n({clutter_type})", ha="center", va="center", fontsize=8)
                ax.set_facecolor("#e9ecef")

            ax.set_xticks([])
            ax.set_yticks([])
            
            if row_idx == 0:
                ax.set_title(f"{clutter_type} {col_idx%2 + 1}", fontsize=10, fontweight="bold")
                
        # Label the rows on the left
        axes[row_idx, 0].set_ylabel(f"Cat: {cat.capitalize()}", fontsize=10, fontweight="bold")

    fig.suptitle("Stimulus Clutter Comparison Grid (Low vs. High Clutter)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    
    out_pdf = f"{OUT_FIG}/fig1_stimulus_clutter_grid.pdf"
    out_png = f"{OUT_FIG}/fig1_stimulus_clutter_grid.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    print(f"Successfully generated grid and saved to {out_pdf}")
    plt.close(fig)

if __name__ == "__main__":
    main()
