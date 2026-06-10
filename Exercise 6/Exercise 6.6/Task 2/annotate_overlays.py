from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE = Path("c:/Users/ABI/Desktop/Sub Docs/Semester 5/ITMLS/2026/Exercise 6/Exercise 6.6/Task 2/output")


candidates = list(BASE.rglob('mispred_*.jpg')) + list(BASE.rglob('fp_*.png')) + list(BASE.rglob('fn_*.png')) + list(BASE.rglob('mispred_*.png'))

if not candidates:
    print("No candidate overlays found.")
    raise SystemExit(0)


try:
    font = ImageFont.truetype("arial.ttf", 18)
except:
    font = ImageFont.load_default()

for p in candidates:
    img = cv2.imread(str(p))
    if img is None:
        continue
    h, w = img.shape[:2]
    
    panel_w = w // 3
    cam_panel = img[:, panel_w:panel_w*2]
    
    gray = cv2.cvtColor(cam_panel, cv2.COLOR_BGR2GRAY)
    # blur to find region
    blur = cv2.GaussianBlur(gray, (9,9), 0)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(blur)
    cx, cy = maxLoc
    
    full_x = panel_w + cx
    full_y = cy
    
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    
    r = max(8, w//120)
    draw.ellipse((full_x-r, full_y-r, full_x+r, full_y+r), outline=(255,0,0), width=3)
    
    label = f"Hotspot ({maxVal:.1f})"
    text_x, text_y = 10, 10
    draw.rectangle((text_x-4, text_y-4, text_x+300, text_y+28), fill=(0,0,0,128))
    draw.text((text_x, text_y), f"{p.parent.name} / {p.name} - {label}", fill=(255,255,255), font=font)
    outp = p.with_name('annotated_' + p.name)
    pil.save(str(outp))
    print('Saved:', outp)

print('Done')
