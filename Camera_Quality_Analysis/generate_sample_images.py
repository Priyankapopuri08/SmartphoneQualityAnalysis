"""
Sample Image Generator for Camera Quality Analysis
Generates sample images matching the three experimental scenarios from Section 3.8:
1. Daylight outdoor
2. Indoor lighting
3. Low-light conditions
Across simulated device aging states (2016 = higher sensor noise/blur, 2017 = moderate, 2018 = sharp/clean).
"""

import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

def create_scene_base(scenario="daylight", width=640, height=480):
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)

    if scenario == "daylight":
        # Sky gradient
        for y in range(height // 2):
            b = int(220 + (255 - 220) * (y / (height // 2)))
            g = int(180 + (220 - 180) * (y / (height // 2)))
            r = int(120 + (180 - 120) * (y / (height // 2)))
            draw.line([(0, y), (width, y)], fill=(r, g, b))
        # Sun / Bright light
        draw.ellipse([width - 150, 40, width - 70, 120], fill=(255, 245, 180))
        # Landscape / Greenery
        draw.polygon([(0, height // 2), (width // 3, height // 3), (width, height // 2), (width, height), (0, height)], fill=(45, 110, 35))
        # Building / structure
        draw.rectangle([100, height // 2 - 80, 260, height - 50], fill=(180, 160, 140))
        for wy in range(height // 2 - 60, height - 70, 30):
            for wx in range(120, 240, 35):
                draw.rectangle([wx, wy, wx + 20, wy + 20], fill=(80, 120, 160))

    elif scenario == "indoor":
        # Neutral indoor wall
        draw.rectangle([0, 0, width, height], fill=(225, 220, 210))
        # Warm ambient lighting gradient
        for x in range(width):
            ratio = 1.0 - (abs(x - width // 2) / (width // 2))
            shade = int(15 * ratio)
            draw.line([(x, 0), (x, height)], fill=(min(255, 225 + shade), min(255, 220 + shade // 2), 210))
        # Table surface
        draw.rectangle([0, height - 140, width, height], fill=(140, 90, 60))
        # Object on table (Cup/Book)
        draw.rectangle([width // 2 - 50, height - 220, width // 2 + 50, height - 120], fill=(70, 90, 130))
        draw.ellipse([width // 2 - 40, height - 250, width // 2 + 40, height - 210], fill=(210, 210, 215))

    elif scenario == "lowlight":
        # Dark night room
        draw.rectangle([0, 0, width, height], fill=(20, 22, 30))
        # Single lamp / weak light source
        draw.ellipse([width // 2 - 40, 80, width // 2 + 40, 160], fill=(230, 200, 120))
        # Dim room outline
        draw.rectangle([80, height - 180, width - 80, height - 40], fill=(35, 38, 48))
        draw.line([(0, height - 20), (width, height - 20)], fill=(45, 48, 60), width=3)

    return img

def apply_device_aging_effects(base_img, year=2018):
    """
    Simulate camera aging characteristics:
    - 2018: Sharp optics, well-calibrated sensor, low noise
    - 2017: Minor optical softening, slight sensor chroma noise
    - 2016: Noticeable lens aberration/softening, sensor noise, reduced dynamic range
    """
    arr = np.array(base_img, dtype=np.float64)

    if year == 2018:
        # High fidelity, minimal noise
        noise = np.random.normal(0, 1.5, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    elif year == 2017:
        # Slight blur + moderate sensor noise
        noise = np.random.normal(0, 4.0, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius=0.4))

    elif year == 2016:
        # Heavy noise + lens softening + contrast drift
        noise = np.random.normal(0, 8.5, arr.shape)
        arr = (arr - 128) * 0.92 + 128  # contrast compression
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius=0.85))

    return img

def generate_all_samples(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    scenarios = ["daylight", "indoor", "lowlight"]
    scenario_names = {
        "daylight": "daylight_outdoor",
        "indoor": "indoor_lighting",
        "lowlight": "low_light"
    }
    years = [2016, 2017, 2018]

    for sc in scenarios:
        base = create_scene_base(sc)
        for yr in years:
            fn = f"{scenario_names[sc]}_{yr}.jpg"
            fp = os.path.join(out_dir, fn)
            final_img = apply_device_aging_effects(base, year=yr)
            final_img.save(fp, quality=95)
            print(f"Generated sample: {fn}")

if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "sample_images")
    generate_all_samples(out)
