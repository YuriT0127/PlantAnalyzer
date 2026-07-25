"""
image_analysis.py
画像解析
"""

import cv2
import numpy as np

from config import COLOR_RANGES


# ==========================
# 画像読込
# ==========================

def load_image(file):

    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)

    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    return image


# ==========================
# BGR→HSV
# ==========================

def to_hsv(image):

    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


# ==========================
# 色別解析
# ==========================

def analyze_colors(image):

    hsv = to_hsv(image)

    results = {}

    total_leaf = 0

    total_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    overlay = image.copy()

    draw_colors = {

        "Dark Green": (0,120,0),
        "Green": (0,255,0),
        "Light Green": (120,255,120),
        "Yellow": (0,255,255),
        "Brown": (30,60,180)

    }

    for name, (lower, upper) in COLOR_RANGES.items():

        mask = cv2.inRange(hsv, lower, upper)

        pixel = int(np.count_nonzero(mask))

        results[name] = pixel

        total_leaf += pixel

        total_mask = cv2.bitwise_or(total_mask, mask)

        color = np.zeros_like(image)

        color[:] = draw_colors[name]

        overlay[mask > 0] = (
            overlay[mask > 0] * 0.4 +
            color[mask > 0] * 0.6
        ).astype(np.uint8)

    return results, total_leaf, total_mask, overlay


# ==========================
# 被覆率
# ==========================

def coverage(total_leaf_pixels, soil_pixels):

    if soil_pixels is None:

        return None

    if soil_pixels == 0:

        return None

    return round(
        total_leaf_pixels / soil_pixels * 100,
        2
    )
  # ==========================
# 色割合
# ==========================

def ratios(results, total_leaf):

    ratio = {}

    if total_leaf == 0:

        for key in results:

            ratio[key] = 0

        return ratio

    for key, value in results.items():

        ratio[key] = round(
            value / total_leaf * 100,
            2
        )

    return ratio
