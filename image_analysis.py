"""
image_analysis.py
PlantAnalyzer Ver2.0

画像解析メイン
"""

import cv2
import numpy as np

from pot_detection import (
    detect_pot,
    calculate_leaf_area,
    calculate_coverage
)

from color_analysis import (
    analyze_colors
)


def load_image(file):

    """
    Streamlitから渡された画像を読む
    """

    file_bytes = np.asarray(
        bytearray(file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    return image


def preprocess_image(image):

    """
    ノイズ除去
    """

    blur = cv2.GaussianBlur(
        image,
        (5, 5),
        0
    )

    return blur


def hsv_leaf_mask(image):

    """
    HSVで葉候補抽出
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    lower = np.array(
        [20, 25, 20],
        dtype=np.uint8
    )

    upper = np.array(
        [95, 255, 255],
        dtype=np.uint8
    )

    mask = cv2.inRange(
        hsv,
        lower,
        upper
    )

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return mask


def grabcut_leaf(image, mask):
    """
    GrabCutで葉抽出
    """

    gc_mask = np.where(
        mask > 0,
        cv2.GC_PR_FGD,
        cv2.GC_BGD
    ).astype("uint8")

    bgd = np.zeros(
        (1, 65),
        np.float64
    )

    fgd = np.zeros(
        (1, 65),
        np.float64
  )

    cv2.grabCut(
        image,
        gc_mask,
        None,
        bgd,
        fgd,
        5,
        cv2.GC_INIT_WITH_MASK
    )

    leaf_mask = np.where(
        (gc_mask == cv2.GC_FGD) |
        (gc_mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype(np.uint8)

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    leaf_mask = cv2.morphologyEx(
        leaf_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    leaf_mask = cv2.morphologyEx(
        leaf_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return leaf_mask


def extract_leaf(image):
    """
    葉抽出メイン
    """

    hsv_mask = hsv_leaf_mask(
        image
    )

    leaf_mask = grabcut_leaf(
        image,
        hsv_mask
    )

    return leaf_mask

def analyze_image(file):
    """
    PlantAnalyzer メイン解析
    """

    image = load_image(file)

    image = preprocess_image(
        image
    )

    leaf_mask = extract_leaf(
        image
    )

    pot_result = detect_pot(
        image
    )

    pot_mask = pot_result[
        "pot_mask"
    ]

    cm_per_pixel = pot_result[
        "cm_per_pixel"
    ]

    warped = pot_result[
        "warped"
    ]

    warped_mask = pot_result[
        "warped_mask"
    ]

    warped_leaf = extract_leaf(
        warped
    )

    color_result = analyze_colors(
        image,
        leaf_mask
    )

    leaf_area = calculate_leaf_area(
        warped_leaf,
        cm_per_pixel
    )

    coverage = calculate_coverage(
        warped_leaf,
        warped_mask
    )

    overlay = color_result[
        "overlay"
    ]

    result = {

    "original": image,

    "overlay": overlay,

    "leaf_mask": leaf_mask,

    "pot_mask": pot_mask,

    "coverage": coverage,

    "leaf_area": leaf_area,

    "best_k": color_result["best_k"],

    "color_ratio": color_result["ratio"],

    "clusters": color_result["clusters"],

    "mapping": color_result["mapping"],

    "cluster_centers": color_result["cluster_centers"],

    "cm_per_pixel": cm_per_pixel

    }

    return result


if __name__ == "__main__":

    print(
        "image_analysis.py loaded successfully."
  )


