"""
pot_detection.py
PlantAnalyzer Ver2.0

植木鉢検出
"""

import cv2
import numpy as np

from config import POT_SIZE_CM


def preprocess_pot(image):
    """
    植木鉢検出用前処理
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    return blur


def detect_edges(gray):
    """
    エッジ抽出
    """

    edges = cv2.Canny(
        gray,
        50,
        150
    )

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    edges = cv2.dilate(
        edges,
        kernel,
        iterations=1
    )

    return edges

def find_pot_contour(edges):
    """
    植木鉢の輪郭を取得
    """

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    # 最大輪郭を植木鉢とみなす
    contour = max(
        contours,
        key=cv2.contourArea
    )

    return contour


def approximate_pot(contour):
    """
    植木鉢を四角形近似
    """

    if contour is None:
        return None

    epsilon = 0.02 * cv2.arcLength(
        contour,
        True
    )

    approx = cv2.approxPolyDP(
        contour,
        epsilon,
        True
    )

    return approx


def get_pot_corners(approx):
    """
    四隅座標を取得
    """

    if approx is None:
        return None

    if len(approx) != 4:
        return None

    corners = approx.reshape(4, 2)

    return corners

def calculate_scale(corners):
    """
    cm/pixel を計算
    """

    if corners is None:
        return None

    lengths = []

    for i in range(4):

        p1 = corners[i]

        p2 = corners[(i + 1) % 4]

        length = np.linalg.norm(
            p1 - p2
        )

        lengths.append(length)

    mean_length = np.mean(lengths)

    cm_per_pixel = (
        POT_SIZE_CM /
        mean_length
    )

    return cm_per_pixel


def create_pot_mask(shape, corners):
    """
    植木鉢内マスク作成
    """

    mask = np.zeros(
        shape[:2],
        dtype=np.uint8
    )

    if corners is None:
        return mask

    cv2.fillPoly(
        mask,
        [corners.astype(np.int32)],
        255
    )

    return mask

def calculate_leaf_area(leaf_mask, cm_per_pixel):
    """
    総葉面積(cm²)
    """

    leaf_pixels = np.count_nonzero(
        leaf_mask
    )

    area = (
        leaf_pixels *
        (cm_per_pixel ** 2)
    )

    return area


def calculate_coverage(
    leaf_mask,
    pot_mask
):
    """
    被覆率(%)
    """

    inside_leaf = cv2.bitwise_and(
        leaf_mask,
        pot_mask
    )

    leaf_pixels = np.count_nonzero(
        inside_leaf
    )

    pot_pixels = np.count_nonzero(
        pot_mask
    )

    if pot_pixels == 0:
        return 0.0

    return (
        leaf_pixels /
        pot_pixels
    ) * 100


def detect_pot(image):

    gray = preprocess_pot(image)

    edges = detect_edges(gray)

    contour = find_pot_contour(edges)

    approx = approximate_pot(contour)

    corners = get_pot_corners(approx)

    cm_per_pixel = calculate_scale(
        corners
    )

    pot_mask = create_pot_mask(
        image.shape,
        corners
    )

    return {

        "corners": corners,

        "cm_per_pixel": cm_per_pixel,

        "pot_mask": pot_mask

    }


if __name__ == "__main__":

    print(
        "pot_detection.py loaded successfully."
    )


