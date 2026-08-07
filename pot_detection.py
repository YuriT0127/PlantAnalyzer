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
    植木鉢らしい四角形輪郭を取得
    """

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    best_contour = None
    best_area = 0

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < 1000:
            continue

        epsilon = 0.02 * cv2.arcLength(
            contour,
            True
        )

        approx = cv2.approxPolyDP(
            contour,
            epsilon,
            True
        )

        if len(approx) != 4:
            continue

        if area > best_area:

            best_area = area
            best_contour = contour

    if best_contour is not None:
        return best_contour

    # 四角形が見つからない場合のみ最大輪郭を返す
    return max(
        contours,
        key=cv2.contourArea
    )


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

def order_corners(corners):
    """
    四隅を
    左上・右上・右下・左下
    の順に並べ替える
    """

    pts = np.array(
        corners,
        dtype=np.float32
    )

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    ordered = np.zeros(
        (4, 2),
        dtype=np.float32
    )

    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]

    return ordered

def warp_pot(image, corners):
    """
    植木鉢を真上から見た画像へ変換
    """

    if corners is None:
        return image

    corners = order_corners(corners)

    size = 500

    dst = np.float32([

        [0, 0],

        [size - 1, 0],

        [size - 1, size - 1],

        [0, size - 1]

    ])

    matrix = cv2.getPerspectiveTransform(
        corners,
        dst
    )

    warped = cv2.warpPerspective(
        image,
        matrix,
        (size, size)
    )

    return warped

def warp_mask(mask, corners):
    """
    マスクを真上画像へ変換
    """

    if corners is None:
        return mask

    corners = order_corners(corners)

    size = 500

    dst = np.float32([

        [0, 0],

        [size - 1, 0],

        [size - 1, size - 1],

        [0, size - 1]

    ])

    matrix = cv2.getPerspectiveTransform(
        corners,
        dst
    )

    warped = cv2.warpPerspective(

        mask,

        matrix,

        (size, size),

        flags=cv2.INTER_NEAREST

    )

    return warped
    
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

    if mean_length <= 0:
        return POT_SIZE_CM / 500

    cm_per_pixel = POT_SIZE_CM / mean_length

    return cm_per_pixel

    if cm_per_pixel is None:
        cm_per_pixel = POT_SIZE_CM / 500
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

    warped = warp_pot(
        image,
        corners
    )

    warped_mask = warp_mask(
        pot_mask,
        corners
    )

    return {

        "corners": corners,

        "cm_per_pixel": cm_per_pixel,

        "pot_mask": pot_mask,

        "warped": warped,

        "warped_mask": warped_mask
    }


if __name__ == "__main__":

    print(
        "pot_detection.py loaded successfully."
    )


