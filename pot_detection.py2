# pot_detection.py
# PlantAnalyzer Ver3.0

import cv2
import numpy as np

from config import POT_SIZE_CM


# =========================
# 前処理
# =========================

def preprocess_pot(image):
    """植木鉢検出用の前処理"""

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


# =========================
# エッジ検出
# =========================

def detect_edges(gray):
    """エッジを抽出"""

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


# =========================
# ポット輪郭検出
# =========================

def find_pot_contour(edges):
    """植木鉢らしい輪郭を取得"""

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

        epsilon = (
            0.02 *
            cv2.arcLength(
                contour,
                True
            )
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

    # 四角形が見つからない場合は最大輪郭
    return max(
        contours,
        key=cv2.contourArea
    )


# =========================
# 四角形近似
# =========================

def approximate_pot(contour):
    """輪郭を四角形に近似"""

    if contour is None:
        return None

    epsilon = (
        0.02 *
        cv2.arcLength(
            contour,
            True
        )
    )

    approx = cv2.approxPolyDP(
        contour,
        epsilon,
        True
    )

    return approx


# =========================
# 四隅取得
# =========================

def get_pot_corners(approx):
    """ポットの四隅を取得"""

    if approx is None:
        return None

    if len(approx) != 4:
        return None

    return approx.reshape(4, 2)


# =========================
# 四隅の順番を統一
# =========================

def order_corners(corners):
    """
    左上
    右上
    右下
    左下
    の順に並べる
    """

    pts = np.asarray(
        corners,
        dtype=np.float32
    )

    s = pts.sum(axis=1)
    diff = np.diff(
        pts,
        axis=1
    )

    ordered = np.zeros(
        (4, 2),
        dtype=np.float32
    )

    ordered[0] = pts[np.argmin(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[2] = pts[np.argmax(s)]
    ordered[3] = pts[np.argmax(diff)]

    return ordered


# =========================
# ポット画像を正面化
# =========================

def warp_pot(image, corners):
    """ポット領域を500×500に変換"""

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


# =========================
# マスクを正面化
# =========================

def warp_mask(mask, corners):
    """マスクを500×500に変換"""

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


# =========================
# スケール計算
# =========================

def calculate_scale(corners):
    """
    1 pixelあたりのcmを計算
    """

    if corners is None:
        return POT_SIZE_CM / 500.0

    lengths = []

    for i in range(4):

        p1 = corners[i]
        p2 = corners[
            (i + 1) % 4
        ]

        length = np.linalg.norm(
            p1 - p2
        )

        if length > 0:
            lengths.append(length)

    if len(lengths) == 0:
        return POT_SIZE_CM / 500.0

    mean_length = np.mean(
        lengths
    )

    if mean_length <= 0:
        return POT_SIZE_CM / 500.0

    return (
        POT_SIZE_CM /
        mean_length
    )


# =========================
# ポットマスク
# =========================

def create_pot_mask(
    shape,
    corners
):
    """ポット内部のマスクを作成"""

    mask = np.zeros(
        shape[:2],
        dtype=np.uint8
    )

    if corners is None:
        return mask

    cv2.fillPoly(
        mask,
        [
            corners.astype(
                np.int32
            )
        ],
        255
    )

    return mask


# =========================
# 葉面積
# =========================

def calculate_leaf_area(
    leaf_mask,
    cm_per_pixel
):
    """総葉面積(cm²)"""

    if leaf_mask is None:
        return 0.0

    if cm_per_pixel is None:
        return 0.0

    leaf_pixels = np.count_nonzero(
        leaf_mask
    )

    return (
        leaf_pixels *
        (cm_per_pixel ** 2)
    )


# =========================
# 被覆率
# =========================

def calculate_coverage(
    leaf_mask,
    pot_mask
):
    """ポット内の葉の被覆率(%)"""

    if leaf_mask is None:
        return 0.0

    if pot_mask is None:
        return 0.0

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
    ) * 100.0


# =========================
# メイン
# =========================

def detect_pot(image):
    """
    植木鉢を検出して、

    corners
    cm_per_pixel
    pot_mask
    warped
    warped_mask

    を返す
    """

    if image is None:
        raise ValueError(
            "画像がありません。"
        )

    gray = preprocess_pot(
        image
    )

    edges = detect_edges(
        gray
    )

    contour = find_pot_contour(
        edges
    )

    approx = approximate_pot(
        contour
    )

    corners = get_pot_corners(
        approx
    )

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


# =========================
# テスト
# =========================

if __name__ == "__main__":

    print(
        "pot_detection.py "
        "loaded successfully."
  )
