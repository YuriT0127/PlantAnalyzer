"""
pot_detection.py
PlantAnalyzer

植木鉢検出・透視補正・面積換算
"""

import cv2
import numpy as np

from config import POT_SIZE_CM


# =========================================================
# 設定
# =========================================================

WARP_SIZE = 500


# =========================================================
# 前処理
# =========================================================

def preprocess_pot(image):
    """
    植木鉢検出用のグレースケール画像を作成
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    return gray


# =========================================================
# エッジ検出
# =========================================================

def detect_edges(gray):
    """
    植木鉢の輪郭候補を抽出
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


# =========================================================
# 四角形の角度確認
# =========================================================

def angle_cosine(p1, p2, p3):
    """
    p2を頂点とした角度のcos
    """

    v1 = p1.astype(np.float64) - p2.astype(np.float64)
    v2 = p3.astype(np.float64) - p2.astype(np.float64)

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 1.0

    return abs(
        np.dot(v1, v2)
        / (norm1 * norm2)
    )


def is_valid_quadrilateral(
    corners,
    image_shape
):
    """
    検出した四角形が
    植木鉢として妥当か確認
    """

    if corners is None:
        return False

    if len(corners) != 4:
        return False

    h, w = image_shape[:2]

    # -------------------------
    # 凸四角形か
    # -------------------------

    contour = corners.reshape(
        (-1, 1, 2)
    ).astype(np.float32)

    if not cv2.isContourConvex(
        contour
    ):
        return False

    # -------------------------
    # 面積
    # -------------------------

    area = abs(
        cv2.contourArea(
            contour
        )
    )

    image_area = h * w

    # 画像の1%未満は小さすぎる
    if area < image_area * 0.01:
        return False

    # 画像全体に近すぎるものも除外
    if area > image_area * 0.98:
        return False

    # -------------------------
    # 各角の形
    # -------------------------

    cosines = []

    for i in range(4):

        p1 = corners[
            (i - 1) % 4
        ]

        p2 = corners[i]

        p3 = corners[
            (i + 1) % 4
        ]

        cosines.append(
            angle_cosine(
                p1,
                p2,
                p3
            )
        )

    # 90°ならcos=0
    # 極端に鋭い角を除外
    if max(cosines) > 0.65:
        return False

    # -------------------------
    # 辺の長さ
    # -------------------------

    lengths = []

    for i in range(4):

        p1 = corners[i]

        p2 = corners[
            (i + 1) % 4
        ]

        length = np.linalg.norm(
            p1.astype(np.float64)
            - p2.astype(np.float64)
        )

        lengths.append(
            length
        )

    if min(lengths) <= 10:
        return False

    # 向かい合う辺が極端に違うものを除外
    ratio1 = (
        max(lengths[0], lengths[2])
        /
        min(lengths[0], lengths[2])
    )

    ratio2 = (
        max(lengths[1], lengths[3])
        /
        min(lengths[1], lengths[3])
    )

    if ratio1 > 2.5:
        return False

    if ratio2 > 2.5:
        return False

    return True


# =========================================================
# 植木鉢輪郭検出
# =========================================================

def find_pot_contour(
    edges,
    image_shape
):
    """
    植木鉢らしい四角形を探す。

    単純に最大輪郭を使わず、
    妥当な四角形の中から最大のものを選ぶ。
    """

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    candidates = []

    image_area = (
        image_shape[0]
        * image_shape[1]
    )

    for contour in contours:

        area = abs(
            cv2.contourArea(
                contour
            )
        )

        if area < image_area * 0.01:
            continue

        perimeter = cv2.arcLength(
            contour,
            True
        )

        if perimeter <= 0:
            continue

        # 複数のepsilonを試す
        for epsilon_ratio in [
            0.01,
            0.015,
            0.02,
            0.03
        ]:

            approx = cv2.approxPolyDP(
                contour,
                epsilon_ratio
                * perimeter,
                True
            )

            if len(approx) != 4:
                continue

            corners = (
                approx.reshape(4, 2)
                .astype(np.float32)
            )

            if not is_valid_quadrilateral(
                corners,
                image_shape
            ):
                continue

            candidates.append(
                (
                    area,
                    contour
                )
            )

            break

    if not candidates:
        return None

    # 最も大きい妥当な四角形
    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return candidates[0][1]


# =========================================================
# 四隅取得
# =========================================================

def get_pot_corners(
    contour
):
    """
    輪郭から四隅を取得
    """

    if contour is None:
        return None

    perimeter = cv2.arcLength(
        contour,
        True
    )

    if perimeter <= 0:
        return None

    best = None

    for epsilon_ratio in [
        0.01,
        0.015,
        0.02,
        0.03,
        0.04
    ]:

        approx = cv2.approxPolyDP(
            contour,
            epsilon_ratio
            * perimeter,
            True
        )

        if len(approx) == 4:

            corners = (
                approx.reshape(4, 2)
                .astype(np.float32)
            )

            best = corners

            break

    return best


# =========================================================
# 四隅の順番を統一
# =========================================================

def order_corners(
    corners
):
    """
    左上・右上・右下・左下
    の順番に並べる
    """

    pts = np.array(
        corners,
        dtype=np.float32
    )

    # 重複点がないか確認
    unique = np.unique(
        pts,
        axis=0
    )

    if len(unique) != 4:
        return None

    s = pts.sum(
        axis=1
    )

    diff = (
        pts[:, 1]
        - pts[:, 0]
    )

    ordered = np.zeros(
        (4, 2),
        dtype=np.float32
    )

    ordered[0] = pts[
        np.argmin(s)
    ]

    ordered[2] = pts[
        np.argmax(s)
    ]

    ordered[1] = pts[
        np.argmin(diff)
    ]

    ordered[3] = pts[
        np.argmax(diff)
    ]

    return ordered


# =========================================================
# 透視補正
# =========================================================

def warp_pot(
    image,
    corners
):
    """
    植木鉢を500×500 pxの正方形へ透視補正
    """

    if corners is None:
        return image.copy()

    corners = order_corners(
        corners
    )

    if corners is None:
        return image.copy()

    size = WARP_SIZE

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
        (size, size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    return warped


# =========================================================
# マスクの透視補正
# =========================================================

def warp_mask(
    mask,
    corners
):
    """
    ポットマスクを500×500へ変換
    """

    if corners is None:
        return np.ones(
            (WARP_SIZE, WARP_SIZE),
            dtype=np.uint8
        ) * 255

    corners = order_corners(
        corners
    )

    if corners is None:
        return np.ones(
            (WARP_SIZE, WARP_SIZE),
            dtype=np.uint8
        ) * 255

    src = np.float32(
        corners
    )

    dst = np.float32([
        [0, 0],
        [WARP_SIZE - 1, 0],
        [WARP_SIZE - 1, WARP_SIZE - 1],
        [0, WARP_SIZE - 1]
    ])

    matrix = cv2.getPerspectiveTransform(
        src,
        dst
    )

    warped = cv2.warpPerspective(
        mask,
        matrix,
        (
            WARP_SIZE,
            WARP_SIZE
        ),
        flags=cv2.INTER_NEAREST
    )

    # 透視補正後の正方形そのものを
    # ポット領域として扱う
    warped = np.where(
        warped > 0,
        255,
        0
    ).astype(
        np.uint8
    )

    # マスクがほぼ空なら全面をポットとする
    if np.count_nonzero(
        warped
    ) < WARP_SIZE * WARP_SIZE * 0.5:

        warped = np.ones(
            (
                WARP_SIZE,
                WARP_SIZE
            ),
            dtype=np.uint8
        ) * 255

    return warped


# =========================================================
# スケール
# =========================================================

def calculate_scale(
    corners=None
):
    """
    透視補正後は500×500 px。

    ポットの一辺を17 cmとして、
    1 pxあたりのcmを計算する。
    """

    return (
        POT_SIZE_CM
        / WARP_SIZE
    )


# =========================================================
# 元画像上のポットマスク
# =========================================================

def create_pot_mask(
    shape,
    corners
):
    """
    元画像上のポット領域
    """

    mask = np.zeros(
        shape[:2],
        dtype=np.uint8
    )

    if corners is None:
        return mask

    ordered = order_corners(
        corners
    )

    if ordered is None:
        return mask

    cv2.fillPoly(
        mask,
        [
            ordered.astype(
                np.int32
            )
        ],
        255
    )

    return mask


# =========================================================
# 葉面積
# =========================================================

def calculate_leaf_area(
    leaf_mask,
    cm_per_pixel
):
    """
    葉面積(cm²)
    """

    if leaf_mask is None:
        return 0.0

    leaf_pixels = np.count_nonzero(
        leaf_mask
    )

    area = (
        leaf_pixels
        * (cm_per_pixel ** 2)
    )

    return float(
        area
    )


# =========================================================
# 被覆率
# =========================================================

def calculate_coverage(
    leaf_mask,
    pot_mask
):
    """
    ポット面積に対する葉の被覆率(%)
    """

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

    coverage = (
        leaf_pixels
        / pot_pixels
    ) * 100.0

    return float(
        coverage
    )


# =========================================================
# メイン
# =========================================================

def detect_pot(
    image
):
    """
    植木鉢を検出し、
    透視補正画像とマスクを返す。
    """

    if image is None:
        raise ValueError(
            "画像がありません。"
        )

    # -------------------------
    # 前処理
    # -------------------------

    gray = preprocess_pot(
        image
    )

    edges = detect_edges(
        gray
    )

    # -------------------------
    # 四角形検出
    # -------------------------

    contour = find_pot_contour(
        edges,
        image.shape
    )

    corners = get_pot_corners(
        contour
    )

    # -------------------------
    # 検出失敗
    # -------------------------

    if corners is None:

        # 透視補正は行わない
        # ただしスケールは
        # 500px = 17cmとして統一
        cm_per_pixel = (
            POT_SIZE_CM
            / WARP_SIZE
        )

        return {
            "corners": None,
            "cm_per_pixel":
                cm_per_pixel,
            "pot_mask":
                np.zeros(
                    image.shape[:2],
                    dtype=np.uint8
                ),
            "warped":
                cv2.resize(
                    image,
                    (WARP_SIZE, WARP_SIZE)
                ),
            "warped_mask":
                np.ones(
                    (
                        WARP_SIZE,
                        WARP_SIZE
                    ),
                    dtype=np.uint8
                ) * 255,
            "detected":
                False
        }

    # -------------------------
    # 四隅の順番
    # -------------------------

    corners = order_corners(
        corners
    )

    if corners is None:

        return {
            "corners": None,
            "cm_per_pixel":
                POT_SIZE_CM
                / WARP_SIZE,
            "pot_mask":
                np.zeros(
                    image.shape[:2],
                    dtype=np.uint8
                ),
            "warped":
                cv2.resize(
                    image,
                    (WARP_SIZE, WARP_SIZE)
                ),
            "warped_mask":
                np.ones(
                    (
                        WARP_SIZE,
                        WARP_SIZE
                    ),
                    dtype=np.uint8
                ) * 255,
            "detected":
                False
        }

    # -------------------------
    # 元画像マスク
    # -------------------------

    pot_mask = create_pot_mask(
        image.shape,
        corners
    )

    # -------------------------
    # 透視補正
    # -------------------------

    warped = warp_pot(
        image,
        corners
    )

    warped_mask = warp_mask(
        pot_mask,
        corners
    )

    # -------------------------
    # スケール
    # -------------------------

    cm_per_pixel = calculate_scale(
        corners
    )

    return {
        "corners":
            corners,

        "cm_per_pixel":
            cm_per_pixel,

        "pot_mask":
            pot_mask,

        "warped":
            warped,

        "warped_mask":
            warped_mask,

        "detected":
            True
    }


# =========================================================
# テスト
# =========================================================

if __name__ == "__main__":

    print(
        "pot_detection.py "
        "loaded successfully."
  )
