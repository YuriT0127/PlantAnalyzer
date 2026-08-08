# image_analysis.py
# PlantAnalyzer Ver3.0

import cv2
import numpy as np

from pot_detection import (
    detect_pot,
    calculate_leaf_area,
    calculate_coverage,
)

from color_analysis import (
    analyze_colors,
)


# =========================
# 画像読み込み
# =========================

def load_image(file):
    """アップロードされた画像をOpenCV画像に変換"""

    if file is None:
        raise ValueError(
            "画像が選択されていません。"
        )

    data = file.read()

    if not data:
        raise ValueError(
            "画像データが空です。"
        )

    array = np.frombuffer(
        data,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError(
            "画像を読み込めませんでした。"
        )

    return image


# =========================
# 葉抽出
# =========================

def extract_leaf(image):
    """
    HSVを利用して植物の葉を抽出。

    緑～黄色系の領域を葉として扱う。
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    # 緑
    lower_green = np.array(
        [25, 30, 20],
        dtype=np.uint8
    )

    upper_green = np.array(
        [95, 255, 255],
        dtype=np.uint8
    )

    mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )

    # 黄色系も含める
    lower_yellow = np.array(
        [15, 30, 20],
        dtype=np.uint8
    )

    upper_yellow = np.array(
        [35, 255, 255],
        dtype=np.uint8
    )

    yellow_mask = cv2.inRange(
        hsv,
        lower_yellow,
        upper_yellow
    )

    mask = cv2.bitwise_or(
        mask,
        yellow_mask
    )

    # ノイズ除去
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


# =========================
# GrabCut
# =========================

def grabcut_leaf(
    image,
    mask
):
    """
    葉マスクをGrabCutで精密化。

    GrabCutが不安定な画像では、
    元のHSVマスクをそのまま使用する。
    """

    if image is None:
        return mask

    if mask is None:
        return None

    foreground = np.count_nonzero(
        mask
    )

    background = (
        mask.size -
        foreground
    )

    # 前景・背景が不足している場合
    if foreground < 100:
        return mask

    if background < 100:
        return mask

    gc_mask = np.full(
        mask.shape,
        cv2.GC_BGD,
        dtype=np.uint8
    )

    # 葉領域を確実な前景として設定
    gc_mask[
        mask > 0
    ] = cv2.GC_PR_FGD

    # 中央付近の葉を前景として扱う
    gc_mask[
        mask > 0
    ] = cv2.GC_FGD

    bgd = np.zeros(
        (1, 65),
        np.float64
    )

    fgd = np.zeros(
        (1, 65),
        np.float64
    )

    try:

        cv2.grabCut(
            image,
            gc_mask,
            None,
            bgd,
            fgd,
            2,
            cv2.GC_INIT_WITH_MASK
        )

    except cv2.error:

        return mask

    result = np.where(
        (
            gc_mask ==
            cv2.GC_FGD
        )
        |
        (
            gc_mask ==
            cv2.GC_PR_FGD
        ),
        255,
        0
    ).astype(
        np.uint8
    )

    return result


# =========================
# 葉マスク改善
# =========================

def clean_leaf_mask(mask):
    """葉マスクの小さなノイズを除去"""

    if mask is None:
        return None

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


# =========================
# メイン解析
# =========================

def analyze_image(file):
    """
    画像1枚を解析。

    戻り値：

    image
    leaf_mask
    pot_result
    warped
    warped_mask
    warped_leaf_mask
    leaf_area
    coverage
    color_result
    """

    # ---------------------
    # 1. 読み込み
    # ---------------------

    image = load_image(
        file
    )

    # ---------------------
    # 2. 葉抽出
    # ---------------------

    leaf_mask = extract_leaf(
        image
    )

    # ---------------------
    # 3. ポット検出
    # ---------------------

    pot_result = detect_pot(
        image
    )

    corners = pot_result[
        "corners"
    ]

    cm_per_pixel = pot_result[
        "cm_per_pixel"
    ]

    pot_mask = pot_result[
        "pot_mask"
    ]

    warped = pot_result[
        "warped"
    ]

    warped_mask = pot_result[
        "warped_mask"
    ]

    # ---------------------
    # 4. 葉マスクをポット内に限定
    # ---------------------

    leaf_mask_inside = (
        cv2.bitwise_and(
            leaf_mask,
            pot_mask
        )
    )

    # ---------------------
    # 5. GrabCut
    # ---------------------

    leaf_mask_refined = (
        grabcut_leaf(
            image,
            leaf_mask_inside
        )
    )

    leaf_mask_refined = (
        clean_leaf_mask(
            leaf_mask_refined
        )
    )

    # ---------------------
    # 6. 正面化した葉マスク
    # ---------------------

    from pot_detection import warp_mask

    warped_leaf_mask = (
        warp_mask(
            leaf_mask_refined,
            corners
        )
    )

    # ---------------------
    # 7. 葉面積
    # ---------------------

    leaf_area = calculate_leaf_area(
        warped_leaf_mask,
        cm_per_pixel
    )

    # ---------------------
    # 8. 被覆率
    # ---------------------

    coverage = calculate_coverage(
        warped_leaf_mask,
        warped_mask
    )

    # ---------------------
    # 9. 色解析
    # ---------------------

    color_result = analyze_colors(
        warped,
        warped_leaf_mask
    )

    # ---------------------
    # 10. 結果
    # ---------------------

    return {

        "image":
            image,

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

        "leaf_mask":
            leaf_mask_refined,

        "warped_leaf_mask":
            warped_leaf_mask,

        "leaf_area":
            leaf_area,

        "coverage":
            coverage,

        "color_result":
            color_result

    }


# =========================
# テスト
# =========================

if __name__ == "__main__":

    print(
        "image_analysis.py "
        "loaded successfully."
  )
