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
    HSV + ExGを利用して葉を抽出。
    緑～黄緑系の領域を候補とし、
    ExGで植物らしい緑を強調する。
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    # -------------------------
    # HSVによる葉候補
    # -------------------------
    lower_green = np.array(
        [25, 45, 35],
        dtype=np.uint8
    )

    upper_green = np.array(
        [90, 255, 255],
        dtype=np.uint8
    )

    hsv_mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )

    # -------------------------
    # ExG (Excess Green)
    # ExG = 2G - R - B
    # -------------------------
    b, g, r = cv2.split(image)

    exg = (
        2.0 * g.astype(np.float32)
        - r.astype(np.float32)
        - b.astype(np.float32)
    )

    exg_mask = np.where(
        exg > 20,
        255,
        0
    ).astype(np.uint8)

    # -------------------------
    # HSVとExGを両方満たす
    # -------------------------
    mask = cv2.bitwise_and(
        hsv_mask,
        exg_mask
    )

    # -------------------------
    # ノイズ除去
    # -------------------------
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

    処理順：
    1. 画像読み込み
    2. 植木鉢検出
    3. 植木鉢の透視補正
    4. 正面化画像で葉を抽出
    5. ポット内部だけに限定
    6. 葉面積計算
    7. 被覆率計算
    8. 葉色解析

    GrabCutは使用しない。
    """

    # =========================
    # 1. 画像読み込み
    # =========================

    image = load_image(file)

    # =========================
    # 2. 植木鉢検出
    # =========================

    pot_result = detect_pot(
        image
    )

    # 検出失敗を明示的に停止
    if not pot_result.get(
        "detected",
        False
    ):
        raise ValueError(
            "植木鉢を正しく検出できませんでした。"
            "植木鉢全体が画像に入っているか確認してください。"
        )

    # =========================
    # 3. 植木鉢情報
    # =========================

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

    # =========================
    # 4. 正面化画像で葉抽出
    # =========================
    #
    # 元画像ではなく、
    # 透視補正後の画像を解析する。
    #

    leaf_mask_warped = extract_leaf(
        warped
    )

    # =========================
    # 5. ポット内部だけに限定
    # =========================

    warped_leaf_mask = cv2.bitwise_and(
        leaf_mask_warped,
        warped_mask
    )

    # =========================
    # 6. ポット境界付近を除外
    # =========================
    #
    # 境界線や床などの誤認識を
    # 少し安全側に除外する。
    #

    kernel = np.ones(
        (7, 7),
        np.uint8
    )

    warped_mask_inner = cv2.erode(
        warped_mask,
        kernel,
        iterations=2
    )

    warped_leaf_mask = cv2.bitwise_and(
        warped_leaf_mask,
        warped_mask_inner
    )

    # =========================
    # 7. 葉マスクのノイズ除去
    # =========================

    warped_leaf_mask = clean_leaf_mask(
        warped_leaf_mask
    )

    # =========================
    # 8. 異常チェック
    # =========================

    pot_pixels = np.count_nonzero(
        warped_mask_inner
    )

    leaf_pixels = np.count_nonzero(
        warped_leaf_mask
    )

    if pot_pixels == 0:
        raise ValueError(
            "植木鉢の解析領域が取得できませんでした。"
        )

    # 葉がポット全体の99%以上なら
    # 明らかな誤認識として扱う
    if (
        leaf_pixels
        / pot_pixels
        > 0.99
    ):
        raise ValueError(
            "葉の認識範囲が異常に広くなっています。"
            "植木鉢の検出または葉の抽出に失敗した可能性があります。"
        )

    # =========================
    # 9. 葉面積
    # =========================

    leaf_area = calculate_leaf_area(
        warped_leaf_mask,
        cm_per_pixel
    )

    # =========================
    # 10. 被覆率
    # =========================

    coverage = calculate_coverage(
        warped_leaf_mask,
        warped_mask_inner
    )

    # =========================
    # 11. 色解析
    # =========================

    color_result = analyze_colors(
        warped,
        warped_leaf_mask
    )

    # =========================
    # 12. 結果
    # =========================

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
            warped_mask_inner,

        "leaf_mask":
            warped_leaf_mask,

        "warped_leaf_mask":
            warped_leaf_mask,

        "leaf_area":
            leaf_area,

        "coverage":
            coverage,

        "color_result":
            color_result
    }
