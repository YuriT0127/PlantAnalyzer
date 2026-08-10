"""
color_analysis.py
PlantAnalyzer Ver2.1

葉色解析
・leaf_mask内から植物らしい色を自動抽出
・茶色、土、黒、灰色などを除外
・Dark Green / Green / Light Green / Yellow に分類
"""

import cv2
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score

from config import (
    MIN_K,
    MAX_K,
    RANDOM_STATE,
    COLOR_NAMES
)


# =========================================================
# 色変換
# =========================================================

def rgb_to_lab(image):
    """
    BGR画像をLab画像へ変換
    """
    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )


# =========================================================
# 植物候補マスク作成
# =========================================================

def create_plant_color_mask(image, leaf_mask):
    """
    leaf_maskの中から、
    緑～黄色の植物らしい色だけを抽出する。

    茶色、土、黒、灰色などは除外する。
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # -----------------------------------------------------
    # 緑
    # OpenCV HSVのHは0～179
    # -----------------------------------------------------

    green = (
        (h >= 30) &
        (h <= 95) &
        (s >= 45) &
        (v >= 35)
    )

    # -----------------------------------------------------
    # 黄色
    # 黄色い葉も残す
    # -----------------------------------------------------

    yellow = (
        (h >= 18) &
        (h < 35) &
        (s >= 55) &
        (v >= 50)
    )

    # -----------------------------------------------------
    # 緑色の追加条件
    #
    # 茶色はRがGより強くなりやすいので、
    # 緑についてはG優勢を要求する
    # -----------------------------------------------------

    b = image[:, :, 0].astype(np.int16)
    g = image[:, :, 1].astype(np.int16)
    r = image[:, :, 2].astype(np.int16)

    green_dominant = (
        (g >= r) &
        (g >= b)
    )

    green = green & green_dominant

    # -----------------------------------------------------
    # 黄色はR,Gが強くBが弱い
    # -----------------------------------------------------

    yellow_dominant = (
        (r >= b + 20) &
        (g >= b + 15)
    )

    yellow = yellow & yellow_dominant

    # 植物候補
    plant_mask = (
        (green | yellow) &
        (leaf_mask > 0)
    )

    # -----------------------------------------------------
    # 小さなノイズを除去
    # -----------------------------------------------------

    plant_mask = (
        plant_mask.astype(np.uint8) * 255
    )

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    plant_mask = cv2.morphologyEx(
        plant_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    plant_mask = cv2.morphologyEx(
        plant_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return plant_mask


# =========================================================
# 葉ピクセル抽出
# =========================================================

def extract_leaf_pixels(
    lab_image,
    image,
    leaf_mask
):
    """
    leaf_maskの中から植物色だけを抽出し、
    Labピクセルを返す。

    戻り値:
        pixels
        plant_mask
    """

    plant_mask = create_plant_color_mask(
        image,
        leaf_mask
    )

    pixels = lab_image[
        plant_mask > 0
    ]

    if len(pixels) == 0:
        return (
            np.empty(
                (0, 3),
                dtype=np.uint8
            ),
            plant_mask
        )

    return (
        pixels,
        plant_mask
    )


# =========================================================
# KMeans
# =========================================================

def evaluate_kmeans(
    pixels,
    k
):
    """
    指定KでKMeans実行
    """

    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init="auto"
    )

    labels = model.fit_predict(
        pixels
    )

    return model, labels


def calculate_scores(
    pixels,
    labels
):
    """
    Silhouette Score と
    Davies-Bouldin Index
    """

    if len(np.unique(labels)) < 2:
        return -1.0, np.inf

    silhouette = silhouette_score(
        pixels,
        labels
    )

    db_index = davies_bouldin_score(
        pixels,
        labels
    )

    return (
        silhouette,
        db_index
    )


def find_best_k(pixels):
    """
    K=MIN_K～MAX_Kから
    最適クラスタ数を決定
    """

    best_model = None
    best_labels = None

    best_score = -9999

    best_k = MIN_K

    for k in range(
        MIN_K,
        MAX_K + 1
    ):

        if len(pixels) <= k:
            break

        model, labels = evaluate_kmeans(
            pixels,
            k
        )

        silhouette, db = calculate_scores(
            pixels,
            labels
        )

        score = silhouette - db

        if score > best_score:

            best_score = score

            best_model = model

            best_labels = labels

            best_k = k

    return (
        best_model,
        best_labels,
        best_k
    )


# =========================================================
# クラスタ分類
# =========================================================

def classify_clusters(
    model
):
    """
    クラスタ中心から

    Dark Green
    Green
    Light Green
    Yellow

    を自動判定する。
    """

    centers = (
        model.cluster_centers_
    )

    cluster_info = []

    for i, center in enumerate(
        centers
    ):

        L = float(center[0])
        A = float(center[1])
        B = float(center[2])

        # Lab → BGR
        lab_pixel = np.float32(
            [[center]]
        )

        bgr = cv2.cvtColor(
            lab_pixel,
            cv2.COLOR_Lab2BGR
        )[0][0]

        bgr = np.clip(
            bgr,
            0,
            255
        ).astype(
            np.uint8
        )

        hsv = cv2.cvtColor(
            np.uint8([[bgr]]),
            cv2.COLOR_BGR2HSV
        )[0][0]

        cluster_info.append({
            "index": i,
            "L": L,
            "A": A,
            "B": B,
            "H": float(hsv[0]),
            "S": float(hsv[1]),
            "V": float(hsv[2]),
            "bgr": bgr.tolist()
        })

    # -----------------------------------------------------
    # Yellow候補
    # -----------------------------------------------------

    yellow_candidates = [
        c
        for c in cluster_info
        if 15 <= c["H"] <= 40
    ]

    mapping = {}

    yellow_index = None

    if yellow_candidates:

        yellow = max(
            yellow_candidates,
            key=lambda x: (
                x["R"] if "R" in x else
                x["B"]
            )
        )

        # BGRなので黄色はBが比較的小さい
        yellow = min(
            yellow_candidates,
            key=lambda x: x["B"]
        )

        yellow_index = yellow[
            "index"
        ]

        mapping[
            yellow_index
        ] = "Yellow"

    # -----------------------------------------------------
    # Yellow以外
    # -----------------------------------------------------

    remaining = [
        c
        for c in cluster_info
        if c["index"] != yellow_index
    ]

    # 明るさ順
    remaining.sort(
        key=lambda x: x["L"]
    )

    n = len(remaining)

    if n == 1:

        mapping[
            remaining[0]["index"]
        ] = "Green"

    elif n == 2:

        mapping[
            remaining[0]["index"]
        ] = "Dark Green"

        mapping[
            remaining[1]["index"]
        ] = "Light Green"

    else:

        mapping[
            remaining[0]["index"]
        ] = "Dark Green"

        mapping[
            remaining[1]["index"]
        ] = "Green"

        for c in remaining[2:]:

            mapping[
                c["index"]
            ] = "Light Green"

    return mapping


# =========================================================
# 色割合
# =========================================================

def calculate_color_ratio(
    labels,
    mapping
):
    """
    各色割合(%)
    """

    result = {
        name: 0.0
        for name in COLOR_NAMES
    }

    total = len(labels)

    if total == 0:
        return result

    for cluster_id, color_name in mapping.items():

        count = np.sum(
            labels == cluster_id
        )

        result[color_name] = (
            count / total
        ) * 100

    return result


# =========================================================
# クラスタ情報
# =========================================================

def create_cluster_info(
    model,
    labels
):
    """
    各クラスタの情報を作成
    """

    centers = (
        model.cluster_centers_
    )

    cluster_list = []

    total = len(labels)

    for i, center in enumerate(
        centers
    ):

        lab = np.float32(
            [[center]]
        )

        bgr = cv2.cvtColor(
            lab,
            cv2.COLOR_Lab2BGR
        )[0][0]

        bgr = np.clip(
            bgr,
            0,
            255
        ).astype(
            np.uint8
        )

        hsv = cv2.cvtColor(
            np.uint8([[bgr]]),
            cv2.COLOR_BGR2HSV
        )[0][0]

        pixel_count = int(
            np.sum(
                labels == i
            )
        )

        cluster_list.append({

            "id": i,

            "lab": center.tolist(),

            "bgr": bgr.tolist(),

            # RGBも明示
            "rgb": [
                int(bgr[2]),
                int(bgr[1]),
                int(bgr[0])
            ],

            "hsv": hsv.tolist(),

            "pixel_count":
                pixel_count,

            "ratio":
                (
                    pixel_count /
                    total
                ) * 100
        })

    return cluster_list


# =========================================================
# Overlay
# =========================================================

def create_color_overlay(
    image,
    plant_mask,
    labels,
    mapping
):
    """
    実際に色解析したピクセルだけを
    Overlayする。
    """

    overlay = image.copy()

    draw_colors = {

        "Dark Green":
            (0, 100, 0),

        "Green":
            (0, 255, 0),

        "Light Green":
            (144, 238, 144),

        "Yellow":
            (0, 255, 255)
    }

    points = np.column_stack(
        np.where(
            plant_mask > 0
        )
    )

    # 念のため長さを一致
    count = min(
        len(points),
        len(labels)
    )

    for i in range(count):

        y, x = points[i]

        cluster = int(
            labels[i]
        )

        if cluster not in mapping:
            continue

        color_name = mapping[
            cluster
        ]

        overlay[
            y,
            x
        ] = draw_colors[
            color_name
        ]

    return overlay


# =========================================================
# メイン
# =========================================================

def analyze_colors(
    image,
    leaf_mask
):
    """
    色解析メイン関数
    """

    lab = rgb_to_lab(
        image
    )

    pixels, plant_mask = (
        extract_leaf_pixels(
            lab,
            image,
            leaf_mask
        )
    )

    # -----------------------------------------------------
    # 葉候補がない
    # -----------------------------------------------------

    if len(pixels) == 0:

        return {

            "best_k": 0,

            "ratio": {
                name: 0
                for name in COLOR_NAMES
            },

            "clusters": [],

            "mapping": {},

            "overlay":
                image.copy(),

            "cluster_centers":
                [],

            "plant_mask":
                plant_mask
        }

    # -----------------------------------------------------
    # KMeans
    # -----------------------------------------------------

    model, labels, best_k = (
        find_best_k(
            pixels
        )
    )

    # -----------------------------------------------------
    # 色分類
    # -----------------------------------------------------

    mapping = classify_clusters(
        model
    )

    # -----------------------------------------------------
    # 割合
    # -----------------------------------------------------

    ratio = calculate_color_ratio(
        labels,
        mapping
    )

    # -----------------------------------------------------
    # クラスタ情報
    # -----------------------------------------------------

    cluster_info = (
        create_cluster_info(
            model,
            labels
        )
    )

    # -----------------------------------------------------
    # Overlay
    # -----------------------------------------------------

    overlay = (
        create_color_overlay(
            image,
            plant_mask,
            labels,
            mapping
        )
    )

    return {

        "best_k":
            best_k,

        "ratio":
            ratio,

        "clusters":
            cluster_info,

        "mapping":
            mapping,

        "overlay":
            overlay,

        "cluster_centers":
            model.cluster_centers_,

        "plant_mask":
            plant_mask
    }


# =========================================================
# テスト
# =========================================================

if __name__ == "__main__":

    print(
        "color_analysis.py loaded successfully."
  )
