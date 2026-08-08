# color_analysis.py
# PlantAnalyzer Ver3.0

import cv2
import numpy as np

from sklearn.cluster import KMeans

from config import (
    MIN_K,
    MAX_K,
    RANDOM_STATE,
    COLOR_NAMES,
    MAX_COLOR_PIXELS,
)


# =========================
# BGR → Lab
# =========================

def bgr_to_lab(image):
    """BGR画像をLab画像へ変換"""

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )


# =========================
# 葉の画素を取得
# =========================

def extract_leaf_pixels(
    lab_image,
    leaf_mask
):
    """葉領域だけのLab画素を取得"""

    pixels = lab_image[
        leaf_mask > 0
    ]

    if len(pixels) == 0:
        return np.empty(
            (0, 3),
            dtype=np.uint8
        )

    return pixels


# =========================
# KMeans用サンプリング
# =========================

def sample_pixels(pixels):
    """
    KMeans・色評価に使用する画素数を制限する。
    """

    if len(pixels) <= MAX_COLOR_PIXELS:
        return pixels

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    indices = rng.choice(
        len(pixels),
        size=MAX_COLOR_PIXELS,
        replace=False
    )

    return pixels[indices]


# =========================
# KMeans
# =========================

def evaluate_kmeans(
    pixels,
    k
):
    """指定したKでKMeansを実行"""

    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=10
    )

    labels = model.fit_predict(
        pixels
    )

    return model, labels


# =========================
# クラスタ評価
# =========================

def calculate_cluster_score(
    model,
    pixels
):
    """
    KMeansのクラスタ中心間距離を利用して
    Kを簡易評価する。

    大きいほど色が分離している。
    """

    centers = model.cluster_centers_

    if len(centers) < 2:
        return -1.0

    distances = []

    for i in range(
        len(centers)
    ):
        for j in range(
            i + 1,
            len(centers)
        ):

            distance = np.linalg.norm(
                centers[i] -
                centers[j]
            )

            distances.append(
                distance
            )

    if len(distances) == 0:
        return -1.0

    return float(
        np.mean(distances)
    )


# =========================
# 最適K
# =========================

def find_best_k(pixels):
    """
    K=2～MAX_Kから最適なKを決定。

    重いSilhouette Scoreは使用せず、
    クラスタ中心間の色差を利用する。
    """

    if len(pixels) < MIN_K:
        return (
            None,
            None,
            0
        )

    sampled = sample_pixels(
        pixels
    )

    best_model = None
    best_labels = None
    best_k = MIN_K
    best_score = -1.0

    max_k = min(
        MAX_K,
        len(sampled)
    )

    for k in range(
        MIN_K,
        max_k + 1
    ):

        model, labels = (
            evaluate_kmeans(
                sampled,
                k
            )
        )

        score = (
            calculate_cluster_score(
                model,
                sampled
            )
        )

        # Kが増えすぎることを
        # 少し抑制する
        adjusted_score = (
            score -
            (k - MIN_K) * 2.0
        )

        if (
            best_model is None
            or adjusted_score >
            best_score
        ):

            best_score = (
                adjusted_score
            )

            best_model = model
            best_labels = labels
            best_k = k

    # 元画像全体について
    # 最適モデルでラベルを作り直す
    if best_model is not None:

        full_labels = (
            best_model.predict(
                pixels
            )
        )

    else:

        full_labels = np.array(
            [],
            dtype=np.int32
        )

    return (
        best_model,
        full_labels,
        best_k
    )


# =========================
# クラスタ色分類
# =========================

def classify_clusters(
    model
):
    """
    クラスタ中心の明るさを基準に
    Dark Green / Green / Light Green / Yellow
    を割り当てる。

    K=2の場合も、
    色見本を見て手動変更できるようにする。
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

        cluster_info.append({
            "index": i,
            "L": L,
            "A": A,
            "B": B
        })

    # 明るさ順
    cluster_info.sort(
        key=lambda x: x["L"]
    )

    mapping = {}

    n = len(
        cluster_info
    )

    if n == 2:

        mapping[
            cluster_info[0]["index"]
        ] = "Dark Green"

        mapping[
            cluster_info[1]["index"]
        ] = "Light Green"

    elif n == 3:

        mapping[
            cluster_info[0]["index"]
        ] = "Dark Green"

        mapping[
            cluster_info[1]["index"]
        ] = "Green"

        mapping[
            cluster_info[2]["index"]
        ] = "Light Green"

    else:

        # 4以上の場合
        # B値が大きいものをYellow候補にする

        yellow_cluster = max(
            cluster_info,
            key=lambda x: x["B"]
        )

        mapping[
            yellow_cluster["index"]
        ] = "Yellow"

        remain = [
            c
            for c in cluster_info
            if c["index"] !=
            yellow_cluster["index"]
        ]

        remain.sort(
            key=lambda x: x["L"]
        )

        names = [
            "Dark Green",
            "Green",
            "Light Green"
        ]

        for cluster, name in zip(
            remain,
            names
        ):

            mapping[
                cluster["index"]
            ] = name

    return mapping


# =========================
# 色割合
# =========================

def calculate_color_ratio(
    labels,
    mapping
):
    """各色の割合を計算"""

    result = {
        name: 0.0
        for name in COLOR_NAMES
    }

    total = len(
        labels
    )

    if total == 0:
        return result

    for cluster_id, color_name in (
        mapping.items()
    ):

        count = np.sum(
            labels == cluster_id
        )

        result[
            color_name
        ] = (
            count /
            total
        ) * 100.0

    return result


# =========================
# Lab → BGR
# =========================

def lab_to_bgr(
    center
):
    """Labクラスタ中心をBGRに変換"""

    lab = np.float32(
        [[center]]
    )

    bgr = cv2.cvtColor(
        lab,
        cv2.COLOR_Lab2BGR
    )[0][0]

    return np.clip(
        bgr,
        0,
        255
    ).astype(
        np.uint8
    )


# =========================
# クラスタ情報
# =========================

def create_cluster_info(
    model,
    labels
):
    """
    各クラスタについて

    - Lab
    - BGR
    - RGB
    - HSV
    - pixel_count
    - ratio

    を保存する。
    """

    centers = (
        model.cluster_centers_
    )

    cluster_list = []

    total = len(
        labels
    )

    for i, center in enumerate(
        centers
    ):

        bgr = lab_to_bgr(
            center
        )

        hsv = cv2.cvtColor(
            np.uint8(
                [[bgr]]
            ),
            cv2.COLOR_BGR2HSV
        )[0][0]

        pixel_count = int(
            np.sum(
                labels == i
            )
        )

        if total > 0:

            ratio = (
                pixel_count /
                total
            ) * 100.0

        else:

            ratio = 0.0

        cluster_list.append({

            "id": i,

            "lab": [
                float(x)
                for x in center
            ],

            "bgr": [
                int(x)
                for x in bgr
            ],

            "rgb": [
                int(bgr[2]),
                int(bgr[1]),
                int(bgr[0])
            ],

            "hsv": [
                int(x)
                for x in hsv
            ],

            "pixel_count":
                pixel_count,

            "ratio":
                ratio

        })

    return cluster_list


# =========================
# 色分類オーバーレイ
# =========================

def create_color_overlay(
    image,
    leaf_mask,
    labels,
    mapping
):
    """
    各葉画素をクラスタの色名で
    オーバーレイ表示する。
    """

    overlay = image.copy()

    draw_colors = {

        "Dark Green":
            (0, 100, 0),

        "Green":
            (0, 180, 0),

        "Light Green":
            (144, 238, 144),

        "Yellow":
            (0, 255, 255)
    }

    points = np.column_stack(
        np.where(
            leaf_mask > 0
        )
    )

    if len(points) != len(
        labels
    ):
        return overlay

    for i, (y, x) in enumerate(
        points
    ):

        cluster = int(
            labels[i]
        )

        if cluster not in mapping:
            continue

        color_name = mapping[
            cluster
        ]

        if color_name not in draw_colors:
            continue

        overlay[
            y,
            x
        ] = draw_colors[
            color_name
        ]

    return overlay


# =========================
# メイン色解析
# =========================

def analyze_colors(
    image,
    leaf_mask
):
    """葉色解析のメイン関数"""

    if image is None:
        raise ValueError(
            "画像がありません。"
        )

    if leaf_mask is None:
        raise ValueError(
            "葉マスクがありません。"
        )

    lab = bgr_to_lab(
        image
    )

    pixels = extract_leaf_pixels(
        lab,
        leaf_mask
    )

    if len(pixels) == 0:

        return {
            "best_k": 0,

            "ratio": {
                name: 0.0
                for name in COLOR_NAMES
            },

            "clusters": [],

            "mapping": {},

            "overlay":
                image.copy(),

            "cluster_centers":
                np.empty(
                    (0, 3)
                )
        }

    model, labels, best_k = (
        find_best_k(
            pixels
        )
    )

    if model is None:

        return {
            "best_k": 0,

            "ratio": {
                name: 0.0
                for name in COLOR_NAMES
            },

            "clusters": [],

            "mapping": {},

            "overlay":
                image.copy(),

            "cluster_centers":
                np.empty(
                    (0, 3)
                )
        }

    mapping = classify_clusters(
        model
    )

    ratio = calculate_color_ratio(
        labels,
        mapping
    )

    cluster_info = (
        create_cluster_info(
            model,
            labels
        )
    )

    overlay = (
        create_color_overlay(
            image,
            leaf_mask,
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
            model.cluster_centers_

    }


# =========================
# テスト
# =========================

if __name__ == "__main__":

    print(
        "color_analysis.py "
        "loaded successfully."
        )
