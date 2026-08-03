"""
color_analysis.py
PlantAnalyzer Ver2.0

葉色解析
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


def rgb_to_lab(image):
    """
    BGR画像をLab画像へ変換
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)


def extract_leaf_pixels(lab_image, leaf_mask):
    """
    葉領域だけ取り出す
    """

    pixels = lab_image[leaf_mask > 0]

    if len(pixels) == 0:
        return np.empty((0, 3), dtype=np.uint8)

    return pixels


def evaluate_kmeans(pixels, k):
    """
    指定KでKMeans実行
    """

    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init="auto"
    )

    labels = model.fit_predict(pixels)

    return model, labels

def calculate_scores(pixels, labels):
    """
    Silhouette Score と
    Davies-Bouldin Index を計算
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

    return silhouette, db_index


def find_best_k(pixels):
    """
    K=2～6から
    最適クラスタ数を決定
    """

    best_model = None
    best_labels = None

    best_score = -9999

    best_k = MIN_K

    for k in range(MIN_K, MAX_K + 1):

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

    return best_model, best_labels, best_k

def classify_clusters(model):
    """
    クラスタ中心から

    Dark Green
    Green
    Light Green
    Yellow

    を自動判定する
    """

    centers = model.cluster_centers_

    cluster_info = []

    for i, center in enumerate(centers):

        L = float(center[0])
        A = float(center[1])
        B = float(center[2])

        cluster_info.append({

            "index": i,

            "L": L,

            "A": A,

            "B": B

        })

    # 明るさ(L)で並べる
    cluster_info.sort(
        key=lambda x: x["L"]
    )

    mapping = {}

    n = len(cluster_info)

    if n == 2:

        mapping[cluster_info[0]["index"]] = "Dark Green"
        mapping[cluster_info[1]["index"]] = "Light Green"

    elif n == 3:

        mapping[cluster_info[0]["index"]] = "Dark Green"
        mapping[cluster_info[1]["index"]] = "Green"
        mapping[cluster_info[2]["index"]] = "Light Green"

    else:

        mapping[cluster_info[0]["index"]] = "Dark Green"
        mapping[cluster_info[1]["index"]] = "Green"

        if n >= 4:

            # b*が最も大きいクラスタをYellowにする
            yellow_cluster = max(
                cluster_info,
                key=lambda x: x["B"]
            )

            mapping[yellow_cluster["index"]] = "Yellow"

            # Yellow以外を明るさ順に並べ直す
            remain = [
                c for c in cluster_info
                if c["index"] != yellow_cluster["index"]
            ]

            remain.sort(
                key=lambda x: x["L"]
            )

            mapping[remain[0]["index"]] = "Dark Green"
            mapping[remain[1]["index"]] = "Green"
            mapping[remain[2]["index"]] = "Light Green"

    return mapping


def calculate_color_ratio(labels, mapping):
    """
    各色割合(%)を計算
    """

    result = {
        name: 0.0
        for name in COLOR_NAMES
    }

    total = len(labels)

    if total == 0:
        return result

    for cluster_id, color_name in mapping.items():

        count = np.sum(labels == cluster_id)

        result[color_name] = (
            count / total
        ) * 100

    return result

def create_cluster_info(model, labels):
    """
    各クラスタの情報を作成
    """

    centers = model.cluster_centers_

    cluster_list = []

    total = len(labels)

    for i, center in enumerate(centers):

        lab = np.float32([[center]])

        bgr = cv2.cvtColor(
            lab,
            cv2.COLOR_Lab2BGR
        )[0][0]

        bgr = np.clip(
            bgr,
            0,
            255
        ).astype(np.uint8)

        hsv = cv2.cvtColor(
            np.uint8([[bgr]]),
            cv2.COLOR_BGR2HSV
        )[0][0]

        pixel_count = int(
            np.sum(labels == i)
        )

        cluster_list.append({

            "id": i,

            "lab": center.tolist(),

            "bgr": bgr.tolist(),

            "hsv": hsv.tolist(),

            "pixel_count": pixel_count,

            "ratio": (
                pixel_count /
                total
            ) * 100

        })

    return cluster_list

def create_color_overlay(image, leaf_mask, labels, mapping):
    """
    色分類画像を作成
    """

    overlay = image.copy()

    draw_colors = {

        "Dark Green": (0, 100, 0),

        "Green": (0, 255, 0),

        "Light Green": (144, 238, 144),

        "Yellow": (0, 255, 255)

    }

    points = np.column_stack(
        np.where(leaf_mask > 0)
    )

    for i, (y, x) in enumerate(points):

        cluster = labels[i]

        color_name = mapping[cluster]

        overlay[y, x] = draw_colors[color_name]

    return overlay


def analyze_colors(image, leaf_mask):
    """
    色解析メイン関数
    """

    lab = rgb_to_lab(image)

    pixels = extract_leaf_pixels(
        lab,
        leaf_mask
    )

    if len(pixels) == 0:
        return {
            "best_k": 0,
            "ratio": {
                name: 0
                for name in COLOR_NAMES
            },
            
            "clusters": [],
            "mapping": {},
            "overlay": image.copy()
        }

    model, labels, best_k = find_best_k(
        pixels
    )

    mapping = classify_clusters(
        model
    )

    ratio = calculate_color_ratio(
        labels,
        mapping
    )
    cluster_info = create_cluster_info(
    model,
    labels
    )

    overlay = create_color_overlay(
        image,
        leaf_mask,
        labels,
        mapping
    )

    return {

    "best_k": best_k,

    "ratio": ratio,

    "clusters": cluster_info,

    "mapping": mapping,

    "overlay": overlay,

    "cluster_centers": model.cluster_centers_

    }


if __name__ == "__main__":

    print("color_analysis.py loaded successfully.")
