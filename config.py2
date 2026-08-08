# config.py
# PlantAnalyzer Ver3.0

# =========================
# 基本設定
# =========================

APP_TITLE = "PlantAnalyzer"

# 植木鉢の実寸（cm）
POT_SIZE_CM = 17.0


# =========================
# 色解析設定
# =========================

# 自動決定するクラスタ数の範囲
MIN_K = 2
MAX_K = 6

# KMeans
RANDOM_STATE = 42


# 表示・分類に使用する色名
COLOR_NAMES = [
    "Dark Green",
    "Green",
    "Light Green",
    "Yellow"
]


# 色名選択用
COLOR_OPTIONS = COLOR_NAMES.copy()


# 色見本・オーバーレイ用のBGR値
# OpenCVはBGR順
DRAW_COLORS = {
    "Dark Green": (0, 100, 0),
    "Green": (0, 180, 0),
    "Light Green": (144, 238, 144),
    "Yellow": (0, 255, 255)
}


# =========================
# KMeans高速化
# =========================

# 色解析に使用する最大ピクセル数
# 元画像が大きい場合でも処理が重くなりすぎないようにする
MAX_COLOR_PIXELS = 10000


# =========================
# グラフ設定
# =========================

COLOR_GRAPH_ORDER = [
    "Dark Green",
    "Green",
    "Light Green",
    "Yellow"
]


# =========================
# 実験条件
# =========================

POT_OPTIONS = [
    f"Pot{i:02d}"
    for i in range(1, 13)
]

CONDITION_OPTIONS = [
    "Control",
    "Shade30",
    "Shade60"
]


# =========================
# ファイル設定
# =========================

EXCEL_FILE = "plant_data.xlsx"
