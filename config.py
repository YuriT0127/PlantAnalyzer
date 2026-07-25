# データ保存フォルダ
DATA_DIR = Path("data")

# Excel出力フォルダ
OUTPUT_DIR = Path("output")

# ==========================
# ポット設定
# ==========================

NUM_POTS = 12

POT_NAMES = [
    "Pot01",
    "Pot02",
    "Pot03",
    "Pot04",
    "Pot05",
    "Pot06",
    "Pot07",
    "Pot08",
    "Pot09",
    "Pot10",
    "Pot11",
    "Pot12",
]

# ==========================
# 処理区設定
# （あとから自由に変更可能）
# ==========================

TREATMENTS = {

    "Pot01": "Control",
    "Pot02": "Control",
    "Pot03": "Control",
    "Pot04": "Control",

    "Pot05": "Shade30",
    "Pot06": "Shade30",
    "Pot07": "Shade30",
    "Pot08": "Shade30",

    "Pot09": "Shade60",
    "Pot10": "Shade60",
    "Pot11": "Shade60",
    "Pot12": "Shade60",

}

# ==========================
# 葉色判定（HSV）
# ※後から調整します
# ==========================

COLOR_RANGES = {

    "Dark Green": (
        np.array([35, 80, 30]),
        np.array([85, 255, 120])
    ),

    "Green": (
        np.array([35, 40, 40]),
        np.array([85, 255, 255])
    ),

    "Light Green": (
        np.array([25, 30, 80]),
        np.array([45, 255, 255])
    ),

    "Yellow": (
        np.array([15, 40, 80]),
        np.array([35, 255, 255])
    ),

    "Brown": (
        np.array([5, 40, 20]),
        np.array([20, 255, 180])
    ),

}

# グラフ表示順
COLOR_ORDER = [
    "Dark Green",
    "Green",
    "Light Green",
    "Yellow",
    "Brown",
]

# ==========================
# 土面積（Pixel）
# 初回測定後に入力
# ==========================

SOIL_PIXELS = {

    "Pot01": None,
    "Pot02": None,
    "Pot03": None,
    "Pot04": None,
    "Pot05": None,
    "Pot06": None,
    "Pot07": None,
    "Pot08": None,
    "Pot09": None,
    "Pot10": None,
    "Pot11": None,
    "Pot12": None,

}

# ==========================
# 保存フォルダ名
# ==========================

ORIGINAL_FOLDER = "Original"

MASK_FOLDER = "Mask"

OVERLAY_FOLDER = "Overlay"

CSV_NAME = "data.csv"

EXCEL_NAME = "Plant_Analysis.xlsx"
