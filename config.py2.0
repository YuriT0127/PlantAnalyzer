"""
PlantAnalyzer Ver.2.0
config.py
各種設定
"""

from pathlib import Path
import numpy as np

# ==========================================
# 基本フォルダ
# ==========================================

BASE_DIR = Path(__file__).parent

DATA_DIR = BASE_DIR / "data"

ORIGINAL_DIR = DATA_DIR / "original"
MASK_DIR = DATA_DIR / "mask"
OVERLAY_DIR = DATA_DIR / "overlay"

CSV_DIR = DATA_DIR / "csv"
EXCEL_DIR = DATA_DIR / "excel"

HISTORY_DIR = BASE_DIR / "history"

# ==========================================
# 作成
# ==========================================

for folder in [
    DATA_DIR,
    ORIGINAL_DIR,
    MASK_DIR,
    OVERLAY_DIR,
    CSV_DIR,
    EXCEL_DIR,
    HISTORY_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)

# ==========================================
# ポット設定
# ==========================================

POT_COUNT = 12

POT_NAMES = [
    "Pot1",
    "Pot2",
    "Pot3",
    "Pot4",
    "Pot5",
    "Pot6",
    "Pot7",
    "Pot8",
    "Pot9",
    "Pot10",
    "Pot11",
    "Pot12",
]

# ==========================================
# 処理区
# ==========================================

TREATMENTS = {
    "Pot1": "Control",
    "Pot2": "Control",
    "Pot3": "Control",
    "Pot4": "Treatment A",
    "Pot5": "Treatment A",
    "Pot6": "Treatment A",
    "Pot7": "Treatment B",
    "Pot8": "Treatment B",
    "Pot9": "Treatment B",
    "Pot10": "Treatment C",
    "Pot11": "Treatment C",
    "Pot12": "Treatment C",
}


# ==========================================
# 画像設定
# ==========================================

IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
]

# ==========================================
# 色判定(BGR)
# ==========================================

GREEN_LOWER = (35, 40, 40)
GREEN_UPPER = (90, 255, 255)

YELLOW_LOWER = (20, 40, 40)
YELLOW_UPPER = (35, 255, 255)

BROWN_LOWER = (5, 30, 20)
BROWN_UPPER = (20, 255, 255)

# ==========================================
# Excel
# ==========================================

EXCEL_FILENAME = "PlantAnalyzer_Result.xlsx"

# ==========================================
# CSV
# ==========================================

CSV_FILENAME = "PlantAnalyzer_Result.csv"
CSV_NAME = CSV_FILENAME
ORIGINAL_FOLDER = "original"
MASK_FOLDER = "mask"
OVERLAY_FOLDER = "overlay"

# ==========================================
# グラフ
# ==========================================

GRAPH_WIDTH = 10
GRAPH_HEIGHT = 6
GRAPH_DPI = 150

# ==========================================
# アプリ名
# ==========================================

APP_NAME = "PlantAnalyzer Ver.2.0"

VERSION = "2.0"

# ==========================================
# file_manager.py 互換設定
# ==========================================

# ==========================================
# image_analysis.py 互換設定
# ==========================================

COLOR_RANGES = {

    "Dark Green": (
        np.array([35, 80, 30]),
        np.array([60, 255, 180])
    ),

    "Green": (
        np.array([35, 40, 40]),
        np.array([90, 255, 255])
    ),

    "Light Green": (
        np.array([40, 20, 120]),
        np.array([90, 120, 255])
    ),

    "Yellow": (
        np.array([20, 40, 40]),
        np.array([35, 255, 255])
    ),

    "Brown": (
        np.array([5, 30, 20]),
        np.array([20, 255, 255])
    ),
}


SOIL_PIXELS = {
    pot: None
    for pot in POT_NAMES
}
