"""
file_manager.py
Plant Analyzer
ファイル・フォルダ・CSV管理
"""

from pathlib import Path
from datetime import datetime
import pandas as pd
import cv2

from config import (
    DATA_DIR,
    POT_NAMES,
    ORIGINAL_FOLDER,
    MASK_FOLDER,
    OVERLAY_FOLDER,
    CSV_NAME,
)


# ==========================
# フォルダ作成
# ==========================

def create_folders():

    DATA_DIR.mkdir(exist_ok=True)

    for pot in POT_NAMES:

        pot_dir = DATA_DIR / pot

        (pot_dir / ORIGINAL_FOLDER).mkdir(
            parents=True,
            exist_ok=True
        )

        (pot_dir / MASK_FOLDER).mkdir(
            parents=True,
            exist_ok=True
        )

        (pot_dir / OVERLAY_FOLDER).mkdir(
            parents=True,
            exist_ok=True
        )

        csv_file = pot_dir / CSV_NAME

        if not csv_file.exists():

            df = pd.DataFrame(columns=[

                "Date",
                "Time",
                "Treatment",

                "Coverage",

                "LeafPixels",

                "DarkGreenPixel",
                "GreenPixel",
                "LightGreenPixel",
                "YellowPixel",
                "BrownPixel",

                "DarkGreenRatio",
                "GreenRatio",
                "LightGreenRatio",
                "YellowRatio",
                "BrownRatio"

            ])

            df.to_csv(csv_file, index=False)


# ==========================
# 現在日時
# ==========================

def get_datetime():

    now = datetime.now()

    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H-%M-%S")

    return date, time


# ==========================
# 画像保存
# ==========================

def save_original_image(pot_name, image):

    date, time = get_datetime()

    filename = f"{date}_{time}.jpg"

    path = (
        DATA_DIR
        / pot_name
        / ORIGINAL_FOLDER
        / filename
    )

    cv2.imwrite(str(path), image)

    return path


def save_mask_image(pot_name, mask):

    date, time = get_datetime()

    filename = f"{date}_{time}.png"

    path = (
        DATA_DIR
        / pot_name
        / MASK_FOLDER
        / filename
    )

    cv2.imwrite(str(path), mask)

    return path


def save_overlay_image(pot_name, overlay):

    date, time = get_datetime()

    filename = f"{date}_{time}.png"

    path = (
        DATA_DIR
        / pot_name
        / OVERLAY_FOLDER
        / filename
    )

    cv2.imwrite(str(path), overlay)

    return path


# ==========================
# CSVへ追加
# ==========================

def save_csv(

        pot_name,
        treatment,
        coverage,
        leaf_pixels,
        pixels,
        ratios

):

    csv_path = DATA_DIR / pot_name / CSV_NAME

    df = pd.read_csv(csv_path)

    date, time = get_datetime()

    row = {

        "Date": date,
        "Time": time,
        "Treatment": treatment,

        "Coverage": coverage,

        "LeafPixels": leaf_pixels,

        "DarkGreenPixel": pixels["Dark Green"],
        "GreenPixel": pixels["Green"],
        "LightGreenPixel": pixels["Light Green"],
        "YellowPixel": pixels["Yellow"],
        "BrownPixel": pixels["Brown"],

        "DarkGreenRatio": ratios["Dark Green"],
        "GreenRatio": ratios["Green"],
        "LightGreenRatio": ratios["Light Green"],
        "YellowRatio": ratios["Yellow"],
        "BrownRatio": ratios["Brown"]

    }

    df.loc[len(df)] = row

    df.to_csv(csv_path, index=False)


# ==========================
# CSV読込
# ==========================

def load_csv(pot_name):

    csv_path = DATA_DIR / pot_name / CSV_NAME

    if csv_path.exists():

        return pd.read_csv(csv_path)

    return pd.DataFrame()
