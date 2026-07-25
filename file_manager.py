"""
file_manager.py
ファイル・フォルダ管理
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

        (pot_dir / ORIGINAL_FOLDER).mkdir(parents=True, exist_ok=True)
        (pot_dir / MASK_FOLDER).mkdir(parents=True, exist_ok=True)
        (pot_dir / OVERLAY_FOLDER).mkdir(parents=True, exist_ok=True)

        csv_path = pot_dir / CSV_NAME

        if not csv_path.exists():

            df = pd.DataFrame(columns=[
                "Date",
                "Time",
                "Treatment",
                "Coverage",
                "LeafPixels",
                "DarkGreen",
                "Green",
                "LightGreen",
                "Yellow",
                "Brown",
                "DarkGreenRatio",
                "GreenRatio",
                "LightGreenRatio",
                "YellowRatio",
                "BrownRatio",
            ])

            df.to_csv(csv_path, index=False)

# ==========================
# 現在日時
# ==========================

def now():

    t = datetime.now()

    return (
        t.strftime("%Y-%m-%d"),
        t.strftime("%H-%M-%S")
    )

# ==========================
# 元画像保存
# ==========================

def save_original(pot_name, image):

    date, time = now()

    filename = f"{date}_{time}.jpg"

    path = (
        DATA_DIR
        / pot_name
        / ORIGINAL_FOLDER
        / filename
    )

    cv2.imwrite(str(path), image)

    return path

# ==========================
# マスク保存
# ==========================

def save_mask(pot_name, image):

    date, time = now()

    filename = f"{date}_{time}.png"

    path = (
        DATA_DIR
        / pot_name
        / MASK_FOLDER
        / filename
    )

    cv2.imwrite(str(path), image)

    return path

# ==========================
# Overlay保存
# ==========================

def save_overlay(pot_name, image):

    date, time = now()

    filename = f"{date}_{time}.png"

    path = (
        DATA_DIR
        / pot_name
        / OVERLAY_FOLDER
        / filename
    )

    cv2.imwrite(str(path), image)

    return path
