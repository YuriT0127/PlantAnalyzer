"""
excel_manager.py
PlantAnalyzer Ver2.0
"""

import os
import pandas as pd

from config import (

    EXCEL_FILE,

    RESULT_COLUMNS,

    DATE_COLUMN,

    POT_COLUMN,

    LIGHT_COLUMN,

    STIMULUS_COLUMN,

    REPLICATE_COLUMN,

    COVERAGE_COLUMN,

    LEAF_AREA_COLUMN,

    DARK_GREEN_COLUMN,

    GREEN_COLUMN,

    LIGHT_GREEN_COLUMN,

    YELLOW_COLUMN
)


def create_record(
    date,
    pot_name,
    light,
    stimulus,
    replicate,
    result_ratio,
    coverage,
    leaf_area
):
    return {

    DATE_COLUMN: date,

    POT_COLUMN: pot_name,

    LIGHT_COLUMN: light,

    STIMULUS_COLUMN: stimulus,

    REPLICATE_COLUMN: replicate,

    COVERAGE_COLUMN: coverage,

    LEAF_AREA_COLUMN: leaf_area,

    DARK_GREEN_COLUMN: result_ratio.get("Dark Green", 0),

    GREEN_COLUMN: result_ratio.get("Green", 0),

    LIGHT_GREEN_COLUMN: result_ratio.get("Light Green", 0),

    YELLOW_COLUMN: result_ratio.get("Yellow", 0)

    }
    """
    1レコード作成
    """

    record = {

        "撮影日": date,

        "ポット名": pot_name,

        "光条件":
            LIGHT_CONDITION[
                pot_name
            ],

        "刺激条件":
            STIMULUS[
                pot_name
            ],

        "反復":
            REPLICATE[
                pot_name
            ],

        "被覆率(%)":
            round(
                coverage,
                2
            ),

        "総葉面積(cm²)":
            round(
                leaf_area,
                2
            )
    }

    for color in [

        "Dark Green",

        "Green",

        "Light Green",

        "Yellow"

    ]:

        record[color] = round(
            result_ratio.get(
                color,
                0
            ),
            2
        )

    return record

def save_to_excel(
    date,
    pot_name,
    result_ratio,
    coverage,
    leaf_area
):
    """
    Excelへ保存
    """

    record = create_record(
        date,
        pot_name,
        light,
        stimulus,
        replicate,
        result_ratio,
        coverage,
        leaf_area
    )

    new_df = pd.DataFrame(

        [new_row],

        columns=RESULT_COLUMNS
    )

    if os.path.exists(EXCEL_FILE):

        old_df = pd.read_excel(
            EXCEL_FILE
        )

        df = pd.concat(
            [
                old_df,
                new_df
            ],
            ignore_index=True
        )

    else:

        df = new_df

    df.to_excel(

        EXCEL_FILE,

        index=False

    )

    return df

def dataframe_to_csv(df):
    """
    DataFrameをCSVデータへ変換
    """

    return df.to_csv(
        index=False,
        encoding="utf-8-sig"
    ).encode("utf-8-sig")


def get_download_filename(date):
    """
    ダウンロード用ファイル名
    """

    return f"PlantAnalyzer_{date}.csv"


def load_excel():
    """
    既存Excelを読み込む
    """

    if os.path.exists(EXCEL_FILE):

        return pd.read_excel(
            EXCEL_FILE
        )

    return pd.DataFrame()

def get_summary():
    """
    保存済みデータの概要を取得
    """

    df = load_excel()

    if df.empty:

        return {

            "count": 0,

            "latest": pd.DataFrame()

        }

    return {

        "count": len(df),

        "latest": df.tail(5)

    }


if __name__ == "__main__":

    print(
        "excel_manager.py loaded successfully."
    )


