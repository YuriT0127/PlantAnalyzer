# excel_manager.py
# PlantAnalyzer Ver3.0

import os
from datetime import datetime

import pandas as pd

from config import EXCEL_FILE, COLOR_NAMES


# =========================
# 保存する列
# =========================

BASE_COLUMNS = [
    "Date",
    "Pot",
    "Condition",
    "Leaf Area (cm2)",
    "Coverage (%)",
    "Best K",
]

COLOR_COLUMNS = [
    f"{name} (%)"
    for name in COLOR_NAMES
]

COLUMNS = (
    BASE_COLUMNS +
    COLOR_COLUMNS
)


# =========================
# 空のDataFrame
# =========================

def create_empty_dataframe():
    """空の実験データを作成"""

    return pd.DataFrame(
        columns=COLUMNS
    )


# =========================
# Excel読み込み
# =========================

def load_data():
    """
    既存のExcelファイルを読み込む。

    ファイルが存在しない場合は
    空のDataFrameを返す。
    """

    if not os.path.exists(
        EXCEL_FILE
    ):
        return create_empty_dataframe()

    try:

        df = pd.read_excel(
            EXCEL_FILE
        )

    except Exception:

        return create_empty_dataframe()

    # 必要な列がない場合は追加
    for column in COLUMNS:

        if column not in df.columns:
            df[column] = 0.0

    # 列順を統一
    df = df[
        COLUMNS
    ]

    return df


# =========================
# 1行作成
# =========================

def create_record(
    date,
    pot,
    condition,
    leaf_area,
    coverage,
    best_k,
    color_ratio
):
    """解析結果から保存用1行を作成"""

    record = {

        "Date":
            date,

        "Pot":
            pot,

        "Condition":
            condition,

        "Leaf Area (cm2)":
            float(leaf_area),

        "Coverage (%)":
            float(coverage),

        "Best K":
            int(best_k)

    }

    for color in COLOR_NAMES:

        record[
            f"{color} (%)"
        ] = float(
            color_ratio.get(
                color,
                0.0
            )
        )

    return record


# =========================
# データ追加
# =========================

def add_record(
    record
):
    """
    Excelに1行追加。

    同じDate・Pot・Conditionが
    既に存在する場合は更新する。
    """

    df = load_data()

    date = record["Date"]
    pot = record["Pot"]
    condition = record[
        "Condition"
    ]

    if len(df) > 0:

        duplicate = (
            (df["Date"].astype(str)
             == str(date))
            &
            (df["Pot"].astype(str)
             == str(pot))
            &
            (df["Condition"].astype(str)
             == str(condition))
        )

        if duplicate.any():

            index = df.index[
                duplicate
            ][0]

            for column in COLUMNS:

                df.loc[
                    index,
                    column
                ] = record.get(
                    column,
                    0
                )

        else:

            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        [record]
                    )
                ],
                ignore_index=True
            )

    else:

        df = pd.DataFrame(
            [record],
            columns=COLUMNS
        )

    save_data(
        df
    )

    return df


# =========================
# Excel保存
# =========================

def save_data(
    df
):
    """DataFrameをExcelに保存"""

    df = df.copy()

    # 日付順
    if "Date" in df.columns:

        try:

            df["_sort_date"] = (
                pd.to_datetime(
                    df["Date"],
                    errors="coerce"
                )
            )

            df = df.sort_values(
                "_sort_date"
            )

            df = df.drop(
                columns=[
                    "_sort_date"
                ]
            )

        except Exception:
            pass

    df.to_excel(
        EXCEL_FILE,
        index=False
    )


# =========================
# CSV的にDataFrame取得
# =========================

def get_data():
    """現在の保存データを返す"""

    return load_data()


# =========================
# 全削除
# =========================

def clear_data():
    """保存データを全削除"""

    df = create_empty_dataframe()

    save_data(
        df
    )

    return df


# =========================
# 今日の日付
# =========================

def today_string():
    """今日の日付をYYYY-MM-DDで返す"""

    return datetime.now().strftime(
        "%Y-%m-%d"
    )


# =========================
# テスト
# =========================

if __name__ == "__main__":

    print(
        "excel_manager.py "
        "loaded successfully."
    )

    print(
        load_data()
      )
