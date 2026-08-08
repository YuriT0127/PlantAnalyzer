# graph_manager.py
# PlantAnalyzer Ver3.0

import pandas as pd
import plotly.express as px

from config import COLOR_NAMES


# =========================
# 共通前処理
# =========================

def prepare_dataframe(df):
    """グラフ用にDataFrameを整える"""

    if df is None:
        return pd.DataFrame()

    if len(df) == 0:
        return df.copy()

    result = df.copy()

    if "Date" in result.columns:
        result["Date"] = pd.to_datetime(
            result["Date"],
            errors="coerce"
        )

    return result


# =========================
# 葉面積の推移
# =========================

def leaf_area_graph(df):
    """日ごとの葉面積を表示"""

    data = prepare_dataframe(df)

    if data.empty:
        return None

    fig = px.line(
        data,
        x="Date",
        y="Leaf Area (cm2)",
        color="Pot",
        markers=True,
        title="Leaf Area Over Time"
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Leaf Area (cm²)",
        legend_title="Pot"
    )

    return fig


# =========================
# 被覆率の推移
# =========================

def coverage_graph(df):
    """日ごとの被覆率を表示"""

    data = prepare_dataframe(df)

    if data.empty:
        return None

    fig = px.line(
        data,
        x="Date",
        y="Coverage (%)",
        color="Pot",
        markers=True,
        title="Coverage Over Time"
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Coverage (%)",
        legend_title="Pot"
    )

    return fig


# =========================
# 条件別の被覆率
# =========================

def condition_coverage_graph(df):
    """条件別の被覆率を表示"""

    data = prepare_dataframe(df)

    if data.empty:
        return None

    if "Condition" not in data.columns:
        return None

    fig = px.line(
        data,
        x="Date",
        y="Coverage (%)",
        color="Condition",
        markers=True,
        title="Coverage by Condition"
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Coverage (%)",
        legend_title="Condition"
    )

    return fig


# =========================
# 色割合
# =========================

def color_ratio_graph(
    df,
    pot=None
):
    """
    色割合の推移を表示。

    potを指定した場合、
    そのPotだけを表示する。
    """

    data = prepare_dataframe(df)

    if data.empty:
        return None

    if pot is not None:
        data = data[
            data["Pot"].astype(str)
            == str(pot)
        ]

    if data.empty:
        return None

    existing_colors = [
        color
        for color in COLOR_NAMES
        if f"{color} (%)"
        in data.columns
    ]

    if not existing_colors:
        return None

    value_columns = [
        f"{color} (%)"
        for color in existing_colors
    ]

    # 日付・Pot・Conditionを残す
    id_columns = [
        column
        for column in [
            "Date",
            "Pot",
            "Condition"
        ]
        if column in data.columns
    ]

    melted = data.melt(
        id_vars=id_columns,
        value_vars=value_columns,
        var_name="Color",
        value_name="Percentage"
    )

    melted["Color"] = (
        melted["Color"]
        .str.replace(
            " (%)",
            "",
            regex=False
        )
    )

    fig = px.line(
        melted,
        x="Date",
        y="Percentage",
        color="Color",
        markers=True,
        title="Leaf Color Composition"
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Percentage (%)",
        legend_title="Color"
    )

    return fig


# =========================
# 色割合の積み上げ
# =========================

def color_stacked_graph(
    df,
    pot=None
):
    """色割合を積み上げ面グラフで表示"""

    data = prepare_dataframe(df)

    if data.empty:
        return None

    if pot is not None:
        data = data[
            data["Pot"].astype(str)
            == str(pot)
        ]

    if data.empty:
        return None

    existing_colors = [
        color
        for color in COLOR_NAMES
        if f"{color} (%)"
        in data.columns
    ]

    if not existing_colors:
        return None

    value_columns = [
        f"{color} (%)"
        for color in existing_colors
    ]

    id_columns = [
        column
        for column in [
            "Date",
            "Pot",
            "Condition"
        ]
        if column in data.columns
    ]

    melted = data.melt(
        id_vars=id_columns,
        value_vars=value_columns,
        var_name="Color",
        value_name="Percentage"
    )

    melted["Color"] = (
        melted["Color"]
        .str.replace(
            " (%)",
            "",
            regex=False
        )
    )

    fig = px.area(
        melted,
        x="Date",
        y="Percentage",
        color="Color",
        groupnorm="percent",
        title="Leaf Color Composition Over Time"
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Percentage (%)",
        legend_title="Color"
    )

    return fig


# =========================
# 条件別の葉面積
# =========================

def condition_leaf_area_graph(df):
    """条件別の葉面積推移"""

    data = prepare_dataframe(df)

    if data.empty:
        return None

    if "Condition" not in data.columns:
        return None

    fig = px.line(
        data,
        x="Date",
        y="Leaf Area (cm2)",
        color="Condition",
        markers=True,
        title="Leaf Area by Condition"
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Leaf Area (cm²)",
        legend_title="Condition"
    )

    return fig


# =========================
# 最適Kの推移
# =========================

def best_k_graph(df):
    """最適Kの推移"""

    data = prepare_dataframe(df)

    if data.empty:
        return None

    if "Best K" not in data.columns:
        return None

    fig = px.line(
        data,
        x="Date",
        y="Best K",
        color="Pot",
        markers=True,
        title="Best K Over Time"
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Best K",
        legend_title="Pot"
    )

    return fig
