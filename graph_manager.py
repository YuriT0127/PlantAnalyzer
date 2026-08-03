"""
graph_manager.py
PlantAnalyzer Ver2.0
"""

import matplotlib.pyplot as plt
import pandas as pd


COLOR_MAP = {

    "Dark Green": "#006400",

    "Green": "#00A651",

    "Light Green": "#90EE90",

    "Yellow": "#FFD700"

}


def create_ratio_pie(result_ratio):
    """
    色割合円グラフ
    """

    labels = []

    sizes = []

    colors = []

    for name, value in result_ratio.items():

        if value <= 0:

            continue

        labels.append(name)

        sizes.append(value)

        colors.append(
            COLOR_MAP[name]
        )

    fig, ax = plt.subplots(
        figsize=(6, 6)
    )

    ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90
    )

    ax.set_title(
        "Leaf Color Ratio"
    )

    ax.axis("equal")

    return fig

def create_coverage_graph(df):
    """
    被覆率推移グラフ
    """

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    ax.plot(
        df["Date"],
        df["Coverage (%)"],
        marker="o"
    )

    ax.set_title(
        "Coverage"
    )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        "Coverage (%)"
    )

    ax.grid(True)

    fig.autofmt_xdate()

    return fig

def create_leaf_area_graph(df):
    """
    葉面積推移
    """

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    ax.plot(
        df["Date"],
        df["Leaf Area (cm²)"],
        marker="o"
    )

    ax.set_title(
        "Leaf Area"
    )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        "Leaf Area (cm²)"
    )

    ax.grid(True)

    fig.autofmt_xdate()

    return fig

def create_color_ratio_graph(df):
    """
    色割合推移
    """

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    colors = [
        "Dark Green",
        "Green",
        "Light Green",
        "Yellow"
    ]

    for color in colors:

        if color not in df.columns:
            continue

        ax.plot(
            df["Date"],
            df[color],
            marker="o",
            label=color
        )

    ax.set_title(
        "Leaf Color Transition"
    )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        "Ratio (%)"
    )

    ax.legend()

    ax.grid(True)

    fig.autofmt_xdate()

    return fig
