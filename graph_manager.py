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


