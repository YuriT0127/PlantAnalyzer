"""
app.py
PlantAnalyzer Ver2.1
"""

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from datetime import date

from image_analysis import analyze_image

from excel_manager import (
    save_to_excel,
    load_excel
)

from graph_manager import (
    create_ratio_pie,
    create_coverage_graph,
    create_leaf_area_graph,
    create_color_ratio_graph
)

from config import (

    POT_NAMES,

    LIGHT_CONDITION,
    STIMULUS,
    REPLICATE,

    COLOR_OPTIONS,
    COLOR_PATCH_SIZE,

    APP_NAME

)

# ==================================================
# Streamlit
# ==================================================

st.set_page_config(

    page_title=APP_NAME,

    layout="wide"

)

st.title(APP_NAME)

st.markdown("---")

# ==================================================
# 撮影情報
# ==================================================

col1, col2 = st.columns(2)

with col1:

    measure_date = st.date_input(

        "撮影日",

        value=date.today()

    )

with col2:

    pot_name = st.selectbox(

        "ポット",

        POT_NAMES

    )

light = LIGHT_CONDITION[pot_name]

stimulus = STIMULUS[pot_name]

replicate = REPLICATE[pot_name]

st.caption(

    f"光条件：{light}　｜　刺激条件：{stimulus}　｜　反復：{replicate}"

)

st.markdown("---")

# ==================================================
# 画像アップロード
# ==================================================

uploaded_file = st.file_uploader(

    "植物画像をアップロード",

    type=["jpg", "jpeg", "png"]

)

if uploaded_file is None:

    st.info("画像をアップロードしてください。")

    st.stop()

# ==================================================
# 画像解析
# ==================================================

with st.spinner("解析中..."):

    result = analyze_image(uploaded_file)

result_ratio = result["color_ratio"]

# ==================================================
# 解析画像
# ==================================================

st.header("解析結果")

col1, col2 = st.columns(2)

with col1:

    st.subheader("元画像")

    st.image(

        cv2.cvtColor(

            result["original"],

            cv2.COLOR_BGR2RGB

        ),

        use_container_width=True

    )

with col2:

    st.subheader("色分類画像")

    st.image(

        cv2.cvtColor(

            result["overlay"],

            cv2.COLOR_BGR2RGB

        ),

        use_container_width=True

    )

st.markdown("---")

# ==================================================
# 数値結果
# ==================================================

c1, c2, c3 = st.columns(3)

with c1:

    st.metric(

        "被覆率",

        f"{result['coverage']:.2f}%"

    )

with c2:

    st.metric(

        "総葉面積",

        f"{result['leaf_area']:.2f} cm²"

    )

with c3:

    st.metric(

        "最適クラスタ数",

        result["best_k"]

    )

st.markdown("---")
# ==================================================
# クラスタ情報
# ==================================================

st.header("葉色クラスタ")

user_mapping = {}

for cluster in result["clusters"]:

    st.markdown("---")

    left, right = st.columns([1, 3])

    # ----------------------------
    # 色見本
    # ----------------------------
    with left:

        b, g, r = cluster["bgr"]

        preview = np.zeros(

            (
                COLOR_PATCH_SIZE,
                COLOR_PATCH_SIZE,
                3
            ),

            dtype=np.uint8

        )

        preview[:] = (b, g, r)

        st.image(

            cv2.cvtColor(

                preview,

                cv2.COLOR_BGR2RGB

            ),

            caption=f"Cluster {cluster['id']}",

            use_container_width=True

        )

    # ----------------------------
    # 詳細情報
    # ----------------------------
    with right:

        st.subheader(f"Cluster {cluster['id']}")

        st.write(
            f"割合：{cluster['ratio']:.2f}%"
        )

        h, s, v = cluster["hsv"]

        l, a, b2 = cluster["lab"]

        st.write(
            f"RGB : ({r}, {g}, {b})"
        )

        st.write(
            f"HSV : ({h:.1f}, {s:.1f}, {v:.1f})"
        )

        st.write(
            f"Lab : ({l:.1f}, {a:.1f}, {b2:.1f})"
        )

        default_name = result["mapping"].get(

            cluster["id"],

            "Green"

        )

        default_index = COLOR_OPTIONS.index(
            default_name
        )

        color_name = st.selectbox(

            "葉色",

            COLOR_OPTIONS,

            index=default_index,

            key=f"cluster_{cluster['id']}"

        )

        user_mapping[
            cluster["id"]
        ] = color_name

st.markdown("---")

# ==================================================
# 色割合再計算
# ==================================================

result_ratio = {

    "Dark Green": 0.0,

    "Green": 0.0,

    "Light Green": 0.0,

    "Yellow": 0.0

}

for cluster in result["clusters"]:

    color_name = user_mapping[
        cluster["id"]
    ]

    result_ratio[color_name] += cluster[
        "ratio"
    ]

# ==================================================
# 最終解析結果
# ==================================================

st.header("最終解析結果")

ratio_df = pd.DataFrame(

    {

        "葉色": list(
            result_ratio.keys()
        ),

        "割合 (%)": [

            round(v, 2)

            for v in result_ratio.values()

        ]

    }

)

st.dataframe(

    ratio_df,

    use_container_width=True

)

st.markdown("---")

# ==================================================
# 円グラフ
# ==================================================

pie_fig = create_ratio_pie(
    result_ratio
)

st.pyplot(
    pie_fig,
    use_container_width=True
)

st.markdown("---")

# ==================================================
# 過去データ
# ==================================================

st.header("解析履歴")

history_df = load_excel()

if history_df is None or history_df.empty:

    st.info("保存されたデータはまだありません。")

else:

    st.write(history_df)

    st.markdown("---")

    st.subheader("被覆率推移")

    coverage_fig = create_coverage_graph(
        history_df
    )

    st.pyplot(
        coverage_fig,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("総葉面積推移")

    area_fig = create_leaf_area_graph(
        history_df
    )

    st.pyplot(
        area_fig,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("葉色割合推移")

    color_fig = create_color_ratio_graph(
        history_df
    )

    st.pyplot(
        color_fig,
        use_container_width=True
    )

st.markdown("---")

# ==================================================
# 保存
# ==================================================

st.header("保存")

col1, col2 = st.columns(2)

# -----------------------------
# Excel保存
# -----------------------------
with col1:

    if st.button("💾 Excelへ保存"):

        save_to_excel(

            measure_date,

            pot_name,

            light,

            stimulus,

            replicate,

            result_ratio,

            result["coverage"],

            result["leaf_area"]

        )

        st.success("Excelへ保存しました。")

# -----------------------------
# CSVダウンロード
# -----------------------------
with col2:

    csv = ratio_df.to_csv(

        index=False,

        encoding="utf-8-sig"

    ).encode("utf-8-sig")

    st.download_button(

        "📄 CSVダウンロード",

        data=csv,

        file_name=f"{pot_name}_{measure_date}.csv",

        mime="text/csv"

    )

st.markdown("---")

st.success("🌱 PlantAnalyzer の解析が完了しました。")


