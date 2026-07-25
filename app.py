"""
app.py
Plant Analyzer
Ver 1.0
"""

import streamlit as st
import pandas as pd
import cv2

from config import (
    APP_NAME,
    POT_NAMES,
    SOIL_PIXELS,
    TREATMENTS
)

from file_manager import (
    create_folders,
    save_original_image,
    save_mask_image,
    save_overlay_image,
    save_csv,
    load_csv
)

from image_analysis import (
    load_image,
    analyze_colors,
    coverage,
    ratios
)

# ==========================
# 初期設定
# ==========================

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🌱",
    layout="wide"
)

create_folders()

st.title("🌱 Plant Analyzer")

st.write("植物画像解析システム")

st.divider()

# ==========================
# サイドバー
# ==========================

st.sidebar.header("設定")

pot = st.sidebar.selectbox(

    "ポット",

    POT_NAMES

)

treatment = TREATMENTS[pot]

st.sidebar.write(f"処理区：{treatment}")

soil_pixels = SOIL_PIXELS[pot]

if soil_pixels is None:

    st.sidebar.warning(
        "土面積が未登録です"
    )

else:

    st.sidebar.success(
        f"土面積：{soil_pixels} Pixel"
    )

# ==========================
# 画像アップロード
# ==========================

uploaded = st.file_uploader(

    "植物写真をアップロード",

    type=[
        "jpg",
        "jpeg",
        "png"
    ]

)

if uploaded is None:

    st.info("画像をアップロードしてください。")

    st.stop()

# ==========================
# 読み込み
# ==========================

image = load_image(uploaded)

rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

st.subheader("元画像")

st.image(
    rgb,
    use_container_width=True
)

st.divider()

# ==========================
# 解析開始
# ==========================

run = st.button(

    "解析開始",
    use_container_width=True

)

if not run:

    st.stop()
  # ==========================
# 画像解析
# ==========================

with st.spinner("画像を解析中..."):

    color_pixels, leaf_pixels, mask, overlay = analyze_colors(image)

    color_ratios = ratios(
        color_pixels,
        leaf_pixels
    )

    cover = coverage(
        leaf_pixels,
        soil_pixels
    )

st.success("解析完了")

st.divider()

# ==========================
# 数値表示
# ==========================

col1, col2 = st.columns(2)

with col1:

    st.subheader("解析結果")

    if cover is None:

        st.warning(
            "土面積が登録されていないため被覆率は計算できません。"
        )

    else:

        st.metric(
            "被覆率",
            f"{cover:.2f}%"
        )

    st.metric(
        "葉面積",
        f"{leaf_pixels:,} Pixel"
    )

with col2:

    st.subheader("色別Pixel数")

    for name, value in color_pixels.items():

        st.write(
            f"{name} : {value:,} Pixel"
        )

st.divider()

# ==========================
# 色割合
# ==========================
st.subheader("色別割合")

ratio_df = pd.DataFrame({

    "Color": list(color_ratios.keys()),

    "Ratio (%)": list(color_ratios.values())

})

st.dataframe(
    ratio_df,
    use_container_width=True,
    hide_index=True
)

st.divider()

# ==========================
# マスク・Overlay画像
# ==========================

mask_rgb = cv2.cvtColor(
    mask,
    cv2.COLOR_GRAY2RGB
)

overlay_rgb = cv2.cvtColor(
    overlay,
    cv2.COLOR_BGR2RGB
)

left, right = st.columns(2)

with left:

    st.subheader("マスク画像")

    st.image(
        mask_rgb,
        use_container_width=True
    )

with right:

    st.subheader("Overlay画像")

    st.image(
        overlay_rgb,
        use_container_width=True
    )

st.divider()

# ==========================
# 保存
# ==========================

save_original_image(
    pot,
    image
)

save_mask_image(
    pot,
    mask
)

save_overlay_image(
    pot,
    overlay
)

save_csv(

    pot_name=pot,

    treatment=treatment,

    coverage=cover,

    leaf_pixels=leaf_pixels,

    pixels=color_pixels,

    ratios=color_ratios

)

st.success("データを保存しました。")

st.divider()

# ==========================
# 過去データ表示
# ==========================

st.subheader("解析履歴")

history = load_csv(pot)

if len(history) == 0:

    st.info("まだデータがありません。")

else:

    st.dataframe(

        history,

        use_container_width=True,

        hide_index=True

    )

st.divider()

# ==========================
# 被覆率グラフ
# ==========================

if len(history) > 0:

    if "Coverage" in history.columns:

        graph = history.copy()

        graph["DateTime"] = graph["Date"] + " " + graph["Time"]

        graph = graph.set_index("DateTime")

        st.subheader("被覆率推移")

        st.line_chart(

            graph["Coverage"]

        )

st.divider()

# ==========================
# 色別Pixelグラフ
# ==========================

pixel_columns = [

    "DarkGreenPixel",

    "GreenPixel",

    "LightGreenPixel",

    "YellowPixel",

    "BrownPixel"

]

exist = [

    c for c in pixel_columns

    if c in history.columns

]

if len(exist) > 0:

    st.subheader("色別Pixel推移")

    pixel_df = history.copy()

    pixel_df["DateTime"] = (

        pixel_df["Date"]

        + " "

        + pixel_df["Time"]

    )

    pixel_df = pixel_df.set_index(

        "DateTime"

    )

    st.line_chart(

        pixel_df[exist]

    )

st.divider()

# ==========================
# 色割合グラフ
# ==========================

ratio_columns = [

    "DarkGreenRatio",

    "GreenRatio",

    "LightGreenRatio",

    "YellowRatio",

    "BrownRatio"

]

exist = [

    c for c in ratio_columns

    if c in history.columns

]

if len(exist) > 0:

    st.subheader("色別割合推移")

    ratio_df = history.copy()

    ratio_df["DateTime"] = (

        ratio_df["Date"]

        + " "

        + ratio_df["Time"]

    )

    ratio_df = ratio_df.set_index(

        "DateTime"

    )

    st.line_chart(

        ratio_df[exist]

    )

st.divider()

st.success("Plant Analyzer Ver1.0")
