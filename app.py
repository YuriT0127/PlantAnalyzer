"""
app.py
PlantAnalyzer Ver2.0
"""

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from image_analysis import analyze_image

st.set_page_config(
    page_title="PlantAnalyzer",
    layout="wide"
)

st.title("🌱 PlantAnalyzer Ver2.0")

uploaded_file = st.file_uploader(
    "画像をアップロード",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    result = analyze_image(
        uploaded_file
    )

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

    st.divider()

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "被覆率",
            f'{result["coverage"]:.2f}%'
        )

    with c2:

        st.metric(
            "総葉面積",
            f'{result["leaf_area"]:.2f} cm²'
        )

    st.divider()

    st.subheader("クラスタ情報")

    from config import (
    COLOR_OPTIONS,
    COLOR_PATCH_SIZE
    )

    user_mapping = {}

    for cluster in result["clusters"]:

        st.markdown("---")

        c1, c2 = st.columns([1, 3])

        with c1:

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
                caption=f"Cluster {cluster['id']}"
            )

        with c2:

            st.write(
                f"RGB : ({r}, {g}, {b})"
            )

            h, s, v = cluster["hsv"]

            st.write(
                f"HSV : ({h}, {s}, {v})"
            )

            l, a, b2 = cluster["lab"]

            st.write(
                f"Lab : ({l:.1f}, {a:.1f}, {b2:.1f})"
            )
            st.write(
    f"割合 : {cluster['ratio']:.2f}%"
            )

            default_name = result[
                "mapping"
            ].get(
                cluster["id"],
                "Green"
            )

            color_name = st.selectbox(

                "色名",

                COLOR_OPTIONS,

                index=color_options.index(
                    default_name
                ),

                key=f"cluster_{cluster['id']}"

            )

            user_mapping[
                cluster["id"]
            ] = color_name

          st.divider()

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

    st.subheader("解析結果")

    ratio_df = pd.DataFrame({

        "色": list(
            result_ratio.keys()
        ),

        "割合(%)": [

            round(v, 2)

            for v in result_ratio.values()

        ]

    })

    st.dataframe(
        ratio_df,
        use_container_width=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("色割合")

        st.bar_chart(
            ratio_df.set_index("色")
        )

    with col2:

        csv = ratio_df.to_csv(
            index=False
        ).encode("utf-8-sig")

        st.download_button(

            "CSVダウンロード",

            data=csv,

            file_name="PlantAnalyzer_Result.csv",

            mime="text/csv"

        )

    st.divider()

    st.success("解析が完了しました。")

else:

    st.info(
        "画像をアップロードしてください。"
    )

