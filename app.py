# app.py
# PlantAnalyzer Ver3.0

import os
from datetime import date

import streamlit as st
import pandas as pd

from config import (
    APP_TITLE,
    POT_OPTIONS,
    CONDITION_OPTIONS,
    COLOR_NAMES,
    EXCEL_FILE,
)

from image_analysis import analyze_image

from excel_manager import (
    load_data,
    create_record,
    add_record,
)

from graph_manager import (
    leaf_area_graph,
    coverage_graph,
    condition_coverage_graph,
    color_ratio_graph,
    color_stacked_graph,
    condition_leaf_area_graph,
    best_k_graph,
)


# =========================================================
# ページ設定
# =========================================================

st.set_page_config(
    page_title=APP_TITLE,
    layout="wide"
)

st.title("🌱 PlantAnalyzer")


# =========================================================
# セッション状態
# =========================================================

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = None

if "color_mapping" not in st.session_state:
    st.session_state.color_mapping = {}

if "save_message" not in st.session_state:
    st.session_state.save_message = ""


# =========================================================
# サイドバー
# =========================================================

st.sidebar.header("Experiment")

selected_date = st.sidebar.date_input(
    "Date",
    value=date.today()
)

selected_pot = st.sidebar.selectbox(
    "Pot",
    POT_OPTIONS
)

selected_condition = st.sidebar.selectbox(
    "Condition",
    CONDITION_OPTIONS
)


# =========================================================
# 画像アップロード
# =========================================================

st.header("1. Image Analysis")

uploaded_file = st.file_uploader(
    "Upload plant image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# =========================================================
# 解析
# =========================================================

if uploaded_file is not None:

    # 新しい画像なら解析結果をリセット
    if (
        st.session_state.uploaded_name
        != uploaded_file.name
    ):

        st.session_state.analysis_result = None
        st.session_state.color_mapping = {}
        st.session_state.uploaded_name = (
            uploaded_file.name
        )

    if st.button(
        "🔬 Analyze Image",
        type="primary"
    ):

        try:

            with st.spinner(
                "解析中..."
            ):

                result = analyze_image(
                    uploaded_file
                )

            st.session_state.analysis_result = (
                result
            )

            # 自動分類結果を初期値にする
            st.session_state.color_mapping = (
                result[
                    "color_result"
                ][
                    "mapping"
                ].copy()
            )

            st.success(
                "解析が完了しました。"
            )

        except Exception as e:

            st.error(
                "画像解析中にエラーが発生しました。"
            )

            st.exception(e)


# =========================================================
# 解析結果
# =========================================================

result = st.session_state.analysis_result


if result is not None:

    st.header("2. Analysis Result")

    # -----------------------------------------------------
    # 基本値
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Leaf Area",
            f'{result["leaf_area"]:.2f} cm²'
        )

    with col2:

        st.metric(
            "Coverage",
            f'{result["coverage"]:.2f} %'
        )

    with col3:

        best_k = result[
            "color_result"
        ][
            "best_k"
        ]

        st.metric(
            "Best K",
            best_k
        )


    # -----------------------------------------------------
    # 画像表示
    # -----------------------------------------------------

    st.subheader(
        "Detected Image"
    )

    image_col1, image_col2 = (
        st.columns(2)
    )

    with image_col1:

        st.image(
            result["image"],
            channels="BGR",
            caption="Original Image"
        )

    with image_col2:

        st.image(
            result["color_result"][
                "overlay"
            ],
            channels="BGR",
            caption="Color Analysis"
        )


    # =====================================================
    # 色見本
    # =====================================================

    st.header(
        "3. Color Samples"
    )

    st.write(
        "クラスタ中心の実際の色を確認し、"
        "Dark Green / Green / Light Green / Yellow "
        "のどれに相当するかを判断してください。"
    )

    color_result = result[
        "color_result"
    ]

    clusters = color_result[
        "clusters"
    ]

    if len(clusters) == 0:

        st.warning(
            "色クラスタが検出されませんでした。"
        )

    else:

        # クラスタごとに表示
        for cluster in clusters:

            cluster_id = cluster[
                "id"
            ]

            rgb = cluster[
                "rgb"
            ]

            ratio = cluster[
                "ratio"
            ]

            auto_name = (
                st.session_state
                .color_mapping
                .get(
                    cluster_id,
                    "Green"
                )
            )

            r, g, b = rgb

            st.markdown(
                f"""
                <div style="
                    display:flex;
                    align-items:center;
                    gap:15px;
                    margin-bottom:10px;
                ">
                    <div style="
                        width:60px;
                        height:60px;
                        background-color:rgb({r},{g},{b});
                        border:1px solid #555;
                        border-radius:8px;
                    "></div>
                    <div>
                        <b>Cluster {cluster_id}</b><br>
                        RGB: ({r}, {g}, {b})<br>
                        Ratio: {ratio:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            selected_color = st.selectbox(
                f"Cluster {cluster_id} の色名",
                COLOR_NAMES,
                index=(
                    COLOR_NAMES.index(
                        auto_name
                    )
                    if auto_name
                    in COLOR_NAMES
                    else 1
                ),
                key=f"color_select_{cluster_id}"
            )

            st.session_state.color_mapping[
                cluster_id
            ] = selected_color


    # =====================================================
    # 色割合再計算
    # =====================================================

    st.subheader(
        "Color Composition"
    )

    labels = None

    # color_analysis.pyでは
    # labelsそのものを保存していないため、
    # 現在のratioをベースに表示する。
    original_ratio = color_result[
        "ratio"
    ]

    corrected_ratio = {
        color: 0.0
        for color in COLOR_NAMES
    }

    for cluster in clusters:

        cluster_id = cluster[
            "id"
        ]

        cluster_ratio = cluster[
            "ratio"
        ]

        color_name = (
            st.session_state
            .color_mapping
            .get(
                cluster_id,
                "Green"
            )
        )

        corrected_ratio[
            color_name
        ] += cluster_ratio


    ratio_df = pd.DataFrame({
        "Color": COLOR_NAMES,
        "Percentage": [
            corrected_ratio[color]
            for color in COLOR_NAMES
        ]
    })

    st.dataframe(
        ratio_df,
        use_container_width=True,
        hide_index=True
    )


    # -----------------------------------------------------
    # 円グラフ
    # -----------------------------------------------------

    st.bar_chart(
        ratio_df.set_index(
            "Color"
        )
    )


    # =====================================================
    # Excel保存
    # =====================================================

    st.header(
        "4. Save Result"
    )

    if st.button(
        "💾 Save to Excel"
    ):

        try:

            record = create_record(

                date=str(
                    selected_date
                ),

                pot=selected_pot,

                condition=selected_condition,

                leaf_area=result[
                    "leaf_area"
                ],

                coverage=result[
                    "coverage"
                ],

                best_k=result[
                    "color_result"
                ][
                    "best_k"
                ],

                color_ratio=corrected_ratio
            )

            add_record(
                record
            )

            st.success(
                "Excelに保存しました。"
            )

        except Exception as e:

            st.error(
                "Excel保存中にエラーが発生しました。"
            )

            st.exception(e)


# =========================================================
# 保存データ
# =========================================================

st.header(
    "5. Recorded Data"
)

df = load_data()

if df.empty:

    st.info(
        "まだ保存されたデータはありません。"
    )

else:

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


    # =====================================================
    # グラフ
    # =====================================================

    st.header(
        "6. Graphs"
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Leaf Area",
            "Coverage",
            "Color",
            "Best K"
        ]
    )


    # -----------------------------------------------------
    # 葉面積
    # -----------------------------------------------------

    with tab1:

        fig = leaf_area_graph(
            df
        )

        if fig is not None:

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        fig2 = (
            condition_leaf_area_graph(
                df
            )
        )

        if fig2 is not None:

            st.plotly_chart(
                fig2,
                use_container_width=True
            )


    # -----------------------------------------------------
    # 被覆率
    # -----------------------------------------------------

    with tab2:

        fig = coverage_graph(
            df
        )

        if fig is not None:

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        fig2 = (
            condition_coverage_graph(
                df
            )
        )

        if fig2 is not None:

            st.plotly_chart(
                fig2,
                use_container_width=True
            )


    # -----------------------------------------------------
    # 色
    # -----------------------------------------------------

    with tab3:

        selected_graph_pot = st.selectbox(
            "Pot",
            ["All"] + POT_OPTIONS,
            key="graph_pot"
        )

        if selected_graph_pot == "All":

            graph_pot = None

        else:

            graph_pot = (
                selected_graph_pot
            )

        fig = color_ratio_graph(
            df,
            graph_pot
        )

        if fig is not None:

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        fig2 = color_stacked_graph(
            df,
            graph_pot
        )

        if fig2 is not None:

            st.plotly_chart(
                fig2,
                use_container_width=True
            )


    # -----------------------------------------------------
    # K
    # -----------------------------------------------------

    with tab4:

        fig = best_k_graph(
            df
        )

        if fig is not None:

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# =========================================================
# ダウンロード
# =========================================================

if os.path.exists(
    EXCEL_FILE
):

    st.header(
        "7. Download"
    )

    with open(
        EXCEL_FILE,
        "rb"
    ) as f:

        st.download_button(
            "📥 Download Excel",
            data=f,
            file_name=EXCEL_FILE,
            mime=(
                "application/"
                "vnd.openxmlformats-officedocument"
                ".spreadsheetml.sheet"
            )
  )
