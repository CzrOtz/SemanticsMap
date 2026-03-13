# python -m streamlit run streamlit_test_2.py
import streamlit as st
import Dynamic_comparisons_2 as dc

st.set_page_config(layout="wide")

st.title("Semantic Analysis Scratchpad")

st.markdown(
    """This window uses hugging face 384 dimensional embeddings, nklt sentence tokenization, and PaCMAP dimensionality reduction to create a 3D scatter plot of the semantic relationships between sentences in different texts. You can paste your own texts in the boxes below, and adjust the settings in the sidebar to see how it changes the visualization. The goal is to help you visually compare the logic and themes of different texts in a more intuitive way."""
)

with st.expander("About this tool"):
    st.write(
        """example"""
    )

#SIDE BAR AREA

st.sidebar.title("controls")

neighbor_count = st.sidebar.number_input("Neighbor count", min_value=2, value=15)
dimension_count = st.sidebar.selectbox("Dimension count/n components", [3, 4], index=0)

if dimension_count == 4:
    color_scale = st.sidebar.selectbox(
        "Color scale",
        ['Viridis', 'Cividis', 'Plasma', 'Inferno', 'Magma', 'Turbo'],
        index=0
    )
else:
    color_scale = None




with st.sidebar.expander("Advanced Settings"):
    MN_ratio = st.number_input("MN_ratio", value=0.5)
    FP_ratio = st.number_input("FP_ratio", value=2.0)
    lr = st.number_input("Learning Rate", value=1.0)
    num_iters = st.number_input("Iterations", value=1000)
    distance = st.selectbox("Distance Metric", ["angular", "euclidean", "manhattan"], index=0)
    random_state = st.number_input("Random State", value=42)
    init = st.selectbox("Initialization Method", ["pca", "random"])
    apply_pca = st.checkbox("Apply PCA", value=True)
    multiplier = st.number_input("Scale multiplier", min_value=1, value=1)
    

pacMap_settings_int_dict = {
    'n_components': dimension_count,
    'n_neighbors': neighbor_count,
    'num_iters': num_iters,
    'random_state': random_state
}

pacMap_settings_float_dict = {
    'MN_ratio': MN_ratio,
    'FP_ratio': FP_ratio,
    'lr': lr
}

pacMap_settings_bool_dict = {
    'apply_pca': apply_pca
}

pacMap_settings_string_dict = {
    'distance': distance,
    'init': init
}

st.divider()

if 'num_texts' not in st.session_state:
    st.session_state.num_texts = 2

col_inputs = st.columns(st.session_state.num_texts)
text_data = []
labels = []

for i in range(st.session_state.num_texts):
    with col_inputs[i]:
        label = st.text_input(f"Label {i+1}", value=f"Doc {i+1}", key=f"label_{i}")
        content = st.text_area(f"Paste Text {i+1}", height=300, key=f"text_{i}")
        if content:
            text_data.append(content)
            labels.append(label)

col1, col2, col3 = st.columns([3, 2, 1])


st.divider()

add_text_col, remove_text_col = st.columns(2)

with add_text_col:
    if st.button("Add Another Text Box"):
        st.session_state.num_texts += 1
        st.rerun()

with remove_text_col:
    if st.button("Remove Text Box"):
        if st.session_state.num_texts > 1:
            st.session_state.num_texts -= 1
            st.rerun()

st.divider()

if st.button("Process Texts"):
    with st.spinner("Processing texts..."):
        pacMap_df = dc.pacMap_dataframe(text_data, multiplier, labels, pacMap_settings_int_dict, pacMap_settings_float_dict, pacMap_settings_bool_dict, pacMap_settings_string_dict)
        scatter_fig = dc.scatter_plot(pacMap_df, 'dim1', 'dim2', 'dim3', dimension_count, color_scale)
        matrix_fig = dc.scatter_matrix(pacMap_df, dimension_count, color_scale)

    st.plotly_chart(scatter_fig, width="stretch")
    st.plotly_chart(matrix_fig, width="stretch")



