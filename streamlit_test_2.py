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

with st.sidebar.expander("Reduction Algorithm"):
    reduction_algorithm = st.selectbox(
        "Select dimensionality reduction algorithm",
        ['PaCMAP', 'UMAP', 'PCA'],
        index=0
    )

st.sidebar.title("controls")


# dimension_count = st.sidebar.selectbox("Dimension count/n components", [3, 4, 5, 6], index=0)





def pacMap_settings_func() -> dict: 
    with st.sidebar.expander("pacmac Advanced Settings"):
        neighbor_count = st.number_input("Neighbor count", min_value=2, value=15)
        dimension_count = st.selectbox("Dimension count/n components", [3, 4, 5, 6], index=0)
        MN_ratio = st.number_input("MN_ratio", value=0.5)
        FP_ratio = st.number_input("FP_ratio", value=2.0)
        lr = st.number_input("Learning Rate", value=1.0)
        num_iters = st.number_input("Iterations", value=1000)
        distance = st.selectbox("Distance Metric", ["angular", "euclidean", "manhattan"], index=0)
        random_state = st.number_input("Random State", value=42)
        init = st.selectbox("Initialization Method", ["pca", "random"])
        apply_pca = st.checkbox("Apply PCA", value=True)
        multiplier = st.number_input("Scale multiplier", min_value=1, value=1)

        return {
            "n_neighbors": neighbor_count,
            "n_components": dimension_count,
            "num_iters": num_iters,
            "random_state": random_state,
            "MN_ratio": MN_ratio,
            "FP_ratio": FP_ratio,
            "lr": lr,
            "apply_pca": apply_pca,
            "distance": distance,
            "init": init,
            "multiplier": multiplier
        }
    
def umap_settings_func() -> dict:
    with st.sidebar.expander("UMAP Advanced Settings"):
        neighbor_count = st.number_input("Neighbor count", min_value=2, value=15)
        dimension_count = st.selectbox("Dimension count/n components", [3, 4, 5, 6], index=0)
        distance = st.selectbox("Distance Metric", ["euclidean", "manhattan", "cosine", "correlation"], index=0)
        random_state = st.number_input("Random State", value=42)
        min_dist = st.number_input("Min Dist", value=0.1)
        multiplier = st.number_input("Scale multiplier", min_value=1, value=1)

        return {
            "n_neighbors": neighbor_count,
            "n_components": dimension_count,
            "random_state": random_state,
            "distance": distance,
            "min_dist": min_dist,
            "multiplier": multiplier
        }
    
def pca_settings_func() -> dict:
    with st.sidebar.expander("PCA Advanced Settings"):
        dimension_count = st.selectbox("Dimension count/n components", [3, 4, 5, 6], index=0)
        svd_solver = st.selectbox("SVD Solver", ["auto", "full", "arpack", "randomized"], index=0)

        if svd_solver == "randomized":
            random_state = st.number_input("Random State", value=42)
        else:
            random_state = None

        whiten = st.checkbox("Whiten", value=False)
        multiplier = st.number_input("Scale multiplier", min_value=1, value=1)

        return {
            "n_components": dimension_count,
            "svd_solver": svd_solver,
            "whiten": whiten,
            "random_state": random_state,
            "multiplier": multiplier
        }


if reduction_algorithm == 'PaCMAP':
    reducer_settings = pacMap_settings_func()
if reduction_algorithm == 'UMAP':
    reducer_settings = umap_settings_func()
if reduction_algorithm == 'PCA':
    reducer_settings = pca_settings_func()

print("reducer:", repr(reducer_settings))

st.write(f"Current reduction algorithm: {reduction_algorithm}")

#conditionals for the plot

#color scale for 4d and 5d concerning scatter, matrix, and cone plot
#3d does not have a color scale option

if reducer_settings['n_components'] >= 4:
    color_scale = st.sidebar.selectbox(
        "Color scale",
        ['Viridis', 'Cividis', 'Plasma', 'Inferno', 'Magma', 'Turbo'],
        index=0
    )
else:
    color_scale = None

#this only concerns only cone plot
if reducer_settings['n_components']  == 6:
    size_mode = st.sidebar.selectbox(
        "Size mode",
        ['scaled', 'absolute'],
        index=0
    )

    size_ref = st.sidebar.number_input("Size reference", value=20.0)


###### TEXT CAPTURE AREA START

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

#TEXT CAPTURE AREA END

#text boxes start #######################

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

# text boxes end #######################

def process_and_plot(reduction_algorithm):
    if st.button("Process Texts"):
        with st.spinner("Processing texts..."):

            print("THIS IS THE REDICTION ALGORITHM PASSED TO THE FUNCTION:", reduction_algorithm)
            data_frame = dc.produce_dataframe(text_data, labels, reduction_algorithm, reducer_settings)

            if reducer_settings['n_components']  == 6:
                st.warning("scatter plot, matrix, and line plot not available for 6 dimensions")
                cone_plot = dc.cone_plot(data_frame, color_scale, size_mode, size_ref)
                st.plotly_chart(cone_plot, width="stretch")
            
            if reducer_settings['n_components'] == 5:
                st.warning("line plot and matrix not available for 5 dimensions")
                st.warning("cone plot not available for dimensions 5 through 3")
                st.warning("matrix plot not available for 5 dimensions and lower")

                scatter_fig = dc.scatter_plot(data_frame, 'dim1', 'dim2', 'dim3', reducer_settings['n_components'], color_scale)

                st.plotly_chart(scatter_fig, width="stretch")

            if reducer_settings['n_components'] == 4:
                st.warning("cone plot not available for dimensions 5 through 3")
                st.warning("line plot not available for 4 dimensions")
                
                scatter_fig = dc.scatter_plot(data_frame, 'dim1', 'dim2', 'dim3', reducer_settings['n_components'], color_scale)
                matrix_fig = dc.scatter_matrix(data_frame, reducer_settings['n_components'], color_scale)

                st.plotly_chart(scatter_fig, width="stretch")
                st.plotly_chart(matrix_fig, width="stretch")
            
            if reducer_settings['n_components'] == 3:
                st.warning("cone plot not available for dimensions 5 through 3")

                scatter_fig = dc.scatter_plot(data_frame, 'dim1', 'dim2', 'dim3', reducer_settings['n_components'], color_scale)
                matrix_fig = dc.scatter_matrix(data_frame, reducer_settings['n_components'], color_scale)
                line_fig = dc.line_plot(data_frame, 'dim1', 'dim2', 'dim3', reducer_settings['n_components'])

                st.plotly_chart(scatter_fig, width="stretch")
                st.plotly_chart(matrix_fig, width="stretch")
                st.plotly_chart(line_fig, width="stretch")





    

process_and_plot(reduction_algorithm)
  





