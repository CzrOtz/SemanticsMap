import plotly.express as px
import nltk
import utils
import pandas as pd
import numpy as np
import plotly.graph_objects as go




def pacMap_dataframe(text_passages: list, multiplier: int, sources: list, pacmap_settings_int_dict: dict, pacmap_settings_float_dict: dict, pacmap_settings_bool_dict: dict, pacmap_settings_string_dict: dict) -> pd.DataFrame:
    embeddings_list = []
    all_sentences = []
    all_sources = []

    for i, source in zip(text_passages, sources):
        text = utils.clean_text(i)
        sentence = nltk.sent_tokenize(text)
        embeddings = utils.embed(sentence)
        embeddings_list.append(embeddings)
        all_sentences.extend(sentence)
        all_sources.extend([source] * len(sentence))

    all_embeddings = np.vstack(embeddings_list)
    pacmap_reduced_embeddings = utils.pacmap_reduce_fully_tunable(all_embeddings, pacmap_settings_int_dict, pacmap_settings_float_dict, pacmap_settings_bool_dict, pacmap_settings_string_dict)

    if pacmap_settings_int_dict['n_components'] == 3:
        df_pacmap_combined = utils.create_3d_dataframe(pacmap_reduced_embeddings, all_sentences, multiplier, all_sources)
    elif pacmap_settings_int_dict['n_components'] == 4:   
        df_pacmap_combined = utils.create_4d_dataframe(pacmap_reduced_embeddings, all_sentences, multiplier, all_sources)

    return df_pacmap_combined


def scatter_plot(data_frame: pd.DataFrame, x_cordinate: str, y_cordinate: str, z_cordinate: str, dimensions: int, color_scale: str) -> px.scatter_3d:
    if dimensions == 4: color_cordinate = "dim4"
    if dimensions == 3: color_cordinate = "source"

    fig_pacmap = px.scatter_3d(
        data_frame,
        x=x_cordinate,
        y=y_cordinate,
        z=z_cordinate,
        color = color_cordinate,
        color_continuous_scale=color_scale,
        symbol = 'source',
        hover_data=['sentences', 'source'],
        title='PaCMAP 3D Scatter Plot',
        height=800,
        width=1200
    )

    fig_pacmap.update_layout(
        paper_bgcolor="#161b22",
        plot_bgcolor="#161b22"
        )

    if dimensions == 4:
        fig_pacmap.update_layout(
            legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top"),
            coloraxis_colorbar=dict(x=1.08, y=0.5, len=0.8),
            margin=dict(r=80)
        )
        fig_pacmap.update_traces(
            marker=dict(
                colorbar=dict(
                    x=1.15,
                    y=0.5,
                    len=0.75,
                    thickness=18
                )
            )
        )   


    return fig_pacmap

def line_plot(data_frame: pd.DataFrame, x_cordinate: str, y_cordinate: str, z_cordinate: str, dimensions: int, color_scale: str) -> px.scatter_3d:
    if dimensions == 4: color_cordinate = "dim4"
    if dimensions == 3: color_cordinate = "source"

    fig_pacmap = px.line_3d(
        data_frame,
        x=x_cordinate,
        y=y_cordinate,
        z=z_cordinate,
        color = color_cordinate,
        symbol = 'source',
        hover_data=['sentences', 'source'],
        title='PaCMAP 3D Line Plot',
        height=800,
        width=1200
    )

    fig_pacmap.update_layout(
        paper_bgcolor="#161b22",
        plot_bgcolor="#161b22"
        )

    if dimensions == 4:
        fig_pacmap.update_layout(
            legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top"),
            coloraxis_colorbar=dict(x=1.08, y=0.5, len=0.8),
            margin=dict(r=80)
        )
        fig_pacmap.update_traces(
            marker=dict(
                colorbar=dict(
                    x=1.15,
                    y=0.5,
                    len=0.75,
                    thickness=18
                )
            )
        )   


    return fig_pacmap

def scatter_matrix(data_frame: pd.DataFrame, dimensions: int, color_scale: str) -> px.scatter_matrix:
    if dimensions == 4: color_cordinate = "dim4"
    if dimensions == 3: color_cordinate = "source"

    fig_matrix = px.scatter_matrix(
    data_frame,
    dimensions=['dim1', 'dim2', 'dim3'],
    color=color_cordinate,
    color_continuous_scale=color_scale, 
    symbol='source',
    height=800,
    width=1200,
    )

    fig_matrix.update_layout(
        paper_bgcolor="#161b22",
        plot_bgcolor="#161b22"
    )

    if dimensions == 4:
        fig_matrix.update_layout(
            legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top"),
            coloraxis_colorbar=dict(x=1.08, y=0.5, len=0.8),
            margin=dict(r=80)
        )

        fig_matrix.update_layout(
            paper_bgcolor="#161b22",
            plot_bgcolor="#161b22",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.05,
                xanchor="left",
                x=0
            )
        )
    
    return fig_matrix
    


def test_cone():
    fig = go.Figure(data=go.Cone(
        x=[0, 1, 2],
        y=[0, 1, 2],
        z=[0, 1, 2],
        u=[1, 1, 1],
        v=[0, 1, 0],
        w=[0, 0, 1],
        sizemode="scaled",
        sizeref=0.5,
        showscale=True
    ))
    return fig



