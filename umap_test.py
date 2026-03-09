import plotly.express as px
import nltk
import utils
import pandas as pd

text1 = utils.open_and_clean_text('text1.txt')

sentences1 = nltk.sent_tokenize(text1)

embeddings1 = utils.embed(sentences1)

umap_reduced_embeddings1 = utils.umap_reduce(embeddings1, 5, 5)
umap_reduced_embeddings2 = utils.umap_reduce(embeddings1, 5, 25)
umap_reduced_embeddings3 = utils.umap_reduce(embeddings1, 5, 50)
umap_reduced_embeddings4 = utils.umap_reduce(embeddings1, 5, 100)

df_umap1 = utils.create_5d_dataframe(umap_reduced_embeddings1, sentences1, 1, 'text1')
df_umap2 = utils.create_5d_dataframe(umap_reduced_embeddings2, sentences1, 1, 'text1')
df_umap3 = utils.create_5d_dataframe(umap_reduced_embeddings3, sentences1, 1, 'text1')
df_umap4 = utils.create_5d_dataframe(umap_reduced_embeddings4, sentences1, 1, 'text1')

fig_umap = px.scatter_3d(
    df_umap1,
    x='dim1',
    y='dim2',
    z='dim3',
    color = 'dim4',
    size = 'dim5',
    symbol = 'source',
    hover_data=['sentences', 'source'],
    title='UMAP 3D Scatter Plot 5 neighbors'
) 


#generate the same plot for df_umap2 through 4

fig_umap2 = px.scatter_3d(
    df_umap2,
    x='dim1',
    y='dim2',
    z='dim3',
    color = 'dim4',
    size = 'dim5',
    symbol = 'source',
    hover_data=['sentences', 'source'],
    title='UMAP 3D Scatter Plot 25 neighbors'
)

fig_umap3 = px.scatter_3d(
    df_umap3,
    x='dim1',
    y='dim2',
    z='dim3',
    color = 'dim4',
    size = 'dim5',
    symbol = 'source',
    hover_data=['sentences', 'source'],
    title='UMAP 3D Scatter Plot 50 neighbors'
)

fig_umap4 = px.scatter_3d(
    df_umap4,
    x='dim1',
    y='dim2',
    z='dim3',
    color = 'dim4',
    size = 'dim5',
    symbol = 'source',
    hover_data=['sentences', 'source'],
    title='UMAP 3D Scatter Plot 100 neighbors'
)

fig_umap.show()
fig_umap2.show()
fig_umap3.show()
fig_umap4.show()