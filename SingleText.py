# from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px
import re
import nltk
import umap

import plotly.graph_objects as go
from cleantext import clean
import utils

print("Starting the process...")

cleaned_text = utils.open_and_clean_text('MyText.txt')


sentences = nltk.sent_tokenize(cleaned_text)

print(sentences)

embeddings = utils.embed(sentences)

print("Embeddings generated.")

pca_reduced_embeddings = utils.pca_reduce(embeddings, dimensions=5)


umap_reduced_embeddings = utils.umap_reduce(embeddings, dimensions=5)


df_pca = utils.create_5d_dataframe(pca_reduced_embeddings, sentences, multiplier=100)

df_umap = utils.create_5d_dataframe(umap_reduced_embeddings, sentences, multiplier=1)


#part 5 - create a 3D scatter plot using plotly to visualize the sentences in 3D space
fig_pca = px.scatter_3d(
    df_pca, 
    x='dim1',
    y='dim2',
    z='dim3',
    color='dim4',
    size='dim5',
    hover_data=['sentences'],
    title=f'PCA 3D Scatter Plot'
    )

fig_umap = px.scatter_3d(
    df_umap, 
    x='dim1',
    y='dim2',
    z='dim3',
    color='dim4',
    size='dim5',
    hover_data=['sentences'],
    title='UMAP 3D Scatter Plot'
    )



fig_pca.show()
fig_umap.show()


print("3D scatter plot created.")



