import plotly.express as px
import nltk
import utils
import pandas as pd

text1 = utils.open_and_clean_text('text1.txt')
text2 = utils.open_and_clean_text('text2.txt')
control = utils.open_and_clean_text('control.txt')

sentences1 = nltk.sent_tokenize(text1)
sentences2 = nltk.sent_tokenize(text2)
sentences_control = nltk.sent_tokenize(control)

embeddings1 = utils.embed(sentences1)
embeddings2 = utils.embed(sentences2)
embeddings_control = utils.embed(sentences_control)





umap_reduced_embeddings1 = utils.umap_reduce(embeddings1, 4, 100)
umap_reduced_embeddings2 = utils.umap_reduce(embeddings2, 4, 100)
umap_reduced_embeddings_control = utils.umap_reduce(embeddings_control, 4, 100)




df_umap1 = utils.create_4d_dataframe(umap_reduced_embeddings1, sentences1, 1, 'text1')
df_umap2 = utils.create_4d_dataframe(umap_reduced_embeddings2, sentences2, 1, 'text2')
df_umap_control = utils.create_4d_dataframe(umap_reduced_embeddings_control, sentences_control, 1, 'control')


df_umap_combined = pd.concat([df_umap1, df_umap2, df_umap_control], ignore_index=True)




fig_umap = px.scatter_3d(
    df_umap_combined,
    x='dim1',
    y='dim2',
    z='dim3',
    color = 'dim4',
    symbol = 'source',
    hover_data=['sentences', 'source'],
    title='UMAP 3D Scatter Plot'
)


fig_umap.show()
