from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px
import re
import nltk
import umap
from plotly.subplots import make_subplots
import plotly.graph_objects as go
print("Starting the process...")
#part 1 - read the text file and split it into sentences, then store the sentences in a list
text_list = []
things_to_remove = ['\u2006', '\u2009', '\u202f', '\xa0', '\ufeff', '\n', '\r', '\t']

with open('MyText.txt', 'r', encoding='utf-8') as file:
    text = file.read()

for chars in things_to_remove:
    text = text.replace(chars, ' ')

sentences = nltk.sent_tokenize(text)

for sentence in sentences:
    sentence = re.sub(r'^\s*\d+', '', sentence)
    sentence = re.sub(r'(?<=\s)\d+(?=[A-Za-z“"\'])', '', sentence)
    sentence = re.sub(r'\d+(?=[A-Za-z])', '', sentence)
    sentence = sentence.strip()
    text_list.append(sentence)


print("done reading and cleaning text. Number of sentences:", len(text_list))

#part 2 - use the sentence transformer model to convert the sentences into embeddings, then store the embeddings in a list
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(text_list)

print("Embeddings generated.")

#part 3 - use PCA to reduce the dimensionality of the embeddings to 3 dimensions, then store the reduced embeddings in a list
pca = PCA(n_components=5)
pca_reduced_embeddings = pca.fit_transform(embeddings)
pca_total_variance = np.sum(pca.explained_variance_ratio_)


umap_reduce = umap.UMAP(n_components=5)
umap_reduced_embeddings = umap_reduce.fit_transform(embeddings)


print("Reduced embeddings generated.")


#part 4 - create a dataframe to store the sentences as x y z cordinates
df_pca = pd.DataFrame(pca_reduced_embeddings, columns=['dim1', 'dim2', 'dim3', 'dim4', 'dim5'])
df_pca['sentences'] = text_list
df_pca['dim1'] *= 100
df_pca['dim2'] *= 100
df_pca['dim3'] *= 100
df_pca['dim4'] *= 100
df_pca['dim5'] *= 100
df_pca['dim5'] = df_pca['dim5'].abs()
print("PCA Dataframe created.")


df_umap = pd.DataFrame(umap_reduced_embeddings, columns=['dim1', 'dim2', 'dim3', 'dim4', 'dim5'])
df_umap['sentences'] = text_list
df_umap['dim5'] = df_umap['dim5'].abs()

print("UMAP Dataframe created.")

#part 5 - create a 3D scatter plot using plotly to visualize the sentences in 3D space
fig_pca = px.scatter_3d(
    df_pca, 
    x='dim1',
    y='dim2',
    z='dim3',
    color='dim4',
    size='dim5',
    hover_data=['sentences'],
    title=f'PCA 3D Scatter Plot (Total Variance Explained: {pca_total_variance:.2%})'
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



