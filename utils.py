import re
from cleantext import clean
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import umap
import pandas as pd

def open_and_clean_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        raw_text = file.read()

    cleaned_text = clean(raw_text,
        fix_unicode=True,
        to_ascii=True,
        lower=False,
        no_urls=True,
        no_emails=True,
        no_numbers=False,
        lang='en',
    )

    cleaned_text = cleaned_text.replace("\\'", "'").replace('\\"', '"')
    cleaned_text = cleaned_text.replace("\\n", " ").replace("\\t", " ")
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    return cleaned_text


def embed(tokenized_sentences):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(tokenized_sentences, show_progress_bar=True)
    return embeddings

def pca_reduce(embeddings, dimensions):
    pca = PCA(n_components=dimensions)
    reduced_embeddings = pca.fit_transform(embeddings)
    return reduced_embeddings

def umap_reduce(embeddings, dimensions):
    umap_reduce = umap.UMAP(n_components=dimensions)
    umap_reduced_embeddings = umap_reduce.fit_transform(embeddings)
    return umap_reduced_embeddings


def create_5d_dataframe(reduced_embeddings, sentences, multiplier):
    df = pd.DataFrame(reduced_embeddings, columns=['dim1', 'dim2', 'dim3', 'dim4', 'dim5'])
    df['sentences'] = sentences
    df['dim1'] *= multiplier
    df['dim2'] *= multiplier
    df['dim3'] *= multiplier
    df['dim4'] *= multiplier
    df['dim5'] *= multiplier
    df['dim5'] = df['dim5'].abs()
    return df
    