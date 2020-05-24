import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import json, string, random
from nlp_pipeline import *
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.decomposition import NMF, PCA
from stopwords_class import StopWords

def json_to_pandas(filepath):
    '''
    Read json file to Pandas dataframe

    Input: json filepath
    Output: Pandas dataframe
    '''
    return pd.read_json(filepath, lines=True)

def clean_df(raw_df):
    raw_df.drop('_id', axis=1,inplace=True)
    raw_df.set_index('name', inplace=True)
    return raw_df


def get_review_corpus(data):
    rows = []
    docs = []
    for i in range(len(data)):
        row_name = data.index[i]
        document = ''
        if data[i] is not None:
            for k, v in data[i].items():
                if type(v) == list:
                    document += (v[2].strip().replace('\n', ' ') + ' ')
            docs.append(document)
            rows.append(row_name)
        else:
            docs.append(document)
            rows.append(row_name)
    return rows, docs

def make_reviews_df(index, documents, col_names=['reviews']):
    '''
    Make review dataframe from hike reviews text.

    Input: Index list, documents list, column names.
    Output: Pandas dataframe
    '''
    return pd.DataFrame(documents, index=index, columns=col_names)

def make_corpus_df(raw_df, review_df):
    df_corpus = pd.DataFrame(df_raw[['url', 'tags', 'main_description', 'secondary_description']])
    df_corpus['review_string'] = df_reviews['reviews']
    df_corpus.fillna('', inplace=True)
    df_corpus['all'] = df_corpus['tags'] + ' ' + df_corpus['main_description'] + ' ' + df_corpus['secondary_description'] + ' ' + df_corpus['review_string']
    return df_corpus

def make_hike_df(raw_df):
    df_hike = raw_df.copy()
    df_hike.drop(['tags', 'main_description', 'secondary_description', 'reviews'], axis=1, inplace=True)
    return df_hike

def get_top_words_tf(X, features, n_words=10):
    '''
    Get top n words from term-frequency matrix.

    Input: X matrix, words, int
    Output: Dictionary of top words and frequency
    '''
    summed = np.sum(X, axis=0)
    summed = np.array(summed).reshape(-1,) 
    indices_top = np.argsort(-summed)[:n_words]
    top_dict = {str(features[i]): int(summed[i]) for i in indices_top}
    return top_dict


if __name__ == "__main__":
    df_raw = json_to_pandas('/Users/annierumbles/Dropbox/raw_colorado_hikes.json')
    df_raw = clean_df(df_raw)
    rows, docs = get_review_corpus(df_raw['reviews'])
    df_reviews = make_reviews_df(rows, docs)
    df_corpus = make_corpus_df(df_raw, df_reviews)
    df_hike = make_hike_df(df_raw)