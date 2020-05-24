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
    raw_df = raw_df.drop_duplicates(subset='url')
    raw_df.set_index('name', inplace=True)
    raw_df.index = raw_df.index.where(~raw_df.index.duplicated(), raw_df.index + '_2')
    raw_df.index = raw_df.index.where(~raw_df.index.duplicated(), raw_df.index + '_3')
    raw_df.index = raw_df.index.where(~raw_df.index.duplicated(), raw_df.index + '_4')
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
                    document += (v[2].strip().replace('\n', ' ').replace('\r', ' ') + ' ')
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

def nmf_topic_modeling(corpus, tfidf_matrix, tfidf_feats, n_topics, n_words=10, max_iter=250, print_tab=False, n_features=15):
    ## NMF
    W, H = get_nmf(tfidf_matrix, n_components=n_topics, max_iter=max_iter)
    top_words = get_topic_words(H, tfidf_feats, n_features)
    df_pretty = print_topics(top_words, False)
    ## Add majority topic to hikes
    copy_maincorpus = corpus.copy()
    copy_maincorpus['topics'] = document_topics(W)
    ## Make df with loadings
    cols = ['topic_'+str(i) for i in range(1,n_topics+1)]
    W_df = pd.DataFrame(W.round(2), index=copy_maincorpus.index, columns=cols)
    W_df['majority_topic'] = document_topics(W)
    W_df['majority_topic'] += 1
    H_df = pd.DataFrame(H.round(2), index=cols, columns=tfidf_feats)
    return df_pretty, W_df, H_df

def hike_url_dict(raw_df):
    return {raw_df.index[i]: raw_df['url'][i] for i in range(len(raw_df))}   

additional_lemmatize_dict = {
    "biking": "bike",
    "bikes": "bike"
}

if __name__ == "__main__":
    df_raw = json_to_pandas('/Users/annierumbles/Dropbox/raw_colorado_hikes.json')
    df_raw = clean_df(df_raw)
    hike_dictionary = hike_url_dict(df_raw)
    rows, docs = get_review_corpus(df_raw['reviews'])
    df_reviews = make_reviews_df(rows, docs)
    df_corpus = make_corpus_df(df_raw, df_reviews)
    df_hike = make_hike_df(df_raw)

    punc = string.punctuation
    stop = StopWords()
    stop_words = stop.all_words
    clean_column(df_corpus, 'all', punc)
    X_tfidf, feats_tfidf, tfidf_vect = vectorize(df_corpus, 'all', stop_words, 6000)
    df_trunc = pd.DataFrame(df_corpus['all'])
    df_pretty, W_df, H_df = nmf_topic_modeling(df_trunc, X_tfidf, feats_tfidf, n_topics=9, n_words=10, n_features=10)

    df_recommendations = df_hike.merge(W_df, left_index=True, right_index=True)



