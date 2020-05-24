import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer, CountVectorizer
from nltk.tokenize import word_tokenize, WhitespaceTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.decomposition import NMF
from tabulate import tabulate

import spacy

def remove_punctuation(string, punc=string.punctuation):
    '''
    Remove punctuation from string.

    Input: string
    Output: string
    '''
    for character in punc:
        string = string.replace(character,'')
    return string

def lemmatize_str(string, wordnet):
    '''
    Lemmatize string using nltk WordNet

    Input: string
    Output: string
    '''
    if wordnet:
        w_tokenizer = WhitespaceTokenizer()
        lemmatizer = WordNetLemmatizer()
        lemmed = " ".join([lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(string)])
        return lemmed

def clean_column(df, column, punctuation):
    '''
    Lowercase words and apply punctuation removal and lemmatizing functions.

    Input: Dataframe, column to clean, punctuation list
    Output: Dataframe with cleaned column
    '''
    df[column] = df[column].apply(lambda x: str(x).lower())
    df[column] = df[column].apply(lambda x: remove_punctuation(x, punctuation))
    df[column] = df[column].apply(lambda x: lemmatize_str(x, wordnet=True))
    return 

def vectorize(df, column, stop_words, max_feat=1500):
    '''
    Create a tfidf vector from dataframe.

    Input: Dataframe, column name, stop word list, maximum words desired.
    Output: X matrix, features from vector, vectorizer object
    '''
    text = df[column].values
    vectorizer = TfidfVectorizer(stop_words = stop_words, max_features=max_feat) 
    X = vectorizer.fit_transform(text)
    features = np.array(vectorizer.get_feature_names())
    return X, features, vectorizer

def count_vectorize(df, column, stop_words, max_feat=1500):
    '''
    Create a term frequency vector from dataframe.

    Input: Dataframe, column name, stop word list, maximum words desired.
    Output: X matrix, features from vector, count vectorizer object
    '''
    text = df[column].values
    vect = CountVectorizer(stop_words=stop_words, max_features=max_feat)
    X = vect.fit_transform(text)
    features = np.array(vect.get_feature_names())
    return X, features, vect

def get_nmf(X, n_components=7, max_iter=200):
    '''
    Create term-feature and feature-document matrices with NMF.

    Input: X matrix (tfidf), number of topics, maximum interations before stopping
    Output: W and H matrices
    '''
    nmf = NMF(n_components=n_components, max_iter=max_iter, random_state=12345, alpha=0.0)
    W = nmf.fit_transform(X)
    H = nmf.components_
    return W, H

def get_topic_words(H, features, n_features):
    '''
    Get certain number of top feature names from H matrix and list of feature terms

    Input: H matrix from NMF, features, number of words 
    Output: Terms
    '''
    top_word_indexes = H.argsort()[:, ::-1][:,:n_features]
    return features[top_word_indexes]

def print_topics(topics, print=True):
    '''
    Print topics in tabular form and get a transposed dataframe of that data.

    Input: top words, boolean
    Output: Dataframe
    '''
    n_words = len(topics[0])
    cols = ["Word #"+ str(i+1) for i in range(n_words)]
    row_idx = [str(i+1) for i in range(len(topics))]
    df_pretty = pd.DataFrame(columns=cols)
    for topic in topics:
        df_pretty = df_pretty.append([dict(zip(cols, topic))])
    df_pretty['Topic #'] = row_idx
    df_pretty = df_pretty.set_index('Topic #')
    if print:
        print(tabulate(df_pretty, headers='keys', tablefmt='github'))
    return df_pretty.transpose()

def document_topics(W):
    '''
    Sort W matrix in descending order.
    '''
    return W.argsort()[:,::-1][:,0]

if __name__ == "__main__":
    pass