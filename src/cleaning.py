# import pandas as pd
# import numpy as np 
import matplotlib.pyplot as plt
import json, string, random
from nlp_pipeline import *
from sklearn.decomposition import NMF, PCA
from stopwords_class import StopWords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

def json_to_pandas(filepath):
    '''
    Read json file to Pandas dataframe

    Input: json filepath
    Output: Pandas dataframe
    '''
    return pd.read_json(filepath, lines=True)

def clean_df(raw_df):
    '''
    Raw json dataframe cleaned and returned.

    Input: Pandas dataframe
    Output: Pandas dataframe
    '''
    raw_df.drop('_id', axis=1,inplace=True)
    raw_df = raw_df.drop_duplicates(subset='url')
    raw_df.set_index('name', inplace=True)
    raw_df.index = raw_df.index.where(~raw_df.index.duplicated(), raw_df.index + '_2')
    raw_df.index = raw_df.index.where(~raw_df.index.duplicated(), raw_df.index + '_3')
    raw_df.index = raw_df.index.where(~raw_df.index.duplicated(), raw_df.index + '_4')
    return raw_df

def get_review_corpus(data):
    '''
    Create dataframe containing concatenated reviews.

    Input: Pandas series.
    Output: List of hike names, list of strings
    '''
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
    '''
    Make main corpus dataframe of documents including: tags, main description, secondary description and reviews.

    Input: Pandas dataframes.
    Output: Pandas dataframe.
    '''
    df_corpus = pd.DataFrame(df_raw[['url', 'tags', 'main_description', 'secondary_description']])
    df_corpus['review_string'] = df_reviews['reviews']
    df_corpus.fillna('', inplace=True)
    df_corpus['all'] = df_corpus['tags'] + ' ' + df_corpus['main_description'] + ' ' + df_corpus['secondary_description'] + ' ' + df_corpus['review_string']
    return df_corpus

def make_hike_df(raw_df):
    '''
    Make main hike dataframe containing: url, location, difficulty, elevation, distance, average rating, number of ratings.

    Input: Pandas dataframe.
    Output: Pandas dataframe.
    '''
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

def nmf_topic_modeling(corpus, tfidf_matrix, tfidf_feats, n_topics, n_words=10, max_iter=250, print_tab=False):
    '''
    Perform NMF, get top n words for each topic, make dataframe for both W and H matrices.

    Input: Pandas dataframe, TFIDF matrix, TFIDF features, number of topics desired, number of top words desired, max iterations, print tabulate.
    Output: Pandas dataframe of topics and top words, pandas dataframes
    '''
    ## NMF
    W, H = get_nmf(tfidf_matrix, n_components=n_topics, max_iter=max_iter)
    top_words = get_topic_words(H, tfidf_feats, n_words)
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
    '''Make dictionary of hikes and their corresponding URLs.'''
    return {raw_df.index[i]: raw_df['url'][i] for i in range(len(raw_df))}   

def make_sim_matrix(merged_df, similarity_measure=cosine_similarity, mean=True, std=True):
    '''Normalize columns.'''
    ss = StandardScaler(with_mean=mean, with_std=std)
    df_scaled = pd.DataFrame(ss.fit_transform(merged_df), columns=merged_df.columns, index=merged_df.index) 
    X = df_scaled.values
    similarity_df = pd.DataFrame(cosine_similarity(X, X), index=df_scaled.index, columns=df_scaled.index)
    return df_scaled, similarity_df

def get_recommendations(baseline_hike, n_hikes):
    sim_series = similarity_df.loc[baseline_hike]
    idx = np.argsort(sim_series)[::-1][1:n_hikes+1]
    hikes = df_merged.index
    recs = [hikes[i] for i in idx]
    print(f"If you enjoyed {baseline_hike}, you may like these:\n")
    for i, rec in zip(idx, recs):
        print(f"{hikes[i]}: {hike_dictionary[rec]}")

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
    df_pretty, W_df, H_df = nmf_topic_modeling(df_trunc, X_tfidf, feats_tfidf, n_topics=8, n_words=10)

    df_merged = df_hike.merge(W_df, left_index=True, right_index=True).drop(['url', 'majority_topic', 'location', 'number_ratings'], axis=1)
    cols_dummy = ['difficulty', 'hike_type']
    df_merged = pd.get_dummies(df_merged, columns=cols_dummy, drop_first=True)
    cols_to_rename = {'hike_type_Out & Back':'out_and_back', 'hike_type_Point to Point':'point_to_point'}
    df_merged = df_merged.rename(columns=cols_to_rename)

    # ss = StandardScaler()
    # df_scaled = pd.DataFrame(ss.fit_transform(df_merged), columns=df_merged.columns, index=df_merged.index) 
    # X = df_scaled.values
    # similarity_df = pd.DataFrame(cosine_similarity(X, X)) 
    df_scaled, similarity_df = make_sim_matrix(df_merged)

    sim_series = similarity_df.loc['Royal Arch Trail']
    idx = np.argsort(sim_series)[::-1][1:11]
    hikes = df_merged.index
    for i in idx:
        print(hikes[i])
