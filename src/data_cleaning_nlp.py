import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import json, string, random
from nlp_pipeline import remove_punctuation, lemmatize_str, clean_column, vectorize, count_vectorize, get_nmf, get_topic_words, print_topics, document_topics
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.decomposition import NMF, PCA

import spacy

from nltk.corpus import stopwords

def json_to_pandas(filepath):
    '''
    Read json file to Pandas dataframe

    Input: json filepath
    Output: Pandas dataframe
    '''
    return pd.read_json(filepath, lines=True)

def clean_hikes(df):
    '''
    Clean hike information dataframe

    Input: Pandas dataframe
    Output: Pandas dataframe, cleaned
    '''
    df = df.drop(['_id'], axis=1)
    cols_to_dummy = ['difficulty', 'hike_type']
    df = pd.get_dummies(df, columns=cols_to_dummy,drop_first=True)
    cols_to_rename = {'hike_type_Out & Back':'out_and_back', 'hike_type_Point to Point':'point_to_point'}
    df = df.rename(columns=cols_to_rename)
    s = pd.Series({i:df['tags'][i] for i in range(len(df))})
    tag_dummies = pd.get_dummies(s.apply(pd.Series).stack()).sum(level=0) ## THIS DOESN'T DROP A CERTAIN COLUMN, FYI
    df = df.join(tag_dummies).drop(['tags'], axis=1)
    tags = df.columns[12:-1]
    n = [t.replace(' ','_') for t in tags]
    rename = {tags[i]:n[i] for i in range(len(tags))}
    df = df.rename(columns=cols_to_rename)
    df.set_index('name', inplace=True)
    df.rename_axis(None, inplace=True)
    return df

def _json_adjustments(json_file):
    '''
    Read in each line of a json file.

    Input: json filepath
    Output: List of lists
    '''
    data = [json.loads(line) for line in open(json_file, 'r')]   
    return data 

def get_corpus_info(json_file, name='hike_name'):
    '''
    Get index information and documents for main corpus.

    Input: json filepath, name associated with json file name (i.e. Hike name)
    Output: Index list, list of documents
    '''
    data = _json_adjustments(json_file)
    rows = []
    docs = []
    for hike in data:
        row_name = hike[name]
        document = ''
        for k, v in hike.items():
            if type(v) == list:
                document += (v[1] + ' ')
            elif len(hike) == 2:
                document = 'None'
        docs.append(document)
        rows.append(row_name)
    return rows, docs


def get_description_text(json_file):
    '''
    Get description dataframe from scraped Alltrails description.

    Input: json filepath
    Output: Pandas dataframe
    '''
    data = _json_adjustments(json_file)
    description = {data[i]['name']: data[i]['description'] for i in range(len(data))}
    df_description = pd.DataFrame.from_dict(description, orient='index', columns=['description'])
    return df_description

def add_tags_to_df(merged_df):
    '''
    Merge individual tags to review text in dataframe.

    Input: Pandas dataframe with reviews and tags columns
    Output: Combined text column
    '''
    text = []
    for hike in range(len(merged_df)):
        tags = merged_df['tags'][hike]
        x = merged_df['reviews'][hike]
        x = str(x)
        if x != 'None':
            for i in range(len(tags)):
                x += str(tags[i]+' ')
        else:
            x = ''
            for i in range(len(tags)):
                x += tags[i]+' '
        text.append(x)
    merged_df['all'] = text
    return merged_df

def reviews_and_description(review_df, desc_df):
    '''
    Combined reviews and description text.

    Input: Reviews Pandas dataframe, hike description Pandas dataframe
    Output: Pandas dataframe with column of all combined text
    '''
    review_df_sorted = review_df.sort_index(axis=0)
    desc_df_sorted = desc_df.sort_index(axis=0)
    merged_corpus = pd.merge(review_df_sorted, desc_df_sorted, left_index=True, right_index=True, how='inner')
    return merged_corpus

def make_corpus_df(index, documents, col_names=['reviews']):
    '''
    Make main corpus from hike reviews text.

    Input: Index list, documents list, column names.
    Output: Pandas dataframe
    '''
    return pd.DataFrame(documents, index=index, columns=col_names)

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
    Run non-negative matrix factorization.

    Input: Pandas dataframe - corpus of documents, NP array - TFIDF matrix, list - feature names from TFIDF vectorizer, int - number 
        of topics, int - number of words, int - number of iterations in NMF, bool - print tabulated dataframe.
    Output: Pandas dataframe - top n words as rows, topic # as column; Pandas dataframe - W matrix, how the topics load onto each
        hike; Pandas dataframe - H matrix, how the words load onto each topic
    '''
    ## NMF
    W, H = get_nmf(tfidf_matrix, n_components=n_topics, max_iter=max_iter)
    top_words = get_topic_words(H, tfidf_feats, n_words=15)
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

if __name__ == "__main__":
    random.seed(9)

    pkl = False
    tags_adding = False
    pipeline = True
    tfidf_pca = False

    ## This is loading each line in individually so I can pull out the correct info
    index, documents = get_corpus_info('data/co_users_raw.json')
    ## Create corpus dataframe
    user_reviews = make_corpus_df(index, documents)
    ## Add description text to user review corpus
    df_description = get_description_text('data/hike_descriptions.json')
    main_corpus = reviews_and_description(user_reviews, df_description)
    main_corpus['all'] = main_corpus['reviews'] + main_corpus['description']
    main_corpus.drop(['reviews', 'description'], axis=1, inplace=True)
    ## Add tags from hiking dataset
    if tags_adding:
        df = pd.read_json('data/co_hikes_raw.json', lines=True)
        df = df.drop(['_id'], axis=1)
        cols_to_dummy = ['difficulty', 'hike_type']
        df = pd.get_dummies(df, columns=cols_to_dummy,drop_first=True)
        cols_to_rename = {'hike_type_Out & Back':'out_and_back', 'hike_type_Point to Point':'point_to_point'}
        df = df.rename(columns=cols_to_rename)
        s = pd.Series({i:df['tags'][i] for i in range(len(df))})
        df_tags = pd.DataFrame(s, columns=['tags'])
        df_tags.set_index(df['name'], inplace=True)

        df_tags_sorted = df_tags.sort_index(axis=0)
        df_corpus_sorted = user_reviews.sort_index(axis=0)
        ## Merge tags and reviews
        merged_corpus = pd.merge(df_corpus_sorted, df_tags_sorted, left_index=True, right_index=True, how='inner')
        merged_corpus.rename_axis(None, inplace=True)
        new_merged_corpus = add_tags_to_df(merged_corpus)
        new_merged_corpus.drop(['tags', 'reviews'], axis=1, inplace=True)
    

    stop_nums = {'10', '100', '1000', '1030', '10am', '11', '11am', '12', '13', '14', '15',
        '16', '17', '18', '1st', '20', '2017', '2018', '2019', '22', '23', '24',
        '25', '2nd', '30', '33', '34', '35', '3rd', '40', '45', '4th', '50', '55',
        '56', '60', '630', '65', '6am', '70', '730', '730am', '75', '7am',
        '830', '85', '8am', '90', '930', '9am','01', '02', '025', '03', '04', '05', '06', '07', '08', '09',
       '10000', '1000am', '1000ft', '101', '1010', '1012', '1014', '1015',
       '102', '1020', '103', '1030am', '104', '1045', '105', '10500',
       '106', '107', '108', '109', '10k', '10th', '110', '1100', '11000',
       '1100am', '111', '1115', '112', '113', '1130', '1130am', '114',
       '1145', '115', '11500', '116', '117', '118', '119', '11k', '11th',
       '120', '1200', '12000', '121', '1215', '122', '123', '1230',
       '1230pm', '124', '125', '12500', '126', '127', '128', '129', '12k',
       '12pm', '12th', '130', '1300', '13000', '130pm', '132', '133',
       '135', '13er', '13ers', '13k', '13th', '1400', '14000', '145',
       '14erscom', '14k', '14ner', '14th', '150', '1500',
       '152', '1520', '15th', '1600', '175', '17th', '1800', '18th', '19',
       '19th', '1hr', '1pm', '200', '2000', '2010', '2011', '2012',
       '2013', '2014', '2015', '2016', '2020', '2030', '20th', '21',
       '215', '21st', '225', '230', '230pm', '23rd', '23rds', '245',
       '24th', '250', '2500', '253', '26', '26th', '27', '275', '28',
       '285', '28th', '29', '2hrs', '2ish', '2l', '2miles', '2pm', '2wd',
       '2x', '300', '3000', '3040', '3045', '30min', '30th', '31', '315',
       '32', '324', '325', '330', '330pm', '33s', '345', '35s', '36',
       '360', '37', '375', '379', '38', '39', '3hrs', '3l', '3pm', '400',
       '41', '415', '42', '425', '43', '430', '430am', '430pm', '44',
       '445', '45min', '46', '47', '48', '49', '4am', '4hrs', '4pm',
       '4runner', '4wheel', '4x4s', '500', '5050', '51',
       '510', '515', '515am', '52', '525', '526', '53', '530', '530am',
       '530pm', '54', '540', '545', '545am', '550', '57', '58', '59',
       '5am', '5pm', '5star', '5th', '600', '61', '610', '615', '615am',
       '616', '62', '620', '622', '625', '629', '63', '630am', '64',
       '645', '645am', '650', '66', '667', '67', '68', '69', '6pm', '6th',
       '700', '700am', '71', '710', '713', '714', '715', '715am', '72',
       '720', '73', '74', '745', '745am', '750', '76', '77', '78', '79',
       '7pm', '7th', '80', '800', '800am', '81', '810', '815', '815am',
       '82', '820', '83', '830am', '84', '845', '845am', '86', '87', '88',
       '89', '8th', '900', '9000', '900am', '91', '910', '914', '915',
       '915am', '92', '93', '930am', '94', '945', '945am', '95', '96',
       '97', '98', '99', '9th'}
    stop_items ={'trail', 'wa', 'hike', 'hiking', 'hiked', 'offer', 'activity', 'around', 'lightly',
        'used', 'primarily', 'number', 'able', 'see', 'located', 'mile', 'also', 'kept', 'option',
        'way', 'get', 'near', 'must', 'pas', 'lot', 'colorado', 'rated', 'trafficked', 'back',
        'use', 'feature', 'upper', 'definitely', 'trailhead', 'one', 
        'along', 'level', 'good',  'point', 'little', 'chance', 'october', 'september', 'pa'}
    stop_tentopics = {'loop', 'moderate', 'april', 'fruita'}
    stop_words = set(stopwords.words('english')).union(stop_nums).union(stop_items)

    ## Using nlp_pipeline.py
    if pipeline:
        three = False
        eight = False
        seven = False
        six = False
        ten = False
        reconstruction_error = False

        punc = string.punctuation
        clean_column(main_corpus, 'all', punc) 
        X_tfidf, feats_tfidf, tfidf_vect = vectorize(main_corpus, 'all', stop_words, 6000)
        if three:
            df_pretty, W_df, H_df = nmf_topic_modeling(main_corpus, X_tfidf, feats_tfidf, n_topics=3, n_words=10)
        if eight:
            df_pretty, W_df, H_df = nmf_topic_modeling(main_corpus, X_tfidf, feats_tfidf, n_topics=8, n_words=10)
        if seven:
            df_pretty, W_df, H_df = nmf_topic_modeling(main_corpus, X_tfidf, feats_tfidf, n_topics=7, n_words=10)
        if six:
            df_pretty, W_df, H_df = nmf_topic_modeling(main_corpus, X_tfidf, feats_tfidf, n_topics=6, n_words=10)
        if ten: 
            df_pretty, W_df, H_df = nmf_topic_modeling(main_corpus, X_tfidf, feats_tfidf, n_topics=10, n_words=10)
            if pkl:
                W_df.to_pickle('data/nmftopics.pkl')
                H_df.to_pickle('data/nmfwords.pkl')
        if reconstruction_error:
            comps = np.arange(1,21,1)
            scores = []
            for i in comps:
                nmf = NMF(n_components=i, max_iter=250, random_state=12345, alpha=0.0)
                nmf.fit_transform(X_tfidf)
                score = nmf.reconstruction_err_
                scores.append(score)
            plt.style.use('ggplot')
            fig, ax = plt.subplots(1,1,figsize=(8,5))
            ax.plot(comps, scores, color='teal', marker='o', linestyle='dashed')
            ax.set_title('Elbow Plot for NMF Topic Modeling', fontsize=20)
            ax.set_xlabel('Number of Topics', fontsize=16)
            ax.set_ylabel('Reconstruction Error', fontsize=16)
            plt.tick_params(labelsize=14)
            plt.tight_layout(pad=1)
            plt.xticks(comps)
            # plt.savefig('images/nmf_elbowplot.png', dpi=80)
            plt.show()

        if tfidf_pca:

            pca = PCA(n_components=3)
            X_pca = pca.fit_transform(X_tfidf.toarray()) 
            fig = plt.figure() 
            ax = fig.add_subplot(111, projection='3d') 
            ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c='b', edgecolor='k', s=40) 
            ax.set_xlabel('PC1') 
            ax.set_ylabel('PC2') 
            ax.set_zlabel('PC3') 
            plt.show()
