from flask import Flask, request, render_template
import pickle, random
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)

def import_csv(filepath, idx_name='Unnamed: 0'):
    df = pd.read_csv(filepath)
    df.set_index(idx_name, inplace=True)
    df.rename_axis(None, inplace=True)
    return df

def hike_url_dict(raw_df):
    '''Make dictionary of hikes and their corresponding URLs.'''
    return {raw_df.index[i]: raw_df['url'][i] for i in range(len(raw_df))}

def make_sim_matrix(merged_df, similarity_measure=cosine_similarity, mean=True, std=True):
    '''
    Normalize columns, create similarity matrix.
    
    Input: Pandas dataframe, similarity metric, boolean
    Output: Pandas dataframes, numpy matrix
    '''
    ss = StandardScaler(with_mean=mean, with_std=std)
    df_scaled = pd.DataFrame(ss.fit_transform(merged_df), columns=merged_df.columns, index=merged_df.index) 
    X = df_scaled.values
    similarity_df = pd.DataFrame(similarity_measure(X, X), index=df_scaled.index, columns=df_scaled.index)
    return df_scaled, similarity_df, X

def get_hike_recommendations(baseline_hike, n_hikes):
    sim_series = similarity_df.loc[baseline_hike]
    idx = np.argsort(sim_series)[::-1][1:n_hikes+1]
    hikes = df_merged.index
    recs = [hikes[i] for i in idx]
    return recs

def get_hike_specs(recs):
    feature_names = ['difficulty', 'hike_type', 'location', 'distance', 'elevation']
    if type(recs) == str:
        specs = {recs: [df_raw.loc[recs][feat] for feat in feature_names]}
    else:
        specs = {rec: [df_raw.loc[rec][feat] for feat in feature_names] for rec in recs}
    return specs

df_raw = import_csv('../data/raw_hiking_data.csv')
df_corpus = import_csv('../data/corpus_data.csv')
df_hike = import_csv('../data/hike_data.csv')
df_merged = import_csv('../data/topics_and_numericalfeatures.csv')
# df_merged[['out_and_back', 'point_to_point']] = df_merged[['out_and_back', 'point_to_point']]*.5
# df_merged[['difficulty_hard', 'difficulty_moderate']] = df_merged[['difficulty_hard', 'difficulty_moderate']]*.5
df_merged.drop(['difficulty_hard', 'difficulty_moderate', 'out_and_back', 'point_to_point'], axis=1, inplace=True)
df_dogs_allowed = import_csv('../data/dogs_allowed.csv')
hike_urls = hike_url_dict(df_raw)
df_topics = df_merged[['topic_1', 'topic_2', 'topic_3', 'topic_4', 'topic_5', 'topic_6', 'topic_7', 'topic_8']]
majority = [np.argmax(df_topics.loc[i])+1 for i in df_topics.index]
df_topics['majority_topics'] = majority

hike_features= ['Hike Name', 'Difficulty', 'Hike Type', 'Location', 'Distance (miles)', 'Elevation Gain (feet)', 'Dogs Allowed?']
df_scaled, similarity_df, X = make_sim_matrix(df_merged)

# home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/trail_recommendations')
def get_recommendation():
    hike_names = df_raw.index
    return render_template('trail_recs.html', names=hike_names)

@app.route('/trail_recommendations', methods=['POST'])
def get_hike_form():
    trail = request.form['hikes']
    n = int(request.form['number of hikes'])
    recs = get_hike_recommendations(trail, n)
    original_hikespecs = get_hike_specs(trail)
    specs = get_hike_specs(recs)
    return render_template('results.html', trail=trail, recs=recs, hike_features=hike_features, hike_specs=specs, 
        urls=hike_urls, original=original_hikespecs, dogs=df_dogs_allowed.index)

@app.route('/get_started')
def group_topics():
    return render_template('get_started.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/bang-for-your-buck')
def t1():
    title = 'Bang for your Buck'
    desc = 'These hikes are solid options if you want an all around trek!'
    recs_all = df_topics[df_topics['majority_topics']==1].index.tolist()
    recs_samp = random.sample(recs_all, 20)
    specs = get_hike_specs(recs_samp)
    return render_template('topics.html', hikes=recs_samp, hike_features=hike_features, hike_specs=specs, title=title, 
        description=desc, urls=hike_urls, dogs=df_dogs_allowed.index)

@app.route('/flora-and-fauna')
def t2():
    title = 'Flora and Fauna'
    desc = 'For nature lovers!'
    recs_all = df_topics[df_topics['majority_topics']==2].index.tolist()
    recs_samp = random.sample(recs_all, 20)
    specs = get_hike_specs(recs_samp)
    return render_template('topics.html', hikes=recs_samp, hike_features=hike_features, hike_specs=specs, title=title, 
        description=desc, urls=hike_urls, dogs=df_dogs_allowed.index)

@app.route('/backcountry-activites')
def t3():
    title = 'Backcountry Activities'
    desc = 'Whether you want to fish, backpack, or camp in the backcountry'
    recs_all = df_topics[df_topics['majority_topics']==3].index.tolist()
    recs_samp = random.sample(recs_all, 20)
    specs = get_hike_specs(recs_samp)
    return render_template('topics.html', hikes=recs_samp, hike_features=hike_features, hike_specs=specs, title=title, 
        description=desc, urls=hike_urls, dogs=df_dogs_allowed.index)

@app.route('/offroad')
def t4():
    title = 'Off-Roading'
    desc = 'Options for people who want to explore via 4x4'
    recs_all = df_topics[df_topics['majority_topics']==4].index.tolist()
    recs_samp = random.sample(recs_all, 20)
    specs = get_hike_specs(recs_samp)
    return render_template('topics.html', hikes=recs_samp, hike_features=hike_features, hike_specs=specs, title=title, 
        description=desc, urls=hike_urls, dogs=df_dogs_allowed.index)

@app.route('/adventures-with-the-pup')
def t5():
    title = 'Adventures with the Pup'
    desc = "Perfect for a jog with your dog!"
    recs_all = df_topics[df_topics['majority_topics']==5].index.tolist()
    recs_samp = random.sample(recs_all, 20)
    specs = get_hike_specs(recs_samp)
    return render_template('topics.html', hikes=recs_samp, hike_features=hike_features, hike_specs=specs, title=title, 
        description=desc, urls=hike_urls, dogs=df_dogs_allowed.index)

@app.route('/water-seekers')
def t6():
    title = 'Water Feature-Seekers'
    desc = 'For waterfall and river admirers'
    recs_all = df_topics[df_topics['majority_topics']==6].index.tolist()
    recs_samp = random.sample(recs_all, 20)
    specs = get_hike_specs(recs_samp)
    return render_template('topics.html', hikes=recs_samp, hike_features=hike_features, hike_specs=specs, title=title, 
        description=desc, urls=hike_urls, dogs=df_dogs_allowed.index)

@app.route('/mtb')
def t7():
    title = 'MTB'
    desc = 'Mountain bikers - these are for you!'
    recs_all = df_topics[df_topics['majority_topics']==7].index.tolist()
    recs_samp = random.sample(recs_all, 20)
    specs = get_hike_specs(recs_samp)
    return render_template('topics.html', hikes=recs_samp, hike_features=hike_features, hike_specs=specs, title=title, 
        description=desc, urls=hike_urls, dogs=df_dogs_allowed.index)

@app.route('/peak-baggers')
def t8():
    title = 'Peak-Baggers'
    desc = 'Reach new heights!'
    recs_all = df_topics[df_topics['majority_topics']==8].index.tolist()
    recs_samp = random.sample(recs_all, 20)
    specs = get_hike_specs(recs_samp)
    return render_template('topics.html', hikes=recs_samp, hike_features=hike_features, hike_specs=specs, title=title, 
        description=desc, urls=hike_urls, dogs=df_dogs_allowed.index)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True, debug=False)
