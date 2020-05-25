from flask import Flask, render_template
import pickle
import pandas as pd
import numpy as np
app = Flask(__name__)

with open('../data/df.pkl', 'rb') as f:
    df = pickle.load(f)

hike_features= []

# home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/next_trail')
def get_recommendation():
    pass

@app.route('/get_started')
def group_topics():
    pass

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True, debug=False)
