import json
import plotly
import re
import pandas as pd

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

from sklearn.base import BaseEstimator, TransformerMixin
nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')


app = Flask(__name__)

def tokenize(text):
     # Normalize and remove punctuations and extra chars such as (, # 
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
    
    # tokenize and remove stop words and lemmatize
    lemmatizer = WordNetLemmatizer()
    
    text = [lemmatizer.lemmatize(word).strip() for word in word_tokenize(text) if word not in stopwords.words('english')]

    return text 

class ColumnExtracter(BaseEstimator, TransformerMixin):
    # return specific column or columns
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.columns]


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Disaster', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 
    
#     dfs = pd.DataFrame([[query, 0, 0, 1]], columns=['message', 'genre_direct','genre_news', 'genre_social'])
    # use model to predict classification for query
    print(model.predict([query]))
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[7:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()