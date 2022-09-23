import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine

from sentence2vector import *

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")

X = df['message']
Y = df.iloc[:,4:-1]
# drop "child_alone" feature: only 0
Y = Y.drop("child_alone", axis=1)
Y_pred = pd.DataFrame(model.predict(X))
Y_pred_sum = pd.DataFrame(Y_pred.sum(),columns=["count"])
Y_sum = pd.DataFrame(Y.sum(),columns=["count"])
Y_pred_sum['type'] = Y.columns
Y_sum['type'] = Y.columns


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    classification_counts_Y = Y_sum['count']
    classification_names_Y = Y_sum['type']
    classification_counts_Y_pred = Y_pred_sum['count']
    classification_names_Y_pred = Y_pred_sum['type']
    
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
        },

        {
            'data': [
                Bar(
                    x=classification_counts_Y,
                    y=classification_names_Y
                )
            ],

            'layout': {
                'title': 'Distribution of Message Classification in Original Data',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Classification"
                }
            }
        },
        
        {
            'data': [
                Bar(
                    x=classification_counts_Y,
                    y=classification_names_Y
                )
            ],

            'layout': {
                'title': 'Distribution of Message Classification in Prediction',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Classification"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    graph_ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=graph_ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

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