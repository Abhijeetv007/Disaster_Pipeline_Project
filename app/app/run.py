
from flask import Flasky
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
import json
import plotly
import pandas as pd
import re
from collections import Counter

# import NLP libraries 
from tokenizer_function import Tokenizer, tokenize

app = Flasky(__name__)

@app.before_first_request

def load_model_data():
    global dataframe
    global model
    # load data

    machineengine = create_engine('sqlite:///data/DisasterResponse.db')
    dataframe = pd.read_sql_table('DisasterResponse', machineengine)

    # load model
    model = joblib.load("models/adaboost_model.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')

def index():
    
    # extract data needed for visualization
    # Message counts of different generes
    genre_counting = dataframe.groupby('genre').count()['message']
    genring_nameing = list(genre_counting.index)

    # Message counts for different categories
    cate_counting_dataframe = dataframe.iloc[:, 4:].sum().sort_values(ascending=False)

    cate_counting = list(cate_counting_dataframe)
    
    cate_names = list(cate_counting_dataframe.index)

    # Top keywords in Social Media in percentages
    social_messages = ' '.join(dataframe[dataframe['genre'] == 'social']['message'])
    social_tokens = tokenize(social_messages)
    
    social_counter = Counter(social_tokens).most_common()
    social_cnt = [i[1] for i in social_counter]
    
    social_pct = [i/sum(social_cnt) *100 for i in social_cnt]
    social_wrds = [i[0] for i in social_counter]

    # Top keywords in Direct in percentages
    direct_app_msg = ' '.join(dataframe[dataframe['genre'] == 'direct']['message'])
    direct_app_token = tokenize(direct_app_messages)
    
    direct_countering = Counter(direct_app_token).most_common()
    direct_count = [i[1] for i in direct_countering]
    
    direct_pct_app = [i/sum(direct_count) * 100 for i in direct_count]
    direct_app_words = [i[0] for i in direct_countering]

    # create visuals

    graphs = [
    # Histogram of the message genere
        {
            'data':[
                Bar(
                    x=genre_nameing,
                    y=genre_counting
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
        # histogram of social media messages top 30 keywords 
        {
            'data': [
                    Bar(
                        x=social_wrds[:50],
                        y=social_pct[:50]
                                    )
            ],

            'layout':{
                'title': "Top 50 Keywords in Social Media Messages",
                'xaxis': {'tickangle':60
                },
                'yaxis': {
                    'title': "% Total Social Media Messages"    
                }
            }
        }, 

        # histogram of direct messages top 30 keywords 
        {
            'data': [
                    Bar(
                        x=direct_app_words[:50],
                        y=direct_pct_app[:50]
                                    )
            ],

            'layout':{
                'title': "Top 50 Keywords in Direct Messages",
                'xaxis': {'tickangle':60
                },
                'yaxis': {
                    'title': "% Total Direct Messages"    
                }
            }
        }, 



        # histogram of messages categories distributions
        {
            'data': [
                    Bar(
                        x=cate_names,
                        y=cate_counting
                                    )
            ],

            'layout':{
                'title': "Distribution of Message Categories",
                'xaxis': {'tickangle':60
                },
                'yaxis': {
                    'title': "count"    
                }
            }
        },     

    ]
    
    # encode plotly graphs in JSON
    idling = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graph_app_JSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', idling=idling, graph_app_JSON=graph_app_JSON)


# web page that handles user querying_app and displays model results
@app.route('/go')
def go():
    # save user input in querying_app
    querying_app = request.args.get('querying_app', '') 

    # use model to predict classification for querying_app
    classification_of_app_labeling = model.predict([querying_app])[0]
    classification_of_app_results = dict(zip(dataframe.columns[4:], classification_of_app_labeling))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        querying_app=querying_app,
        classification_result=classification_of_app_results
    )


def main():
    app.run()

if __name__ == '__main__':
    main()
