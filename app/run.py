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

import re
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.corpus import stopwords


app = Flask(__name__)

def tokenize(text):
    """
    Custom Tokenize function that:
        - Removes any punctation and converts text in lower case
        - Tokenizes in words
        - Removes Stopwords
        - Lemmatization on nouns and verbs
    
    :param text: Message as text data (string)
    
    :returns lemm: Processed text (list)
    """
    #Remove Punctation and convert to lower case
    text=re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
    #Tokenize in words
    text=word_tokenize(text)
    #Remove Stop words
    stop_words = stopwords.words('english')
    st_text=[w for w in text if w not in stop_words]
    #Lemmatization  
    #First lemmatization on nouns
    lemm = [WordNetLemmatizer().lemmatize(w) for w in st_text]
    #Second Lemmatization on verbs
    lemm = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemm]
    #return the processed text
    return lemm

# load data
engine = create_engine('sqlite:///../data/Disaster_Response_new.db')
df = pd.read_sql_table('Categorized_Messages', engine)
df_c = pd.read_sql_table('Country_Count', engine)

# load model
model = joblib.load("../models/model_classifier_new_new.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    
    category_train_count = df[df.columns[4:]].sum().sort_values(ascending=False)
    category_names = list(category_train_count.index)
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_train_count
                )
            ],

            'layout': {
                'title': 'Distribution of Message Classification in Training Data',
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
                    x=list(df_c['country']),
                    y=list(df_c['count_messages'])
                )
            ],

            'layout': {
                'title': 'Distribution of Message Origins (Countries) in Training Data',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Countries"
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

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


#def main():
#    app.run(host='127.0.0.1', port=3000, debug=True)

#for udacity workspace 0.0.0.0
#for local workspace 127.0.0.1

#if __name__ == '__main__':
#    main()