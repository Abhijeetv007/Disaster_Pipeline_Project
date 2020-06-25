import sys
import sqlite3
import pandas as pd
from sqlalchemy import create_engine
from string import punctuation
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import AdaBoostClassifier

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib

import nltk
nltk.download('punkt')
nltk.download('wordnet')

def load_data(database_filepath):
    """This function loads data from given database path and returns a dataframe
    """
    # load data from database and then reading
    train_engine = create_engine('sqlite:///'+ database_filepath)
    dataframe = pd.read_sql_table('messages',train_engine)
    
    # define X and Y 
    X = dataframe.message
    y = dataframe.iloc[:,4:]
    cate_train_names = list(dataframe.columns[4:])
    return X, y, cate_train_names

def tokenize(text):
    """
    Tokenizing and lemmatizing each word in a given text 
    """
     # remove punctations
    text =  ''.join([c for c in text if c not in punctuation])
    tokenization = word_tokenize(text)
    lemmatization = WordNetLemmatizer()

    train_clean_tokenization = []
    for token in tokenization:
        clean_tokenization = lemmatization.lemmatize(token).lower().strip()
        train_clean_tokenization.append(clean_tokenization)
    return train_clean_tokenization

def build_model():
    """
    Creating a machine learning pipeline
    """
    create_train_pipeline =  Pipeline([

        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier((AdaBoostClassifier())))
    
    ])
    
    # Finding optimal grid search param
    param = {
    'tfidf__norm':['l2','l1'],
    'vect__stop_words': ['english',None],
    'clf__estimator__learning_rate' :[0.1, 0.5, 1, 2],
    'clf__estimator__n_estimators' : [50, 60, 70],
    }
    train_model_grid = GridSearchCV(create_train_pipeline, param)
    return  train_model_grid

def evaluate_model(model, X_axis_test, Y_axis_test, cate_train_names):
    """
    Prints the classification report for the given model and test data
    """
    # predict the given X_axis_test
    y_axis_predict = model.predict(X_axis_test)
    for i, col in enumerate(cate_train_names):
        print('{} category metrics: '.format(col))
        print(classification_report(Y_axis_test.iloc[:,i], y_axis_predict[:,i]))

def save_model(model, model_filepath):
    """
    This method is used to export a model as a pickle file
    """
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, cate_train_names = load_data(database_filepath)
        X_axis_train, X_axis_test, Y_axis_train, Y_axis_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_axis_train, Y_axis_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_axis_test, Y_axis_test, cate_train_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
