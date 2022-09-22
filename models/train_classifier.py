# import libraries
import sys

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import pandas as pd
import re
from sqlalchemy import create_engine

from sklearn.metrics import classification_report
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
import sentence2vector as s2v

from catboost import CatBoostClassifier
from xgboost import XGBClassifier

def load_data(database_filepath):
    '''
    load_data
    Load data from database.db
    
    Input:
    database_filepath   filepath to database .db file

    Returns:
    X, Y    features and target dataframes
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages', con = engine)
    X = df['message']
    Y = df.iloc[:,4:-1]
    # drop "child_alone" feature: only 0
    Y = Y.drop("child_alone", axis=1)

    return X, Y, Y.columns

def tokenize(text):
    '''
    tokenize
    Tokenize, remove ponctuation, numbers (that are not part of words) and stopwords
    
    Input:
    text    string message entry

    Returns:
    clean_tokens    clean tokens string list
    '''
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r" \d+", " ", text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        if tok not in stopwords.words('english'):
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)
    
    return clean_tokens

def build_model():
    '''
    build_model
    Builds pipeline model with GridSearvhCV

    Input:

    Returns:
    cv    GridSearvhCV model
    '''

    pipeline = Pipeline([('sentence2vector', s2v.Sentence2Vectorizer()),
        ('clf', MultiOutputClassifier(XGBClassifier(max_depth=4,n_estimators=100)))
    ])

    # parameters = {
    # 'clf__estimator__n_estimators': [200],
    # 'clf__estimator__min_samples_split': [4]
    # }

    # cv = GridSearchCV(pipeline, param_grid=parameters)

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    evaluate_model
    Prints model reports by running classification_report() for each column in Y
    
    Input:
    model   Pipeline model
    X_test  Test instance of feature dataframe
    Y_test  Test instance of target dataframe
    category_names  Y column names
    
    Returns:
    '''

    Y_pred = pd.DataFrame(model.predict(X_test))

    print(len(Y_pred.columns) - len(Y_test.columns))
    for i in range(len(Y_test.columns)):
        print(Y_test.columns[i])
        print(classification_report(Y_test.iloc[:,i], Y_pred.iloc[:,i], zero_division=False))

def save_model(model, model_filepath):
    '''
    save_model
    Saves trained model in model_filepath path

    Input:
    model   Pipeline trained model
    model_filepath  Path for saving the model

    Returns:
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    '''
    main
    Loads data from database_filepath; 
    split training and test sets;
    builds, fits and evaluate model;
    saves model to model_filepath.
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        # X, Y = X.iloc[0:99], Y.iloc[0:99,:]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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