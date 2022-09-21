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
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
import sentence2vector as s2v

def load_data(database_filepath):
    # lod data from database.db
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages', con = engine)
    X = df['message']
    Y = df.iloc[:,4:-1]
    # drop "child_alone" feature: only 0
    Y = Y.drop("child_alone", axis=1)

    return X, Y, Y.columns

def tokenize(text):
    # tokenize, remove ponctuation, numbers (that are not part of words) and stopwords
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
    
    pipeline = Pipeline([('sentence2vectorize', s2v.Sentence2Vectorizer()),
                     ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    parameters = {
    'clf__estimator__n_estimators': [50, 100, 200],
    'clf__estimator__min_samples_split': [2, 3, 4]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = pd.DataFrame(model.predict(X_test))

    print(len(Y_pred.columns) - len(Y_test.columns))
    for i in range(len(Y_test.columns)):
        print(Y_test.columns[i])
        print(classification_report(Y_test.iloc[:,i], Y_pred.iloc[:,i], zero_division=False))

def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
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