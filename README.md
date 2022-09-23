# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### About the algorithm

1. Raw text was tokenized, lemmatized and cleaned (numbers outside words and punctuation were removed)

2. Sentence2Vectorizer() is a custom transform that uses Word2Vector and TfidfVectorizer to create vectors representing messages or sentences:
    - TfidfVectorizer was used to assess idf values for each word.
    - A vector was created to designate each word in the whole of the text messages.
    - Each word vector inside the message is then multiplied by idf value 
    - For each message, the mean of the word vectors is calculated.
    
3. XGBoost was used as classifier.