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

2. Sentence2Vectorizer() is a custom transform that uses Word2Vector to create vectors representing messages or sentences: 
    - A vector was created to designate each word in the whole of the text messages. 
    - For each message, the mean of the word vectors is calculated.
    
3. I used a random forest classifier - though other algorithms might perform best - because of hardware limitations.
