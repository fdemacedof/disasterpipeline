from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('stopwords')
from pandas import DataFrame
from gensim.models import Word2Vec
import re

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
        
        # messages such as '//// // @:@ ' would result in an empty token
        # instead, inseart "NOTOKENS" as a missing value indicator
        
        if clean_tokens == []:
            clean_tokens.append("NOTOKENS")

        return clean_tokens

class Sentence2Vectorizer(BaseEstimator, TransformerMixin):

    def sentence2vectorize(self, X):
        # tokenize, remove ponctuation, numbers (that are not part of words) and stopwords
        tokens = [tokenize(x) for x in X]
        
        # create word2vec model and tfdif transform
        model = Word2Vec(tokens, min_count=1)

        # sentence vector - the average of all word vectors (multiplied by idf) for each entry
        sentence_vectors = []
        for sentence in tokens:
            world_vector_list = [model.wv[word] for word in sentence]
            average_vector = [sum(i)/len(sentence) for i in zip(*world_vector_list)]
            sentence_vectors.append(average_vector)

        sentence_vectors = DataFrame(sentence_vectors)
        
        return sentence_vectors

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.sentence2vectorize(X)
