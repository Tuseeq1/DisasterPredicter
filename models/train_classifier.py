import sys
import re
import pandas as pd
import numpy as np
import pickle
import nltk
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, classification_report, precision_score
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# from column_extracter import ColumnExtracter

from sklearn.base import BaseEstimator, TransformerMixin

nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')


class ColumnExtracter(BaseEstimator, TransformerMixin):
    # return specific column or columns
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.columns]

def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('Disaster', engine)
#     X = df[['message', 'genre_direct', 'genre_news', 'genre_social']]
#     Y = df[[col for col in df.columns if col not in ['id', 'message', 'original', 'genre_direct', 'genre_news',
#        'genre_social']]]
    X = df[['message']]
    Y = df[[col for col in df.columns if col not in ['id', 'message', 'original', 'genre_direct', 'genre_news',
       'genre_social', 'genre']]]
    return X, Y, Y.columns


def tokenize(text):
    # Normalize and remove punctuations and extra chars such as (, # 
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
    
    # tokenize and remove stop words and lemmatize
    lemmatizer = WordNetLemmatizer()
    
    text = [lemmatizer.lemmatize(word).strip() for word in word_tokenize(text) if word not in stopwords.words('english')]

    return text 


def build_model():
    # pipeline to extract message data vectoize it and perform TFIDF
#     nlp_pipeline = Pipeline([
#         ('msgExtractor', ColumnExtracter('message')),
#         ('vect', CountVectorizer(tokenizer=tokenize)),
#         ('tfidf', TfidfTransformer())
#     ])

#     # pipeline to extract rest of the columns to add later into df
#     gnre_pipeline = Pipeline([
#         ('msgExtractor', ColumnExtracter(['genre_direct', 'genre_news', 'genre_social'])),
#     ])
    
#     # final pipeline that mixes nlp result and other features
#     pipeline = Pipeline([
#         ('features', FeatureUnion([
#             ('nlp', nlp_pipeline),
#             ('gnre', gnre_pipeline)
#         ])),
#         ('clf',  MultiOutputClassifier(RandomForestClassifier()))
#     ])
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf',  MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # paramters for grid search
    parameters = {
        'clf__estimator__n_estimators': [50, 60, 70]
    }
    cv =  GridSearchCV(pipeline, parameters)
    
    return cv
    
    
def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test['message'])
    
    # make it a df with column names same as Y_test
    dfy_pred = pd.DataFrame(y_pred, columns = category_names)
    
    for col in category_names:
        print(f'{col}:')
        print(classification_report(Y_test[col], dfy_pred[col]))
        print('--------------------------------------------------------\n')


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb') )


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train['message'], Y_train)
        
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