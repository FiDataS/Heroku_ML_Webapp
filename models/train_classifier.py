# import libraries
import sys
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from sqlalchemy import create_engine
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import pickle

def load_data(database_filepath):
    """
    Loads data from SQL Database and returns
    feature and target variables and the category names
    
    :param database_filepath: SQL database file (string)
    
    :returns X: Features (messages) (dataframe)
    :returns Y: Targets (categories) (dataframe)
    :returns category_names: Labels of categories (Targets) (list)
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('Categorized_Messages', con=engine)
    #Define feature and target variables (X and Y)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = df.iloc[:, 4:].columns
    return X, Y, category_names

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

def build_model():
    #Create a ML Pipeline with Tdidf Vectorizer and OneVsRestClassifier + LinearSVC as MultiOutputClassifier
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
        ('multi_clf', MultiOutputClassifier(OneVsRestClassifier(LinearSVC())))])
    #Define parameters for GridSearch
    parameters = {
        ''
        'tfidf__ngram_range': ((1, 1), (1, 2)), #consider unigrams and bigrams
        'tfidf__use_idf': [True, False], #Enable/Disable document frequency reweighting
        'multi_clf__estimator__estimator__dual':[True, False] #Select algorithm to either solve dual or primal optimization problem
    }
    #create a GridSearch object
    cv = GridSearchCV(pipeline, param_grid=parameters)
    #return the GridSearch object
    return cv
    


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates the model's performance with test data and prints out the Classification Report
    
    :param model: model from Gridsearch (GridSearchCV Object)
    :param X_test: Test features (dataframe)
    :param Y_test: Test targets (dataframe)
    :param category_names: Labels of targets (Index object)
    
    :return: None
    """
    #predict with X_test (cv.predict calls on the estimator with the best found parameters)
    y_pred = model.predict(X_test)
    #print the best parameters for the model
    print("\nBest Parameters:", model.best_params_)
    #print the classification report for the prediction 
    print("\nClassification Report:\n")
    print(classification_report(Y_test, y_pred,target_names=category_names))
    

def save_model(model, model_filepath):
    """
    Saves trained model as pickle file
    
    Comment: When loading that file again and using model.predict
    The best model from the GridSearchCV Object is used since
    cv.predict calls on the estimator with the best found parameters
    
    :param model: Trained model (GridSearchCV Object)
    :param model_filepath: Filepath for storing the model (string)
    
    :return: None
    """
    # save model as pickle file
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