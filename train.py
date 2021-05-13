# importing required libraries
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


data = pd.read_csv('twitter_sentiments.csv')
train, test = train_test_split(data, test_size = 0.2, stratify = data['label'], random_state=21)

tfidf_vectorizer = TfidfVectorizer(lowercase= True, max_features=1000, stop_words=ENGLISH_STOP_WORDS)
tfidf_vectorizer.fit(train.tweet)

train_idf = tfidf_vectorizer.transform(train.tweet)
test_idf  = tfidf_vectorizer.transform(test.tweet)
model_LR = LogisticRegression()

model_LR.fit(train_idf, train.label)

# save the model using joblib
import joblib
model_file_name = 'tweeter_lr_model.pkl'
vectorizer_filename = 'tweeter_vector.pkl'
joblib.dump(model_LR , model_file_name)
joblib.dump(tfidf_vectorizer , vectorizer_filename)

