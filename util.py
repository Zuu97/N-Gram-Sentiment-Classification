import os
import re
import json
import operator
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from variables import*

np.random.seed(seed)

from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

initial = {} # start of a phrase
second_word = {}
transitions = {}

def remove_stop_words(stopwords_list,sentence):
    return [k for k in sentence if k not in stopwords_list]

def preprocess_one(row):
    review = row['extract']
    stopwords_list = stopwords.words('english')
    tokenizer = RegexpTokenizer(r'\w+')
    review = review.lower() # Lowercase the review
    remove_punc = tokenizer.tokenize(review) # Remove puntuations
    remove_num = [re.sub('[0-9]', '', i) for i in remove_punc] # Remove Numbers
    updated_review = [i for i in remove_num if len(i)>0] # Remove empty strings
    updated_review = remove_stop_words(stopwords_list,updated_review) # remove stop words
    updated_review = ' '.join(updated_review)
    return updated_review

def update_csv_data(csv_path, raw_csv):
    df = pd.read_csv(raw_csv)
    if 'pre_extracted' not in df.columns.values:
        print("Creating {}".format(csv_path))
        df['pre_extracted'] = df.apply(preprocess_one, axis=1)
        df = df.dropna(axis = 0, how ='any')
        df.to_csv(csv_path, encoding='utf-8')

def get_train_data(classifier_idx):
    if not os.path.exists(train_csv):
        update_csv_data(train_csv, raw_train_csv)
    df = pd.read_csv(train_csv)
    df = df.dropna(axis = 0, how ='any')
    reviews = df['pre_extracted'].values
    scores  = df['score'].values
    reviews, scores = shuffle(reviews, scores)

    Ntrain = int(len(scores) * cutoff)
    TRAINreviews, TRAINscores = reviews[:Ntrain], scores[:Ntrain]
    VALreviews, VALscores = reviews[Ntrain:], scores[Ntrain:]

    Ytrain = (TRAINscores >= 8 )
    Yval = (VALscores >= 8 )

    vectorizer, Xtrain = embedding(classifier_idx, TRAINreviews)
    Xval = vectorizer.transform(VALreviews)
    return Xtrain, Ytrain, Xval, Yval, vectorizer

def get_test_data(vectorizer):
    if not os.path.exists(test_csv):
        update_csv_data(test_csv, raw_test_csv)
    df = pd.read_csv(test_csv)
    df = df.dropna(axis = 0, how ='any')
    reviews = df['pre_extracted'].values
    original_reviews = df['extract'].values
    Xtest = vectorizer.transform(reviews)
    return original_reviews, Xtest

def embedding(classifier_idx, reviews):
    print(" Vectorization.......\n")
    if classifier_idx == 1:
        ngram_range, min_df = (1,1), 10
    elif classifier_idx == 2:
        ngram_range, min_df = (1,2), 20
    elif classifier_idx == 3:
        ngram_range, min_df = (1,3), 30

    print(" Building C1assifier [ {} ]\n ngram range = {}\n minimum document appearence = {}\n".format(model_name, ngram_range, min_df))
    vectorizer = CountVectorizer(ngram_range=ngram_range, min_df=min_df)
    vectorizer.fit(reviews)
    X = vectorizer.transform(reviews)
    return vectorizer, X

def get_sentiment_data(classifier_idx = classifier_idx):
    Xtrain, Ytrain, Xval, Yval, vectorizer = get_train_data(classifier_idx)
    XtestOrg, Xtest = get_test_data(vectorizer)

    print(" Xtrain Shape : {}".format(Xtrain.shape))
    print(" Ytrain Shape : {}".format(Ytrain.shape))
    print(" Xval Shape   : {}".format(Xval.shape))
    print(" Yval Shape   : {}".format(Yval.shape))
    print(" Xtest Shape  : {}\n".format(Xtest.shape))

    return Xtrain, Ytrain, Xval, Yval, Xtest, XtestOrg