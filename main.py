#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json

import numpy as np
import pandas as pd
import scipy.sparse as sp
import tensorflow as tf
from gensim.models.word2vec import Word2Vec
from keras.models import load_model
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
from idfs import idfs  # numpy array with our pre-computed idfs

tf.logging.set_verbosity(tf.logging.ERROR)


# subclass of TfidfVectorizer. Used to re-create the tfidf model
class MyVectorizer(TfidfVectorizer):
    # plug our pre-computed IDFs
    TfidfVectorizer.idf_ = idfs


tokenizer = TweetTokenizer()
n_dim = 200


# To get the pre-built tfidf
def getTfidf():
    # instantiate vectorizer
    vectorizer = MyVectorizer(lowercase=False,
                              min_df=10,
                              norm='l2',
                              smooth_idf=True)
    # plug _tfidf._idf_diag
    vectorizer._tfidf._idf_diag = sp.spdiags(idfs,
                                             diags=0,
                                             m=len(idfs),
                                             n=len(idfs))
    vectorizer.vocabulary_ = vocabulary
    tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
    return tfidf


# Load the pre-computed models
print "Loading pre-computed word2vec model"
Vmodel = Word2Vec.load('W2V Model.txt')

print "Loading pre-computed trained model with Sigmoid Activation function"
Lmodel = load_model('Model.h5')

print "Fetching pre-build Tf-idf"
vocabulary = json.load(open('vocabulary.json', mode='rb'))
tfidf = getTfidf()


# To tokenize the tweets
def tokenize(tweet):
    try:
        tweet = tweet.lower()
        tokens = tokenizer.tokenize(tweet)
        tokens = filter(lambda t: not t.startswith('@'), tokens)
        tokens = filter(lambda t: not t.startswith('#'), tokens)
        tokens = filter(lambda t: not t.startswith('http'), tokens)
        tokens = filter(lambda t: not t.startswith(','), tokens)
        tokens = filter(lambda t: not t.startswith('.'), tokens)
        tokens = filter(lambda t: not t.startswith('!'), tokens)
        tokens = filter(lambda t: not t.startswith('-'), tokens)
        return tokens
    except:
        return 'NC'


# To build the word vector using pre-computed word2vec model and tfidf
def buildWordVector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            # Multipy the embedded word2vec with its weights (tfidf)
            vec += Vmodel.wv[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


def predictSentiment(tweet):
    print "----------------------------------------"

    # Tokenize tweets into words
    print "Tokenizing Tweet"
    words = tokenize(tweet)

    # Convert words into word2vec and predict model
    tweet_w2v = buildWordVector(words, n_dim)
    tweet_w2v = tweet_w2v[..., np.newaxis]

    print "Predicting the accuracy"
    predicted_value = Lmodel.predict(tweet_w2v)

    print "Predicted value = ", predicted_value.flat[0]
    return predicted_value.flat[0]


dataframe = pd.read_csv(sys.argv[1], encoding='latin-1')
count = 0
retweet_predictions = []
tweetID_sentiment_dict = {}
tweetID_tweet_dict = {}

actual_tweetID_veracity_dict = {}
predicted_tweetID_veracity_dict = {}

row_iterator = dataframe.itertuples(index=True, name='Pandas')
# for x in range(23):
#     row_iterator.next()
for row in row_iterator:
    tweet_id = getattr(row, "TWEET_ID")
    tweetID_tweet_dict[tweet_id] = getattr(row, 'TWEET')
    actual_tweetID_veracity_dict[tweet_id] = getattr(row, 'RUMOUR_NONRUMOR')
    for inner_row in row_iterator:
        # Loop until the tweet_id is same and keep count of the same rows with same twet_id
        if getattr(inner_row, "TWEET_ID") == tweet_id:
            prediction = predictSentiment(getattr(inner_row, "RETWEETS"))
            retweet_predictions.append(prediction)
            # print "Retweet = ", getattr(inner_row, "RETWEETS"), "Sentiment = ", prediction
            count += 1
        # If tweet_id changes, then jump iterator count number of times
        else:
            for x in range(count):
                try:
                    row_iterator.next()
                except StopIteration:
                    count = 0
                    break
            count = 0
            break
    # Get average sentiment of all retweet predictions
    avg_sentiment = np.array(retweet_predictions).mean()
    tweetID_sentiment_dict[tweet_id] = avg_sentiment
    predicted_tweetID_veracity_dict[tweet_id] = 1 if avg_sentiment > 0.5 else 0

print "----------------------------------------"

total = len(actual_tweetID_veracity_dict)
print "Total = ", total
same = 0
different = 0
for key, value in actual_tweetID_veracity_dict.iteritems():
    if predicted_tweetID_veracity_dict[key] == value:
        same += 1
    else:
        print(key)
        different += 1

print 'Accuracy = ', (same / float(total))

with open("results.csv", 'w') as i_file:
    i_file.write('TWEET_ID,TWEET,ACTUAL,PREDICTED\n')
    for key, value in actual_tweetID_veracity_dict.iteritems():
        # i_file.write(str(key) + "," + tweetID_tweet_dict[key].encode('utf-8') + "," + str(value) + "," + str(
        #     predicted_tweetID_veracity_dict[key]) + "\n")
        i_file.write(str(key) + "," + "," + str(value) + "," + str(predicted_tweetID_veracity_dict[key]) + "\n")
