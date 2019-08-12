import json
import sys

import gensim
import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec
from keras.layers import Conv1D, Flatten, MaxPool1D
from keras.layers import Dense
from keras.models import Sequential
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from tqdm import tqdm

pd.options.mode.chained_assignment = None
LabeledSentence = gensim.models.doc2vec.LabeledSentence
tqdm.pandas(desc="progress-bar")
tokenizer = TweetTokenizer()


def ingest():
    data = pd.read_csv('training_tweets.csv', encoding='latin-1')
    data.drop(['ItemID', 'SentimentSource', 'Date',
               'Blank'], axis=1, inplace=True)
    data = data[data.Sentiment.isnull() == False]
    data['Sentiment'] = data['Sentiment'].map(
        {4: 1, 0: 0})  # Convert 4 -> 1 and 0 -> 0
    data = data[data['SentimentText'].isnull() == False]
    data.reset_index(inplace=True)
    data.drop('index', axis=1, inplace=True)
    print 'Dataset loaded with shape', data.shape
    return data


dataset = ingest()


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


def postprocess(data, no=1600000):
    data = data.head(no)
    data['tokens'] = data['SentimentText'].progress_map(tokenize)
    data = data[data.tokens != 'NC']
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    return data


dataset = postprocess(dataset)

n = 1600000
n_dim = 200

x_train, x_test, y_train, y_test = train_test_split(
    np.array(dataset.head(n).tokens), np.array(dataset.head(n).Sentiment), test_size=0.2)


def labelizeTweets(tweets, label_type):
    labelized = []
    for i, v in tqdm(enumerate(tweets)):
        label = '%s_%s' % (label_type, i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized


x_train = labelizeTweets(x_train, 'TRAIN')
x_test = labelizeTweets(x_test, 'TEST')

tweet_w2v = Word2Vec(size=n_dim, min_count=10)
tweet_w2v.build_vocab([x.words for x in tqdm(x_train)])
tweet_w2v.train([x.words for x in tqdm(x_train)],
                total_examples=tweet_w2v.corpus_count, epochs=tweet_w2v.iter)

# Save the word2vec model
tweet_w2v.save('W2V Model.txt')

print 'Building tf-idf matrix ...'
vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
matrix = vectorizer.fit_transform([x.words for x in x_train])
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
print 'vocab size :', len(tfidf)

# Save the idfs for future use
np.set_printoptions(threshold=sys.maxsize)
print repr(vectorizer.idf_)

# Save the tfidf vocabulary for future use
json.dump(vectorizer.vocabulary_, open('vocabulary.json', mode='wb'))


def buildWordVector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            # Add the embedded word2vec of each word in the tweet
            vec += tweet_w2v[word].reshape((1, size))  # * tfidf[word]
            # Number pf words in count
            count += 1.
        except KeyError:
            continue
    # Get average tweet vector
    if count != 0:
        vec /= count
    return vec


# Convert tweet texts into vectors
train_vecs_w2v = np.concatenate(
    [buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, x_train))])
train_vecs_w2v = scale(train_vecs_w2v)
train_vecs_w2v = train_vecs_w2v[..., np.newaxis]

test_vecs_w2v = np.concatenate(
    [buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, x_test))])
test_vecs_w2v = scale(test_vecs_w2v)
test_vecs_w2v = test_vecs_w2v[..., np.newaxis]

# Model Initialization
model = Sequential()
model.add(Conv1D(filters=50, kernel_size=2, input_shape=(200, 1)))
model.add(MaxPool1D(pool_size=1))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='mae',
              metrics=['accuracy'])

# Fit the model
model.fit(train_vecs_w2v, y_train, epochs=1, batch_size=32, verbose=2)

# Evaluate the model and get the accuracy
accuracy = model.evaluate(test_vecs_w2v, y_test, batch_size=128, verbose=2)
print "Accuracy of the model = ", accuracy

# Predict the test vectors
prediction = model.predict(test_vecs_w2v)

# Store the results in the file "Predicted Data.csv"
dataframe = pd.DataFrame(x_test, columns=['SentimentText', 'Label'])


# To remove 'u'
def convertToString(arr):
    return [r.encode("utf-8") for r in arr]


dataframe['SentimentText'] = dataframe['SentimentText'].apply(convertToString)
dataframe = dataframe.assign(Sentiment=y_test)
dataframe['predictions'] = prediction
dataframe.to_csv('Predicted Data.csv')

# Save the model
model.save('Sentiment Analysis Model.h5')
