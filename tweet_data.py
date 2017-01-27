import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
from sklearn.cluster import KMeans
from nltk.stem.snowball import EnglishStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

def tweet_filter(all_tweets):
    bad_idxs = all_tweets[all_tweets['text'].str.contains('"@')].index
    print bad_idxs
    all_tweets.drop(all_tweets.index[[bad_idxs]], inplace=True)
    print all_tweets.head()
    print all_tweets.shape
    return all_tweets

def compute_tfidf(articles, max_features=2000):
    articles = [clean_document(article) for article in articles]
    tfidf_model = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = tfidf_model.fit_transform(articles).todense()
    return (tfidf_model, tfidf_matrix)

def clean_document(document):
    # document = document.encode('utf-8', 'ignore')
    document = document.split()
    document = [word.lower() for word in document]
    document = [filter(lambda c: c in string.letters, word) for word in document]
    document = filter(lambda w: w not in ENGLISH_STOP_WORDS,document)
    es = EnglishStemmer()
    document = [es.stem(word) for word in document]
    document = filter(lambda w: w != '',document)
    document = string.join(document, ' ')
    # print type(document)
    # print document[0]
    return document

def kmeans_tfidf(tfidf_matrix, k=10):
    kmeans_model = KMeans(n_clusters=k)
    kmeans_model.fit(tfidf_matrix)
    return kmeans_model

def invert_vocabulary(vocabulary):
    inv = {}
    for k,v in vocabulary.iteritems():
        inv[v] = k
    return inv

def print_important_words(clusters, inv):
    for i,cluster in enumerate(clusters):
        print 'CLUSTER {}'.format(i)
        top_word_inds = (-cluster).argsort()[:10]
        top_words = [inv[word] for word in top_word_inds]
        for word in top_words:
            print word
        print ''

def print_title_sample(kmeans_model, df):
    for i in np.unique(kmeans_model.labels_):
        art_idxs = kmeans_model.labels_ == i
        cluster_arts = np.where(art_idxs)[0]
        size = cluster_arts.size
        inds_print = np.random.choice(cluster_arts, 10, replace=False)
        print 'CLUSTER {0}, SIZE {1}'.format(i, size)
        print df.loc[inds_print, 'text']
        print ''

if __name__ == '__main__':
    plt.style.use('fivethirtyeight')
    k = 20
    all_tweets = pd.read_csv('umbrella_data.csv')
    tweets = tweet_filter(all_tweets)
    articles = tweets['text']
    # tfidf_model, tfidf_matrix = compute_tfidf(articles)
    # kmeans_model = kmeans_tfidf(tfidf_matrix, k=k)
    # clusters = kmeans_model.cluster_centers_
    # inv = invert_vocabulary(tfidf_model.vocabulary_)
    # print_important_words(clusters, inv)
    # print_title_sample(kmeans_model, tweets)
