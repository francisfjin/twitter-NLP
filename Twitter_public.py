#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[ ]:


from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)


# In[ ]:


get_ipython().system('pip install python-twitter')
import twitter
import en_core_web_sm
nlp = en_core_web_sm.load()


# In[ ]:


import numpy as np
import pandas as pd
import spacy
import string
import re
import csv
import time
import tweepy
import matplotlib.pyplot as plt

from gensim.models.word2vec import Word2Vec
from string import punctuation 

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import label_binarize, MultiLabelBinarizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding,LSTM
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
punctuations = string.punctuation
#nlp = spacy.load('en_core_web_md')
from nltk.tokenize import word_tokenize


# In[ ]:


APIkey = 'YOURAPIKEY'
APIsecretkey = 'YOURAPISECRETKEY'
accesstokensecret = 'ACCESSTOKENSECRET'
accesstoken = 'ACCESSTOKEN'

twitter_api = twitter.Api(consumer_key=APIkey,consumer_secret=APIsecretkey,access_token_key=accesstoken,access_token_secret=accesstokensecret)


# # Search Set

# In[ ]:


auth = tweepy.OAuthHandler(APIkey, APIsecretkey )
auth.set_access_token(accesstoken, accesstokensecret)
api = tweepy.API(auth, wait_on_rate_limit=True)

tweets = []

def querytweets(text_query, count):
#tweepy API to query tweets for desired search term
        tweets = tweepy.Cursor(api.search, q=text_query, lang='en', 
                               #geocode=coordinates,
                               ).items(count)
        tweets_list = [[tweet.created_at, tweet.text] for tweet in tweets]
        tweets_df = pd.DataFrame(tweets_list, columns=['Datetime', 'Text'])
        return tweets_df


# In[ ]:


text_query = input("Enter a search keyword: ") 

#number of desired tweets here
count = 10
searchsetdf = querytweets(text_query, count)


# In[ ]:


searchsetdf


# # Training Set

# In[ ]:


def buildTrainingSet(corpusFile, tweetDataFile):
#use corpusFile which has id keys for tweets to grab from twitter API as to not violate any developer rules
    corpus=[]
    
    with open(corpusFile,'r') as csvfile:
        lineReader = csv.reader(csvfile, delimiter=',', quotechar="\"")
        for row in lineReader:
            corpus.append({"tweet_id":row[2], "label":row[1], "topic":row[0]})

    rate_limit=180
    sleep_time=900/180
    trainingDataSet=[]
    for tweet in corpus:
        try:
            status = twitter_api.GetStatus(tweet["tweet_id"])
            print("Tweet fetched" + status.text)
            tweet["text"] = status.text
            trainingDataSet.append(tweet)
            time.sleep(sleep_time)
        except: 
            continue
    # Now we write them to the empty CSV file
    with open(tweetDataFile,'wb') as csvfile:
        linewriter=csv.writer(csvfile,delimiter=',',quotechar="\"")
        for tweet in trainingDataSet:
            try:
                linewriter.writerow([tweet["tweet_id"],tweet["text"],tweet["label"],tweet["topic"]])
            except Exception as e:
                print(e)
    return trainingDataSet


# In[ ]:


#corpusFile = "/YOURFILEPATH/corpus.csv"
#tweetDataFile = "/YOURFILEPATH/tweetDataFile.csv"
#trainingData = buildTrainingSet(corpusFile, tweetDataFile)


# In[ ]:


# Once we ran buildTrainingSet once (takes a few hours), now we have csv of all the tweets for training

trainingdata = pd.read_csv("/YOURFILEPATH/full-corpus.csv")


# In[ ]:


trainingdata


# In[ ]:


trainingdata['Sentiment'] = trainingdata['Sentiment'].map({'negative':0,'neutral':1,'positive':2,'irrelevant':4})
trainingdata['Sentiment'].value_counts()


# # Pre-Processing

# In[ ]:


def cleanup_text(docs, logging=False):
    texts = []
    counter = 1
    table = str.maketrans({key: None for key in string.punctuation})

    for doc in docs:
        if counter % 1000 == 0 and logging:
            print("Processed %d out of %d documents." % (counter, len(docs)))
        counter += 1
        doc = nlp(doc, disable=['parser', 'ner'])
        #Convert text to lowercase, strip whitespace and remove personal pronouns
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
        #Remove stopwords
        tokens = [tok.translate(table) for tok in tokens if tok not in stopwords ]
        tokens = ' '.join(tokens)
        #Remove extra whitespace
        tokens = ' '.join(tokens.split())
        # remove URLs
        tokens = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tokens)
        # remove usernames
        tokens = re.sub('@[^\s]+', 'AT_USER', tokens)
        # remove the # in #hashtag
        tokens = re.sub(r'#([^\s]+)', r'\1', tokens)
        # remove repeated characters (helloooooooo into hello)
        #tokens = word_tokenize(tokens)
        texts.append(tokens)
    return pd.Series(texts)


# In[ ]:


trainingdata = trainingdata[['TweetText','Sentiment']]


# In[ ]:


trainingdata['data_clean']= cleanup_text(trainingdata['TweetText'],logging=True)


# In[ ]:


trainingdata.isnull().sum()


# In[ ]:


trainingdata = trainingdata[trainingdata['Sentiment'] < 4]


# In[ ]:


train,test = train_test_split(trainingdata,test_size=0.2)  


# In[ ]:


test = test.sort_index()
train = train.sort_index()


# In[ ]:


searchsetdf['data_clean'] = cleanup_text(searchsetdf['Text'],logging=True)
searchsetdf


# # AutoML

# In[ ]:


automltraining = trainingdata
automltraining = automltraining[['TweetText','Sentiment']]


# In[ ]:


automltraining


# In[ ]:


automltraining['Sentiment'].value_counts()


# In[ ]:


#write to csv for automl training data, only need to run once 

#automltraining.to_csv("/content/gdrive/My Drive/Colab Notebooks/portfolio/twitter/automltraining.csv",index=None)


# In[ ]:


searchsetexport = searchsetdf['data_clean']


# In[ ]:


#export each tweet as separate txt file in folder as per AutoML batchpredict file input requirements

for key in searchsetdf.index.unique(): 
    searchsetexport = searchsetdf[searchsetdf.index == key]   
    searchsetexport['data_clean'].to_csv("/content/gdrive/My Drive/Colab Notebooks/portfolio/twitter/tweets/%s.txt" % key, header=False)  


# In[ ]:


#write a CSV of each txt file name as reference as per AutoML batchpredict file input requirements

tweetlist = []
for i in searchsetdf.index.unique():
  tweetlist.append("gs://YOURBUCKETURI/%i.txt" % i)
  df = pd.DataFrame(tweetlist)
  df.to_csv("/YOURFILEPATH/tweetlist.csv",index=None,header=None)


# # Feature Selection

# ## bagofwords 

# In[ ]:


count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train['data_clean'])
X_train_counts = X_train_counts.toarray()

X_test_counts = count_vect.transform(test['data_clean']).toarray()


# In[ ]:


searchset_counts = count_vect.fit_transform(searchsetdf['data_clean'])
searchset_counts = searchset_counts.toarray()


# ## tfidf

# In[ ]:


tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf = X_train_tfidf.toarray()

X_test_tfidf = tfidf_transformer.transform(X_test_counts).toarray()


# In[ ]:


searchset_tfidf = tfidf_transformer.fit_transform(searchset_counts)
searchset_tfidf = searchset_tfidf.toarray()

searchset_tfidf = tfidf_transformer.transform(searchset_counts).toarray()
searchset_tfidf.shape


# ## glove

# In[ ]:


train_vec_glove = []
counter = 0
for doc in nlp.pipe(train["data_clean"], batch_size=500):
    if counter % 500 == 0:
        print("Processed %d out of %d documents." % (counter, len(train["data_clean"])))
    if doc.has_vector:
        train_vec_glove.append(doc.vector)
    else:
        train_vec_glove.append(np.zeros((300,), dtype="float32"))
    counter +=1
        
train_vec_glove = np.array(train_vec_glove)

counter = 0
test_vec_glove = []
for doc in nlp.pipe(test["data_clean"], batch_size=500):
    if counter % 1000 == 0:
        print("Processed %d out of %d documents." % (counter, len(test["data_clean"])))
    if doc.has_vector:
        test_vec_glove.append(doc.vector)
    else:# If doc doesn't have a vector, then fill it with zeros.
        test_vec_glove.append(np.zeros((300,), dtype="float32"))
    counter +=1
     
test_vec_glove = np.array(test_vec_glove)


# In[ ]:


print("Train word vector shape:", train_vec_glove.shape)
print("Test word vector shape:", test_vec_glove.shape)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'search_vec_glove = []\ncounter = 0\nfor doc in nlp.pipe(searchsetdf["data_clean"], batch_size=500):\n    if counter % 500 == 0:\n        print("Processed %d out of %d documents." % (counter, len(searchsetdf["data_clean"])))\n    if doc.has_vector:\n        search_vec_glove.append(doc.vector)\n    else:\n        search_vec_glove.append(np.zeros((300,), dtype="float32"))\n    counter +=1\n    \nsearch_vec_glove = np.array(search_vec_glove)')


# In[ ]:


print("Search word vector shape:", search_vec_glove.shape)


# # Classifier
# 

# ## Keras Neural Network

# In[ ]:


# Transform labels into one hot encoded format.

y_train_one = label_binarize(train["Sentiment"], classes=train["Sentiment"].unique())


# ### Glove NN

# In[ ]:


# Define number of epochs

epochs = 100
ann_layers = 1
#Standard fully connected network

K.clear_session()

model = Sequential()
model.add(Dense(512, activation='relu', kernel_initializer='he_normal', input_dim=96))
model.add(Dropout(0.3))

for i in range(ann_layers):
    model.add(Dense(512, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.3))

model.add(Dense(3, activation='softmax'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'])

early_stop = EarlyStopping(monitor='loss', patience=3, verbose=1)

# Fit the model to the training data
ann_glove = model.fit(train_vec_glove, y_train_one, epochs=epochs,validation_split=0.1, verbose=1,
                      callbacks=[early_stop])


# In[ ]:


print(f"Training accuracy: {100*ann_glove.history['acc'][-1]}")
print(f"Validation accuracy: {100*ann_glove.history['val_acc'][-1]} for the Glove embeddings.")


# In[ ]:


#Predict class labels on search set

searchlabelsglove = np.argmax(model.predict(search_vec_glove), axis=-1)
searchsetdf['glovelabel'] = searchlabelsglove

print(f'Percentage of Negative(0), Neutral(1), and Positive(2) Tweets for keyword: {text_query} from a total of {count} tweets')
searchsetdf['glovelabel'].value_counts(normalize=True,sort=True) * 100


# In[ ]:




