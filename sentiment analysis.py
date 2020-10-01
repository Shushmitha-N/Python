# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# The objective is to detect hate speech in tweets.
# For the sake of simplicity, we say a tweet contains hate speech if it has a racist or sexist sentiment associated with it. 
# So, the task is to classify racist or sexist tweets from other tweets.
# ‘1’ denotes the tweet is racist/sexist and label ‘0’ denotes the tweet is not racist/sexist

import re    # for regular expressions 
import nltk  # for text manipulation 
import string 
import warnings 
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt  

pd.set_option("display.max_colwidth", 200) 
warnings.filterwarnings("ignore", category=DeprecationWarning) 


train  = pd.read_csv('train_E6oV3lV.csv') 
test = pd.read_csv('test_tweets_anuFYb8.csv')

# Data Inspection

# Non racist/sexist tweets
train[train['label'] == 0].head(10)
# Racist/sexist tweets
train[train['label'] == 1].head(10)

# Dimensions of the train and test dataset
train.shape, test.shape

# Distribution
train["label"].value_counts()
# In the train data, 2,242 (~7%) are labeled as racist/sexist, and 29,720 (~93%) are labeled as non racist/sexist. 
# So, it is an imbalanced classification challenge. A bias may exist in the model.

# distribution of length of the tweets, in terms of words, in both train and test data.
length_train = train['tweet'].str.len() 
length_test = test['tweet'].str.len() 
plt.hist(length_train, bins=20, label="train_tweets") 
plt.hist(length_test, bins=20, label="test_tweets") 
plt.legend() 
plt.show()

# Data Cleaning

# In any natural language processing task, cleaning raw text data is an important step. 
# It helps to get rid of unwanted words and characters which helps in obtaining better features. 
# If we skip this step then there is a higher chance that you are working with noisy and inconsistent data. 
# The objective of this step is to clean noise those are less relevant to find the sentiment of tweets 
# such as punctuation, special characters, numbers, and terms which don’t carry much weightage in context to the text.

# Before we begin cleaning, let’s first combine train and test datasets. 
# Combining the datasets will make it convenient for us to preprocess the data. 
# Later we will split it back into train and test data.

combi = train.append(test, ignore_index=True) 
combi.shape

# Given below is a user-defined function to remove unwanted text patterns from the tweets.

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    return input_txt   

# FOUR STEPS TO CLEAN THE TWITTER DATA

# 1. Removing Twitter Handles (@user)
combi['tidy_tweet'] = np.vectorize(remove_pattern)(combi['tweet'], "@[\w]*") 
combi.head()

# 2. Removing Punctuations, Numbers, and Special Characters
combi['tidy_tweet'] = combi['tidy_tweet'].str.replace("[^a-zA-Z#]", " ") 
combi.head(10)

# 3. Removing Short Words 
# We have to be a little careful here in selecting the length of the words which we want to remove.
# So, I have decided to remove all the words having length 3 or less. 
# For example, terms like “hmm”, “oh” are of very little use. It is better to get rid of them.
combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
combi.head()

# 4. Text Normalization
# Here we will use nltk’s PorterStemmer() function to normalize the tweets.
# Normalizing - many words may carry same meaning but only the tense will be different - to change this. 
# But before that we will have to tokenize the tweets. 
# Tokens are individual terms or words. 
# tokenization is the process of splitting a string of text into tokens.
tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split()) # tokenizing 
tokenized_tweet.head()
# Now we can normalize the tokenized tweets
from nltk.stem.porter import * 
stemmer = PorterStemmer() 
tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
# stitch these tokens back together. It can easily be done using nltk’s MosesDetokenizer function.
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])    
combi['tidy_tweet'] = tokenized_tweet

# PROBABLE QUESTIONS
# 1. What are the most common words in the entire dataset?
# 2. What are the most common words in the dataset for negative and positive tweets, respectively?
# 3. How many hashtags are there in a tweet?
# 4. Which trends are associated with my dataset?
# 5. Which trends are associated with either of the sentiments? Are they compatible with the sentiments?

# Understanding the common words used in the tweets: WordCloud
# How well the given sentiments are distributed across the train dataset. 
# One way to accomplish this task is by understanding the common words by plotting wordclouds.
# A wordcloud is a visualization wherein the most frequent words appear in large size and the less frequent words appear in smaller sizes.
all_words = ' '.join([text for text in combi['tidy_tweet']]) 
from wordcloud import WordCloud 
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words) 
plt.figure(figsize=(10, 7)) 
plt.imshow(wordcloud, interpolation="bilinear") 
plt.axis('off') 
plt.show()
# Observations - most words seem positive/neutral - because most tweets are non sexist/racist.

# Words in non racist/sexist tweets
normal_words =' '.join([text for text in combi['tidy_tweet'][combi['label'] == 0]]) 
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words) 
plt.figure(figsize=(10, 7)) 
plt.imshow(wordcloud, interpolation="bilinear") 
plt.axis('off') 
plt.show()

# Words in Racist/Sexist Tweets
negative_words = ' '.join([text for text in combi['tidy_tweet'][combi['label'] == 1]]) 
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(negative_words) 
plt.figure(figsize=(10, 7)) 
plt.imshow(wordcloud, interpolation="bilinear") 
plt.axis('off') 
plt.show()

# Understanding the impact of Hashtags on tweets sentiment
# Hashtags in twitter are synonymous with the ongoing trends on twitter at any particular point in time. 
# Try to check whether these hashtags add any value to our sentiment analysis task, 
# i.e.,if they help in distinguishing tweets into the different sentiments.
# The hashtags may also convey the same feeling as the tweet. 
# We will store all the trend terms in two separate lists — one for non-racist/sexist tweets and the other for racist/sexist tweets.

# function to collect hashtags
def hashtag_extract(x):  # Loop over the words in the tweet    
    hashtags = []        
    for i in x:        
        ht = re.findall(r"#(\w+)", i)        
        hashtags.append(ht)     
    return hashtags 
# extracting hashtags from non racist/sexist tweets     
HT_regular = hashtag_extract(combi['tidy_tweet'][combi['label'] == 0])  
# extracting hashtags from racist/sexist tweets 
HT_negative = hashtag_extract(combi['tidy_tweet'][combi['label'] == 1])  
# unnesting list 
HT_regular = sum(HT_regular,[]) 
HT_negative = sum(HT_negative,[])

# we have prepared our lists of hashtags for both the sentiments. we can plot the top ‘n’ hashtags. 
# Non racist/non sexist tweets
a = nltk.FreqDist(HT_regular) 
d = pd.DataFrame({'Hashtag': list(a.keys()),                 
                  'Count': list(a.values())}) 
# selecting top 20 most frequent hashtags    
d = d.nlargest(columns="Count", n = 20) 
plt.figure(figsize=(16,5)) 
ax = sns.barplot(data=d, x= "Hashtag", y = "Count") 
ax.set(ylabel = 'Count') 
plt.show()

# Racist/Sexist Tweets
b = nltk.FreqDist(HT_negative) 
e = pd.DataFrame({'Hashtag': list(b.keys()), 'Count': list(b.values())}) 
# selecting top 20 most frequent hashtags 
e = e.nlargest(columns="Count", n = 20)   
plt.figure(figsize=(16,5)) 
ax = sns.barplot(data=e, x= "Hashtag", y = "Count")

#To analyse a preprocessed data, it needs to be converted into features. 
# Depending upon the usage, text features can be constructed using the following techniques 
# Bag of Words, TF-IDF, and Word Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer 
import gensim

# Bag of words model
# It will create a sparse matrix containing the frequency of the words (refer machine learning A_Z notes)
# Building a bag of words (bow) vectorizer
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english') 
bow = bow_vectorizer.fit_transform(combi['tidy_tweet']) 
bow.shape

# TF-IDF
# Similar to bag of words model but it takes into account not just the occurrence of a word in a single document (or tweet) but in the entire corpus.
# TF-IDF works by penalising the common words by assigning them lower weights while giving importance to words which are rare in the entire corpus but appear in good numbers in few documents.
# Terms in TF-IDF
# TF = (Number of times term t appears in a document)/(Number of terms in the document)
# IDF = log(N/n), where, N is the number of documents and n is the number of documents a term t has appeared in.
# TF-IDF = TF*IDF
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english') 
tfidf = tfidf_vectorizer.fit_transform(combi['tidy_tweet']) 
tfidf.shape

# WORD EMBEDDING
# 1. Word2Vec Features
# Word embeddings are the modern way of representing words as vectors.
# The objective of word embeddings is to redefine the high dimensional word features 
# into low dimensional feature vectors by preserving the contextual similarity in the corpus. 
# They are able to achieve tasks like King -man +woman = Queen, which is mind-blowing.

# advantages of using word embeddings over BOW or TF-IDF are:
# 1. Dimensionality reduction - significant reduction in the no. of features required to build a model.
# 2. It capture meanings of the words, semantic relationships and the different types of contexts they are used in.

# Word2Vec is not a single algorithm but a combination of two techniques – CBOW (Continuous bag of words) and Skip-gram model. 
# Both of these are shallow neural networks (Refer notes_nlp word doc)

# We will train a Word2Vec model on our data to obtain vector representations for all the unique words present in our corpus. 
# There is one more option of using pre-trained word vectors instead of training our own model. 
# Some of the freely available pre-trained vectors are:
# 1. Google News Word Vectors
# 2. Freebase names
# 3. DBPedia vectors (wiki2vec)
# However, for this course, we will train our own word vectors since size of the pre-trained word vectors is generally huge.
# Let’s train a Word2Vec model on our corpus.
tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split()) # tokenizing 
model_w2v = gensim.models.Word2Vec(
            tokenized_tweet,
            size=200, # desired no. of features/independent variables
            window=5, # context window size
            min_count=2,
            sg = 1, # 1 for skip-gram model
            hs = 0,
            negative = 10, # for negative sampling
            workers= 2, # no.of cores
            seed = 34) 

model_w2v.train(tokenized_tweet, total_examples= len(combi['tidy_tweet']), epochs=20)

# Let’s play a bit with our Word2Vec model and see how does it perform. 
# We will specify a word and the model will pull out the most similar words from the corpus.
model_w2v.wv.most_similar(positive="dinner")
model_w2v.wv.most_similar(positive="trump")

# From the above two examples, we can see that our word2vec model does a good job of finding the most similar words for a given word. 
# But how is it able to do so? 
# That’s because it has learned vectors for every unique word in our data and it uses cosine similarity to find out the most similar vectors (words).

# Let’s check the vector representation of any word from our corpus.
model_w2v['food']
len(model_w2v['food']) #The length of the vector is 200

# Preparing Vectors for tweets.
# Our data contains tweets and not just words
# need to figure out a way to use the word vectors from word2vec model to create vector representation for an entire tweet. 
# There is a simple solution to this
# we can simply take mean of all the word vectors present in the tweet. 
# The length of the resultant vector will be the same, i.e. 200. 
# We will repeat the same process for all the tweets in our data and obtain their vectors. 
# Now we have 200 word2vec features for our data.

# We will use the below function to create a vector for each tweet by taking the average of the vectors of the words present in the tweet
def word_vector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += model_w2v[word].reshape((1, size))
            count += 1.
        except KeyError: # handling the case where the token is not in vocabulary                                     continue
            if count != 0:
                vec /= count
    return vec

# Preparing word2vec feature set
wordvec_arrays = np.zeros((len(tokenized_tweet), 200)) 
for i in range(len(tokenized_tweet)):
    wordvec_arrays[i,:] = word_vector(tokenized_tweet[i], 200)
    wordvec_df = pd.DataFrame(wordvec_arrays)
wordvec_df.shape  
# Now we have 200 new features, whereas in Bag of Words and TF-IDF we had 1000 features.
    
# 2. Doc2Vec Embedding
# Doc2Vec model is an unsupervised algorithm to generate vectors for sentence/paragraphs/documents. 
# This approach is an extension of the word2vec. 
# The major difference between the two is that doc2vec provides an additional context which is unique for every document in the corpus. 
# This additional context is nothing but another feature vector for the whole document. 
# This document vector is trained along with the word vectors.

#  load the required libraries.
from tqdm import tqdm 
tqdm.pandas(desc="progress-bar") 
from gensim.models.doc2vec import LabeledSentence

# To implement doc2vec, we have to labelise or tag each tokenised tweet with unique IDs. 
# We can do so by using Gensim’s LabeledSentence() function.
def add_label(twt):
    output = []
    for i, s in zip(twt.index, twt):
        output.append(LabeledSentence(s, ["tweet_" + str(i)]))
    return output
labeled_tweets = add_label(tokenized_tweet) # label all the tweets

# Let’s have a look at the result.
labeled_tweets[:6]

# Now let’s train a doc2vec model.
model_d2v = gensim.models.Doc2Vec(dm=1, # dm = 1 for ‘distributed memory’ model                                   
                                  dm_mean=1, # dm = 1 for using mean of the context word vectors                                  
                                  size=200, # no. of desired features                                  
                                  window=5, # width of the context window                                  
                                  negative=7, # if > 0 then negative sampling will be used                                 
                                  min_count=5, # Ignores all words with total frequency lower than 2.                                  
                                  workers=3, # no. of cores                                  
                                  alpha=0.1, # learning rate                                  
                                  seed = 23) 
model_d2v.build_vocab([i for i in tqdm(labeled_tweets)])
model_d2v.train(labeled_tweets, total_examples= len(combi['tidy_tweet']), epochs=15)

#Preparing doc2vec Feature Set
docvec_arrays = np.zeros((len(tokenized_tweet), 200)) 
for i in range(len(combi)):
    docvec_arrays[i,:] = model_d2v.docvecs[i].reshape((1,200))    

docvec_df = pd.DataFrame(docvec_arrays) 
docvec_df.shape

# We are now done with all the pre-modeling stages required  
# We will be building models on the datasets with different feature sets prepared in the earlier sections — Bag-of-Words, TF-IDF, word2vec vectors, and doc2vec vectors. 
# We will use the following algorithms to build models:
# 1. Logistic Regression
# 2. Support Vector Machine
# 3. RandomForest
# 4. XGBoost

# Evaluation Metric:
# F1 score is being used as the evaluation metric. 
# It is the weighted average of Precision and Recall. 
# Therefore, this score takes both false positives and false negatives into account. 
# It is suitable for uneven class distribution problems.

# The important components of F1 score are:
# True Positives (TP) - These are the correctly predicted positive values which means that the value of actual class is yes and the value of predicted class is also yes.
# True Negatives (TN) - These are the correctly predicted negative values which means that the value of actual class is no and value of predicted class is also no.
# False Positives (FP) – When actual class is no and predicted class is yes.
# False Negatives (FN) – When actual class is yes but predicted class in no.
# Precision = TP/TP+FP
# Recall = TP/TP+FN

# F1 Score = 2(Recall Precision) / (Recall + Precision)

# MODELLING

# 1. Logistic regression
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import f1_score

# 1a. using Bag of words features
# Extracting train and test BoW features 
train_bow = bow[:31962,:] 
test_bow = bow[31962:,:] 
# splitting data into training and validation set 
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train['label'],                                                            
                                                          random_state=42,                                                          
                                                          test_size=0.3)
lreg = LogisticRegression() 
# training the model 
lreg.fit(xtrain_bow, ytrain) 
prediction = lreg.predict_proba(xvalid_bow) # predicting on the validation set 
prediction_int = prediction[:,1] >= 0.3 # if prediction is greater than or equal to 0.3 than 1 else 0 
prediction_int = prediction_int.astype(np.int) 
f1_score(yvalid, prediction_int) # calculating f1 score for the validation set

# make predictions for the test dataset and create a submission file.
test_pred = lreg.predict_proba(test_bow) 
test_pred_int = test_pred[:,1] >= 0.3 
test_pred_int = test_pred_int.astype(np.int) 
test['label'] = test_pred_int 
submission = test[['id','label']] 
submission.to_csv('sub_lreg_bow.csv', index=False) # writing data to a CSV file

# 1b. using TF-IDF features
train_tfidf = tfidf[:31962,:] 
test_tfidf = tfidf[31962:,:] 
xtrain_tfidf = train_tfidf[ytrain.index] 
xvalid_tfidf = train_tfidf[yvalid.index]
lreg.fit(xtrain_tfidf, ytrain) 
prediction = lreg.predict_proba(xvalid_tfidf) 
prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(np.int) 
f1_score(yvalid, prediction_int) # calculating f1 score for the validation set

# 1c. Using word2vec features
train_w2v = wordvec_df.iloc[:31962,:] 
test_w2v = wordvec_df.iloc[31962:,:] 
xtrain_w2v = train_w2v.iloc[ytrain.index,:] 
xvalid_w2v = train_w2v.iloc[yvalid.index,:]
lreg.fit(xtrain_w2v, ytrain) 
prediction = lreg.predict_proba(xvalid_w2v) 
prediction_int = prediction[:,1] >= 0.3 
prediction_int = prediction_int.astype(np.int) 
f1_score(yvalid, prediction_int)

# 1d. Using doc2vec features
train_d2v = docvec_df.iloc[:31962,:] 
test_d2v = docvec_df.iloc[31962:,:] 
xtrain_d2v = train_d2v.iloc[ytrain.index,:] 
xvalid_d2v = train_d2v.iloc[yvalid.index,:]
lreg.fit(xtrain_d2v, ytrain) 
prediction = lreg.predict_proba(xvalid_d2v) 
prediction_int = prediction[:,1] >= 0.3 
prediction_int = prediction_int.astype(np.int) 
f1_score(yvalid, prediction_int)

# 2. Support Vector Machine SVM
from sklearn import svm

# 2a. Bag-of-Words Features
svc = svm.SVC(kernel='linear', C=1, probability=True).fit(xtrain_bow, ytrain) 
prediction = svc.predict_proba(xvalid_bow) 
prediction_int = prediction[:,1] >= 0.3 
prediction_int = prediction_int.astype(np.int) 
f1_score(yvalid, prediction_int)

# make predictions for the test dataset and create another submission file.
test_pred = svc.predict_proba(test_bow) 
test_pred_int = test_pred[:,1] >= 0.3 
test_pred_int = test_pred_int.astype(np.int) 
test['label'] = test_pred_int 
submission = test[['id','label']] 
submission.to_csv('sub_svm_bow.csv', index=False)

# 2b. Using TF-IDF features
svc = svm.SVC(kernel='linear', 
C=1, probability=True).fit(xtrain_tfidf, ytrain) 
prediction = svc.predict_proba(xvalid_tfidf) 
prediction_int = prediction[:,1] >= 0.3 
prediction_int = prediction_int.astype(np.int) 
f1_score(yvalid, prediction_int)

# 2c. Using Word2vec features
svc = svm.SVC(kernel='linear', C=1, probability=True).fit(xtrain_w2v, ytrain) 
prediction = svc.predict_proba(xvalid_w2v) 
prediction_int = prediction[:,1] >= 0.3 
prediction_int = prediction_int.astype(np.int) 
f1_score(yvalid, prediction_int)

# 2d. Using Doc2Vec Features
svc = svm.SVC(kernel='linear', C=1, probability=True).fit(xtrain_d2v, ytrain) 
prediction = svc.predict_proba(xvalid_d2v) 
prediction_int = prediction[:,1] >= 0.3 
prediction_int = prediction_int.astype(np.int) 
f1_score(yvalid, prediction_int)

# 3. Random Forest Classification
from sklearn.ensemble import RandomForestClassifier

# 3a. Using Bag of words features
rf = RandomForestClassifier(n_estimators=400, random_state=11).fit(xtrain_bow, ytrain) 
prediction = rf.predict(xvalid_bow) 
# validation score 
f1_score(yvalid, prediction)

#  make predictions for the test dataset and create another submission file.
test_pred = rf.predict(test_bow) 
test['label'] = test_pred 
submission = test[['id','label']] 
submission.to_csv('sub_rf_bow.csv', index=False)

# 3b. Using TF-IDF features
rf = RandomForestClassifier(n_estimators=400, random_state=11).fit(xtrain_tfidf, ytrain) 
prediction = rf.predict(xvalid_tfidf) 
f1_score(yvalid, prediction)

# 3c. Using Words2vec features
rf = RandomForestClassifier(n_estimators=400, random_state=11).fit(xtrain_w2v, ytrain) 
prediction = rf.predict(xvalid_w2v) 
f1_score(yvalid, prediction)

# 3d. Using Doc2vec features
rf = RandomForestClassifier(n_estimators=400, random_state=11).fit(xtrain_d2v, ytrain) 
prediction = rf.predict(xvalid_d2v) 
f1_score(yvalid, prediction)

# 4. XGBoost
from xgboost import XGBClassifier

# 4a. Using Bag of Words features
xgb_model = XGBClassifier(max_depth=6, n_estimators=1000).fit(xtrain_bow, ytrain) 
prediction = xgb_model.predict(xvalid_bow) 
f1_score(yvalid, prediction)

# Save the predictions in a csv file
test_pred = xgb_model.predict(test_bow) 
test['label'] = test_pred 
submission = test[['id','label']] 
submission.to_csv('sub_xgb_bow.csv', index=False)

# 4b. Using TF-IDF Features
xgb = XGBClassifier(max_depth=6, n_estimators=1000).fit(xtrain_tfidf, ytrain) 
prediction = xgb.predict(xvalid_tfidf) 
f1_score(yvalid, prediction)

# 4c. Using Word2vec features
xgb = XGBClassifier(max_depth=6, n_estimators=1000, nthread= 3).fit(xtrain_w2v, ytrain) 
prediction = xgb.predict(xvalid_w2v) 
f1_score(yvalid, prediction)

# 4d. Using Doc2vec features
xgb = XGBClassifier(max_depth=6, n_estimators=1000, nthread= 3).fit(xtrain_d2v, ytrain) 
prediction = xgb.predict(xvalid_d2v) 
f1_score(yvalid, prediction)

# PARAMETER TUNING USING XGBOOST - NOT WORKING? WHY?
# XGBoost with Word2Vec model has given us the best performance so far. 
# Let’s try to tune it further to extract as much from it as we can. 
# XGBoost has quite a many tuning parameters and sometimes it becomes tricky to properly tune them. 
import xgboost as xgb

# Here we will use DMatrices. A DMatrix can contain both the features and the target.
dtrain = xgb.DMatrix(xtrain_w2v, label=ytrain) 
dvalid = xgb.DMatrix(xvalid_w2v, label=yvalid) 
dtest = xgb.DMatrix(test_w2v)
# Parameters that we are going to tune 
params = {
    'objective':'binary:logistic',
    'max_depth':6,
    'min_child_weight': 1,
    'eta':.3,
    'subsample': 1,
    'colsample_bytree': 1
 }

# custom evaluation metric to calculate F1 score.
def custom_eval(preds, dtrain):
    labels = dtrain.get_label().astype(np.int)
    preds = (preds >= 0.3).astype(np.int)
    return [('f1_score', f1_score(labels, preds))]

# General Approach for Parameter Tuning
# We will follow the steps below to tune the parameters.
# 1. Choose a relatively high learning rate. Usually a learning rate of 0.3 is used at this stage.
# 2. Tune tree-specific parameters such as max_depth, min_child_weight, subsample, colsample_bytree keeping the learning rate fixed.
# 3. Tune the learning rate.
# 4. Finally tune gamma to avoid overfitting.

# Tuning max_depth and min_child_weight
gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in range(6,10)
     for min_child_weight in range(5,8)
 ]
max_f1 = 0. # initializing with 0 
best_params = None 
for max_depth, min_child_weight in gridsearch_params:
    print("CV with max_depth={}, min_child_weight={}".format(
                             max_depth,
                             min_child_weight))
     # Update our parameters
    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight

     # Cross-validation
    cv_results = xgb.cv(        params,
        dtrain,        feval= custom_eval,
        num_boost_round=200,
        maximize=True,
        seed=16,
        nfold=5,
        early_stopping_rounds=10
    )     
# Finding best F1 Score
    
mean_f1 = cv_results['test-f1_score-mean'].max()
    
boost_rounds = cv_results['test-f1_score-mean'].argmax()    
print("\tF1 Score {} for {} rounds".format(mean_f1, boost_rounds))    
if mean_f1 > max_f1:
        max_f1 = mean_f1
        best_params = (max_depth,min_child_weight) 

print("Best params: {}, {}, F1 Score: {}".format(best_params[0], best_params[1], max_f1))

# Updating max_depth and min_child_weight parameters.
params['max_depth'] = 9
params['min_child_weight'] = 7

# Tuning subsample and colsample
gridsearch_params = [
    (subsample, colsample)
    for subsample in [i/10. for i in range(5,10)]
    for colsample in [i/10. for i in range(5,10)] ]
max_f1 = 0. 
best_params = None 
for subsample, colsample in gridsearch_params:
    print("CV with subsample={}, colsample={}".format(
                             subsample,
                             colsample))
     # Update our parameters
    params['colsample'] = colsample
    params['subsample'] = subsample
    cv_results = xgb.cv(
        params,
        dtrain,
        feval= custom_eval,
        num_boost_round=200,
        maximize=True,
        seed=16,
        nfold=5,
        early_stopping_rounds=10
    )
     # Finding best F1 Score
    mean_f1 = cv_results['test-f1_score-mean'].max()
    boost_rounds = cv_results['test-f1_score-mean'].argmax()
    print("\tF1 Score {} for {} rounds".format(mean_f1, boost_rounds))
    if mean_f1 > max_f1:
        max_f1 = mean_f1
        best_params = (subsample, colsample) 

print("Best params: {}, {}, F1 Score: {}".format(best_params[0], best_params[1], max_f1))

# Updating subsample and colsample_bytree
params['subsample'] = .9 
params['colsample_bytree'] = .9

# Tuning the learning rate
max_f1 = 0. 
best_params = None 
for eta in [.8, .7, .6, .5, .4, .3]:
    print("CV with eta={}".format(eta))
     # Update ETA
    params['eta'] = eta

     # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        feval= custom_eval,
        num_boost_round=1000,
        maximize=True,
        seed=16,
        nfold=5,
        early_stopping_rounds=20
    )

     # Finding best F1 Score
    mean_f1 = cv_results['test-f1_score-mean'].max()
    boost_rounds = cv_results['test-f1_score-mean'].argmax()
    print("\tF1 Score {} for {} rounds".format(mean_f1, boost_rounds))
    if mean_f1 > max_f1:
        max_f1 = mean_f1
        best_params = eta 
print("Best params: {}, F1 Score: {}".format(best_params, max_f1))

# Update eta 
params['eta'] = .3

# Final parameters
params

# Finally we can now use these tuned parameters in our xgboost model. 
# We have used early stopping of 10 
# which means if the model’s performance doesn’t improve under 10 rounds, then the model training will be stopped.
xgb_model = xgb.train(
    params,
    dtrain,
    feval= custom_eval,
    num_boost_round= 1000,
    maximize=True,
    evals=[(dvalid, "Validation")],
    early_stopping_rounds=10
 )

# gnerating a file.
test_pred = xgb_model.predict(dtest) 
test['label'] = (test_pred >= 0.3).astype(np.int) 
submission = test[['id','label']] 
submission.to_csv('sub_xgb_w2v_finetuned.csv', index=False)








