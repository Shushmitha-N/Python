# -*- coding: utf-8 -*-

import nltk  
import numpy as np  
import random  
import string 
import pandas as pd
import re
import heapq

dataset = pd.read_csv('Reviews_formatted.csv', delimiter = ',')
messages= dataset.values.tolist()
para=""
for i in range(0,len(messages)):
    para=para+str(messages[i])
    
corpus = nltk.sent_tokenize(para)

for i in range(len(corpus )):
    corpus [i] = corpus [i].lower()
    corpus [i] = re.sub(r'\W',' ',corpus [i])
    corpus [i] = re.sub(r'\s+',' ',corpus [i])
    
wordfreq = {}
for sentence in corpus:
    tokens = nltk.word_tokenize(sentence)
    for token in tokens:
        if token not in wordfreq.keys():
            wordfreq[token] = 1
        else:
            wordfreq[token] += 1
            
most_freq = heapq.nlargest(200, wordfreq, key=wordfreq.get)

df = pd.DataFrame(data=wordfreq, index=[0])

df = (df.T)

print (df)

df.to_excel('dict1.xlsx')