# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 18:37:35 2018

@author: Mohammed Barkath
"""
# -*- coding: utf-8 -*-
"""
Description: Data cleaner.
https://medium.com/ml2vec/cleaning-data-for-machine-learning-ca476ac5ae4e
Regex cheatsheet - https://www.debuggex.com/cheatsheet/regex/python
https://medium.com/ml2vec/cleaning-data-for-machine-learning-ca476ac5ae4e
"""
#Loading the Training and Test data
import pandas as pd;
#import numpy as np;
import re;
import os;
#import json;
import gensim;
from textblob import TextBlob;
#import time;
#from nltk.corpus import stopwords;
#import sys;
#import pickle;

#import matplotlib.pyplot as plt;
#import seaborn as sns;
#from sklearn import preprocessing;
#from IPython.display import clear_output;
#set the option for max_colwidth to better visualize dataframes
pd.set_option('display.max_colwidth', -1)

#Function for identifying contractions and return mapped word
# Expand all contractions. 
#This is an important step, because it will reduce disambiguation 
#between similar phrases such as « I’ll » and « I will ». 
#contractions = json.load(open('contractions.json', 'r'));
# =============================================================================
# from pycontractions import Contractions
# # Load your favorite word2vec model
# cont = Contractions('GoogleNews-vectors-negative300.bin')
# #optional, prevents loading on first expand_texts call
# cont.load_models()
# =============================================================================

#Download the data, load it into a Pandas Dataframe, and set the columns:
data_location = "C:\\Users\\DELL\\Desktop\\Python";
os.chdir(data_location);
data_train = pd.read_excel('WP_DE_SoW Review Tracker-2018 V0.21 Python.xlsx');
data_train.columns = ['Unique ID', 'REVIEW_REMARKS'];
#data_test.columns = ['Unique_ID', 'REVIEW_REMARKS'];
# =============================================================================
# print(len(data_train));
# 
# #use a smaller subset of rows to verify that the pipeline works without errors
# np.random.seed(1024);
# #We'll use ~20% of the dataset for now
# total_samples = int(0.30 * len(data_train))
# rand_indices = np.random.choice(len(data_train), total_samples, replace=False);
# 
# train_split_index = int(0.75 * total_samples);
# dev_split_index   = int(0.95 * total_samples);
# data_sample_train = data_train.iloc[rand_indices[:train_split_index]]; 
# 
# data_sample_dev   = data_train.iloc[rand_indices[train_split_index:dev_split_index]];
# data_sample_test  = data_train.iloc[rand_indices[dev_split_index:]];
# 
# sample_ratio = len(data_train) / len(data_sample_train);
# print("Amount of data being trained on: " + str(100.0 / sample_ratio) + '%')
# 
# =============================================================================
#Processing Data Labels
#Let’s use the first 100 rows in this case.
#data_subsample = data_sample_train.iloc[0:100];

#Step1 : Remove all NaNs
#data_sub_filtered = data_subsample.dropna();
data_sub_filtered = data_train.dropna(); 

# =============================================================================
#https://www.analyticsvidhya.com/blog/2018/02/the-different-methods-deal-text-data-predictive-python/
# Text Pre-processing of text data
# Lower casing
# Punctuation removal
# Stopwords removal
# Frequent words removal
# Rare words removal
# Spelling correction
# Tokenization
# Stemming
# Lemmatization
# =============================================================================

# =============================================================================
# #change all lines to lower case 
for row in data_sub_filtered.index :
   data_sub_filtered.loc[row, 'REVIEW_REMARKS'] = data_sub_filtered.loc[row, 'REVIEW_REMARKS'];
# =============================================================================

#Punctuation removal
for row in data_sub_filtered.index :
   data_sub_filtered.loc[row, 'REVIEW_REMARKS'] =  data_sub_filtered.loc[row, 'REVIEW_REMARKS'].str.replace('[^\w\s]','');

# Stopwords removal
from nltk.corpus import stopwords
stop = stopwords.words('english')
for row in data_sub_filtered.index :
   data_sub_filtered.loc[row, 'REVIEW_REMARKS'] =  data_sub_filtered.loc[row, 'REVIEW_REMARKS'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

# Frequent words removal / Common word removal
freq = pd.Series(' '.join(data_sub_filtered.loc[row, 'REVIEW_REMARKS']).split()).value_counts()[:10]
freq = list(freq.index)
for row in data_sub_filtered.index :
   data_sub_filtered.loc[row, 'REVIEW_REMARKS'] =  data_sub_filtered.loc[row, 'REVIEW_REMARKS'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

#Rare words removal
   
#Spelling correction
#https://www.analyticsvidhya.com/blog/2018/02/the-different-methods-deal-text-data-predictive-python/
data_sub_filtered['REVIEW_REMARKS'].apply(lambda x: str(TextBlob(x).correct()));


#Now we can iterate over each row, expanding contractions wherever they appear.
# =============================================================================
# for row in data_sub_filtered.index:
#     data_sub_filtered.loc[row, 'REVIEW_REMARKS'] =  ' '.join(([cont.expand_texts(word) for word in data_sub_filtered.loc[row,  'REVIEW_REMARKS'].split()]));
# 
# =============================================================================
   
#Summariation
   #https://radimrehurek.com/gensim/summarization/summariser.html
   #gensim.summarization.summarizer.summarize(text, ratio=0.2, word_count=None, split=False)
   #https://towardsdatascience.com/text-summarization-extractive-approach-567fe4b85c23
#change all lines to lower case 
for row in data_sub_filtered.index :
    data_sub_filtered.loc[row, 'REVIEW_REMARKS'] = gensim.summarization.summarize( data_sub_filtered.loc[row, 'REVIEW_REMARKS'].lower(), ratio = 0.80);
#Summarization
#https://pypi.org/project/PyTLDR/
# =============================================================================
# Using the TextRank algorithm (based on PageRank)
# Using Latent Semantic Analysis
# Using a sentence relevance score
# =============================================================================

ExcelFileWriter = pd.ExcelWriter('SoWReviewTrackerCleaned.xlsx', engine='xlsxwriter')
data_sub_filtered.to_excel(ExcelFileWriter,sheet_name='Risky Comments By Project' )
