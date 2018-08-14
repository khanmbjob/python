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
#Add extracolumn to store sentences and bag of word
#data_sub_filtered.columns = data_sub_filtered.columns + ['Sentences', 'BagOfWords']
#new_cols =  ['Sentences', 'BagOfWords']
#data_sub_filtered = data_sub_filtered.reindex(data_sub_filtered.columns.union(new_cols), axis=1)


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

#Spelling correction
#https://www.analyticsvidhya.com/blog/2018/02/the-different-methods-deal-text-data-predictive-python/
data_sub_filtered['REVIEW_REMARKS'].apply(lambda x: str(TextBlob(x).correct()));

# =============================================================================
# #remove all URLs in the dataset
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+';
for row in data_sub_filtered.index :
   data_sub_filtered.loc[row, 'REVIEW_REMARKS'] = re.sub(url_regex, '', data_sub_filtered.loc[row, 'REVIEW_REMARKS']).lower();
# =============================================================================

#cleanup the words specific the Review Remarks. Like "- Risk", "-Risk", "-risk", "- risk", "- High Risk", "- Low risk" etc..
pattern_to_find = "\bhigh risk\b|- risk|\blow risk\b|\bHigh Risk\b"; 
pattern_to_repl = "";
for row in data_sub_filtered.index:
   data_sub_filtered.loc[row, 'REVIEW_REMARKS'] = re.sub(pattern_to_find, pattern_to_repl, data_sub_filtered.loc[row, 'REVIEW_REMARKS']);    

#remove all symbols such as « ! % & * @ # », and fix inconsistent-casings.
pattern_to_find = "[^a-zA-Z0-9.' ]";
pattern_to_repl = "";
for row in data_sub_filtered.index :
   data_sub_filtered.loc[row, 'REVIEW_REMARKS'] = re.sub(pattern_to_find, pattern_to_repl, data_sub_filtered.loc[row, 'REVIEW_REMARKS']);    

#Insert a newline for each point in a Review Remarks
pattern_to_find = "\d+."; 
pattern_to_repl = "\n";
for row in data_sub_filtered.index :
   data_sub_filtered.loc[row, 'REVIEW_REMARKS'] = re.sub(pattern_to_find, pattern_to_repl, data_sub_filtered.loc[row, 'REVIEW_REMARKS'].strip());    

##Tokenization
##Tokenization refers to dividing the text into a sequence of words or sentences.
##python -m textblob.download_corpora
#for row in data_sub_filtered.index :
#   mytext =  data_sub_filtered.loc[row, 'REVIEW_REMARKS'] ;
#   data_sub_filtered.loc[row, 'Sentences'] =  str(TextBlob(mytext).sentences) + "\n";
#   data_sub_filtered.loc[row, 'BagOfWords'] =  str(TextBlob(mytext).words) + "\n";
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
#for row in data_sub_filtered.index :
#    data_sub_filtered.loc[row, 'REVIEW_REMARKS'] = gensim.summarization.summarize( data_sub_filtered.loc[row, 'REVIEW_REMARKS'].lower(), ratio = 0.80);
#Summarization
#https://pypi.org/project/PyTLDR/
# =============================================================================
# Using the TextRank algorithm (based on PageRank)
# Using Latent Semantic Analysis
# Using a sentence relevance score
# =============================================================================

ExcelFileWriter = pd.ExcelWriter('TextPreProcessing_Stage1.xlsx', engine='xlsxwriter')
data_sub_filtered.to_excel(ExcelFileWriter,sheet_name='Risky Comments By Project' )
