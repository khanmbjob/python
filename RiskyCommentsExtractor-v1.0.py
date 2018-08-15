# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 17:05:23 2018
@author: Mohammed Barkath
"""
# -*- coding: utf-8 -*-
"""
Description: Risky Comments Extractor Based on Risky category bag of words.
"""
import numpy as np
import os
import pandas as pd
import re
from textblob import TextBlob;
from textblob import Word;


#Clear Memory & remove global object:
clear = lambda: os.system('cls')
clear()
        
path = "C:\\Users\\DELL\\Desktop\\Python"
os.chdir(path)        
        
#print (os.getcwd())
#https://www.datacamp.com/community/tutorials/python-excel-tutorial
# Assign spreadsheet filename to `file`
RcBOWfile = 'WP_DE_SoW Review Tracker-2018 V0.21 Python.xlsx'
Sowfile = 'TextPreProcessing_Stage1.xlsx'
# Load spreadsheet
SoWTracker = pd.ExcelFile(Sowfile)
RcBowTracker =pd.ExcelFile(RcBOWfile)
RRfile = open("SoWRiskyRemarksNew.txt", "w", encoding="utf-8")
 
#https://medium.com/ml2vec/cleaning-data-for-machine-learning-ca476ac5ae4e
#Must do Cleaning Data for Machine Learning
#call the function Data cleaner


# Print the sheet names
#print(xl.sheet_names)
# Load a sheet into a DataFrame by name: df_MasterData , df_Category
#df_MasterData = SoWTracker.parse(sheet_name='Risky Comments By Project', skiprows=0)
df_ReviewRemark = SoWTracker.parse(sheet_name='Risky Comments By Project', skiprows=0)
#df_Category = RcBowTracker.parse('Risk Category BOW')

df_Risk_Cat = RcBowTracker.parse('Risk Category BOW')
#Cleanup special charenters from the BOW 
pattern_to_find = "[^a-zA-Z0-9.', ]";
pattern_to_repl = "";
for row in df_Risk_Cat.index:
   df_Risk_Cat.loc[row, 'Bag of words'] = re.sub(pattern_to_find, pattern_to_repl, df_Risk_Cat.loc[row, 'Bag of words']).lower();    
           
#Extracting specific columns of a pandas dataframe
#df_ReviewRemark = df_MasterData[["Unique ID","REVIEW_REMARKS"]]
strLine = "Unique ID" + "|" + "" + "|" + "Risk Remarks" + "\n"
#LstLine = [['Unique ID','Risk Category', 'Risk Remarks']] 
LstLine =[]

RRfile.write(strLine)
#for indexRR, rowRR in df_ReviewRemark.iterrows() :
for rowRR in df_ReviewRemark.index :    
   if type(df_ReviewRemark.loc[rowRR,'REVIEW_REMARKS']) != float :
        #LstRR = list(map(str.strip,rowRR["REVIEW_REMARKS"].split("\n")))
        LstRR = list(map(str.strip, df_ReviewRemark.loc[rowRR, 'REVIEW_REMARKS'].split("\n")))        
        # print("Remarks by each Line:",rowRR["Unique ID"],"=",LstRR)
        #Extracting specific rows of a pandas dataframe
        #dfCatRow = df_Category.loc[0,"Bag of words"]
        #Extracting specific columns of a pandas dataframe
        #df_Risk_Cat = df_Category[["Risk Category","Bag of words"]]

        #for index, row in df_Risk_Cat.iterrows() :
        for row in df_Risk_Cat.index :
            # Convert LstBow to  _sre.SRE_Pattern for pandas pattern search
                for strRRLine in LstRR :
                    #if type(row["Bag of words"]) != float :
                    if type(df_Risk_Cat.loc[row,'Bag of words']) != float :                    
                        #LstBow = list(map(str.strip,row["Bag of words"].split(",")))
                        LstBow = list(map(str.strip, df_Risk_Cat.loc[row, 'Bag of words'].split(",")))
                    for StrBow in LstBow :
                       #print(StrBow)
                       if re.search(StrBow, strRRLine) :
                           #strLine = str(rowRR["Unique ID"])  + "|" + str(row["Risk Category"]) + "|" + str(strRRLine)+"\n"
                           strLine = str(df_ReviewRemark.loc[rowRR,'Unique ID'])  + "|" + str(df_Risk_Cat.loc[row,'Risk Category']) + "|" + str(strRRLine)+"\n"                           
                           RRfile.write(strLine)
                           LstLine.append(list(map(str.strip, strLine.split("|"))))
                       break
RRfile.close()  

#Load List into the data frame
DfRRfile = pd.DataFrame(LstLine, columns = ['Unique ID','Risk Category', 'Risk Remarks'])

#Add BOW & Sentence columns
#DfRRfile.columns = DfRRfile.columns + ['Sentences', 'BagOfWords']
new_cols =  ['Sentences', 'BagOfWords']
DfRRfile = DfRRfile.reindex(DfRRfile.columns.union(new_cols), axis=1)


# Tokenation & remove non english owrds from bag of words
# =============================================================================
# remove non english owrds from bag of words
# You can use the words corpus from NLTK:
# import nltk
# words = set(nltk.corpus.words.words())
# # sent = "Io andiamo to the beach with my amico."
# " ".join(w for w in nltk.wordpunct_tokenize(sent) \
#          if w.lower() in words or not w.isalpha())
# =============================================================================

import nltk
words = set(nltk.corpus.words.words())
for row in DfRRfile.index :
   mytext =  DfRRfile.loc[row, 'Risk Remarks'] ;
   DfRRfile.loc[row, 'Sentences'] =  str(TextBlob(mytext).sentences) + "\n";
#   DfRRfile.loc[row, 'BagOfWords'] =  str(TextBlob(mytext).words) + "\n";
   DfRRfile.loc[row, 'BagOfWords'] =  " ".join(w for w in nltk.wordpunct_tokenize(mytext) \
                                                   if w.lower() in words or not w.isalpha())
#   DfRRfile.loc[row, 'BagOfWords'] =  " ".join(w for w in str(TextBlob(mytext).words) if w.lower() in words or not w.isalpha())
# Stopwords removal
#stop words (or commonly occurring words) should be removed from the text data.   
from nltk.corpus import stopwords
stop = stopwords.words('english')
DfRRfile['BagOfWords'] = DfRRfile['BagOfWords'].apply(lambda x: " ".join(x for x in x.split() if x not in stop));
DfRRfile['Sentences'] = DfRRfile['Sentences'].apply(lambda x: " ".join(x for x in x.split() if x not in stop));

# Frequent words removal / Common word removal 
#freq = pd.Series(' '.join(DfRRfile['BagOfWords']).split()).value_counts()[:3]
#freq = list(freq.index)
#DfRRfile['BagOfWords'] =  DfRRfile['BagOfWords'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
#DfRRfile['Sentences'] = DfRRfile['Sentences'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

#Rare words removal
freq = pd.Series(' '.join(DfRRfile['BagOfWords']).split()).value_counts()[-3:]
freq = list(freq.index)
DfRRfile['BagOfWords'] =  DfRRfile['BagOfWords'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
DfRRfile['Sentences'] = DfRRfile['Sentences'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

#######################################################################################
#BELOW CODE IS REPLACED WITH MACHINE LEARNING ALGORTHIM - TfidfVectorizer
#lemmatization
##Tokenization refers to dividing the text into a sequence of words or sentences.
#DfRRfile['BagOfWords'] = DfRRfile['BagOfWords'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
#DfRRfile['Sentences'] = DfRRfile['Sentences'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

#Add bag of word 
#for row in DfRRfile.index :
#   mytext =  DfRRfile.loc[row, 'BagOfWords'] ;
#   DfRRfile.loc[row, 'BagOfWords'] =  str(TextBlob(mytext).words) + " ";
#######################################################################################
import matplotlib.pyplot as plt;
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
#######################################################################################
# Machine learning Code 
#######################################################################################
# select Risk Remarks and Risk Category column from data 
df = DfRRfile

# assign numbers to Risk Category[0,1,2]
df['category_id'] = df['Risk Category'].factorize()[0]
category_id_df = df[['Risk Category', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id','Risk Category']].values)
df.head

# check the doucment occurance of each other by ploting the count of document with respect to authors provided.
#fig = plt.figure(figsize = (8,6))
#DfRRfile.groupby('Risk Category').count().plot.bar(ylim=0)
#plt.show()

#######################################################################################
#TfidfVectorizer - Naive Bayes classifier for multinomial models
#######################################################################################

# tfidf vectorizer is calculated removing the stopwards and ngram values (1,2) which is unigram to bigram range combiniation.
# normaized to ridge regression using l2 
# Features genation using Term_frequency inverse document frequency calculation

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(DfRRfile['BagOfWords']).toarray()
labels = DfRRfile['Risk Category']
features.shape
# features generated after performing Tf_idf which is the occurance of each word 
#features

##Add a column for predicted Category by ML
new_cols =  ['PredictedCategory1', 'PredictedCategory2']
DfRRfile = DfRRfile.reindex(DfRRfile.columns.union(new_cols), axis=1)



N = 2
for risk_category, category_id in sorted(category_to_id.items()) :
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
#  print(format(risk_category))  
#  print("Most correlated unigrams:{}".format('\n'.join(unigrams[-N:])))
#  print("Most correlated bigrams:{}".format('\n'.join(bigrams[-N:])))
  
X_train, X_test, y_train, y_test = train_test_split(df['Risk Remarks'], df['Risk Category'], test_size=0.33, random_state = 0)

#CountVectorizer
#Tokenize the documents and count the occurrences of token and return them as a sparse matrix
count_vect = CountVectorizer()

#Learn vocabulary and idf, return term-document matrix.
X_train_counts = count_vect.fit_transform(X_train)

#Apply Term Frequency Inverse Document Frequency normalization to a sparse matrix of occurrence counts.
tfidf_transformer = TfidfTransformer()

#Transform a count matrix to a normalized tf or tf-idf representation
#Tf means term-frequency while tf-idf means term-frequency times inverse document-frequency. 
#This is a common term weighting scheme in information retrieval, 
#that has also found good use in document classification.
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#Fit Naive Bayes classifier according to X, y
clf = MultinomialNB().fit(X_train_tfidf, y_train)

for row in DfRRfile.index :
    feature =  DfRRfile.loc[row, 'BagOfWords'] ;
    #Convert to list containing if it contains single element to avoid error
    feature = [feature]; 
    X_new_counts = count_vect.transform(feature);
    X_new_tfidf = tfidf_transformer.transform(X_new_counts);
    #Perform classification on an array of test vectors X.
    predicted = clf.predict(X_new_tfidf);
    DfRRfile.loc[row,'PredictedCategory1'] = predicted;

#Another Model
#model = MultinomialNB()
#X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.33, random_state=0)
#model.fit(X_train, y_train)
#y_pred = model.predict(X_test)
#
#for row in DfRRfile.index :
#    feature =  DfRRfile.loc[row, 'BagOfWords'] ;
#    #Convert to list containing if it contains single element to avoid error
#    feature = [feature]; 
#    predicted = model.predict(feature);
#    DfRRfile.loc[row,'PredictedCategory2'] = predicted;
#   
"""


#######################################################################################
# Check the doucment occurance of each other by ploting the count of document with respect to authors provided.
#######################################################################################
fig = plt.figure(figsize = (8,6))
DfRRfile.groupby('Risk Category').count().plot.bar(ylim=0)
plt.show()
#######################################################################################
# Commonly used unigram and bigram of each author
#######################################################################################

#Heat map to see confusion matrix / prediction accuracy
from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df['Risk Category'].values, yticklabels=category_id_df['Risk Category'].values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Metric for Multinomial Nieve Bayes Classifer \\n')
plt.show()

"""
#Write into Excel File
ExcelFileWriter = pd.ExcelWriter('SoWRiskyRemarksNew.xlsx', engine='xlsxwriter')
DfRRfile.to_excel(ExcelFileWriter,sheet_name='Risky Comments By Project', index = False )

#Need some datacleansing using AI
#https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/
#https://www.udemy.com/the-complete-python-course-for-machine-learning-engineers/?couponCode=JAN92018
#https://machinelearningmastery.com/data-cleaning-turn-messy-data-into-tidy-data/
