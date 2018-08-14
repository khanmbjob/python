# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 20:37:14 2018

@author: DELL
"""

#load the iris dataset as an example
import pandas as pd
import os
print(os.getcwd())
os.chdir('C:\\Users\\DELL\\Desktop\\Python')
# store the feature matrix (x) and response vector (y)

#simple_train = ['call me tonight', 'Call me a cab', 'Please call me.. PLEASE!']
#with open('MasterDatav1.csv', encoding="utf8", errors='ignore') as f:
#    MasterData = f.read()
    
MasterData = pd.read_excel('MasterData.xlsx')
#read_csv('MasterDatav',)



simple_train = MasterData['Comments']
    


from sklearn.feature_extraction.text import CountVectorizer 
vect = CountVectorizer()

#Learn the "Vocabulary" of training data
vect.fit(simple_train)

#Exampine the fitted vocabulary
vect.get_feature_names()

print(vect.get_feature_names())

#Transform training data into Document-Term matrix
simple_train_dtm = vect.transform(simple_train)
simple_train_dtm

#Convert sparse matrix to dense matrix
simple_train_dtm.toarray()

#Examine the Vocabulary and document-term matrix
pd.DataFrame(simple_train_dtm.toarray(), columns=vect.get_feature_names())
print(pd.DataFrame(simple_train_dtm.toarray(), columns=vect.get_feature_names()))
#check the type of document-term matrix
type(simple_train_dtm)
print(simple_train_dtm)

#example text for model testig
simple_test = ["Where is  the holiday support for this project"]

#transform simple_test data in to a document term-matrix
simple_test_dtm = vect.transform(simple_test)
simple_test_dtm.toarray()
pd.DataFrame(simple_test_dtm.toarray(),columns=vect.get_feature_names())

type(simple_test_dtm)
print(simple_test_dtm)
#Bag of Words
bow = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(1,1),analyzer = "word")
train_bow = bow.fit_transform(simple_train)
print(type(train_bow))