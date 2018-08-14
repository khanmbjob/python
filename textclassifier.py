# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 09:58:40 2018

@author: DELL
"""

from sklearn.feature_extraction.text import CountVectorizer
# list of text documents
text = ["The quick brown fox jumped over the lazy dog."]
# create the transform
vectorizer = CountVectorizer()
# tokenize and build vocab
vectorizer.fit(text)
# summarize
print(vectorizer.vocabulary_)
# encode document
vector = vectorizer.transform(text)
# summarize encoded vector
# =============================================================================
# print(vector.shape)
# print(type(vector))
# print(vector.toarray())
# =============================================================================
