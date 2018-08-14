# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 19:41:21 2018

@author: DELL
"""
#https://www.analyticsvidhya.com/blog/2017/09/naive-bayes-explained/
# =============================================================================
# How to build a basic model using Naive Bayes in Python?
# Again, scikit learn (python library) will help here to build a Naive Bayes model in Python. There are three types of Naive Bayes model under scikit learn library:
# 
# Gaussian: It is used in classification and it assumes that features follow a normal distribution.
# 
# Multinomial: It is used for discrete counts. For example, let’s say,  we have a text classification problem. Here we can consider bernoulli trials which is one step further and instead of “word occurring in the document”, we have “count how often word occurs in the document”, you can think of it as “number of times outcome number x_i is observed over the n trials”.
# 
# Bernoulli: The binomial model is useful if your feature vectors are binary (i.e. zeros and ones). One application would be text classification with ‘bag of words’ model where the 1s & 0s are “word occurs in the document” and “word does not occur in the document” respectively.
# 
# Based on your data set, you can choose any of above discussed model. Below is the example of Gaussian model.
# 
# Python Code
# =============================================================================
#Import Library of Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
import numpy as np

#assigning predictor and target variables
x = np.array([[-3,7],[1,5], [1,2], [-2,0], [2,3], [-4,0], [-1,1], [1,1], [-2,2], [2,7], [-4,1], [-2,7]])
Y = np.array([3, 3, 3, 3, 4, 3, 3, 4, 3, 4, 4, 4])
#Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets 
model.fit(x, Y)

#Predict Output 
predicted= model.predict([[1,2],[3,4]])
print(predicted)
