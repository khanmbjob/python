# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 00:09:47 2018

@author: Mohammed Barkath 
Program Name: My First Machine learning Program
"""

#Decision tree model
from sklearn import tree
features = [[140, 1, 1], [130, 1, 0],[150, 0, 0], [170, 0, 1]]
lables = [0, 0, 1, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, lables)
print(clf.predict([[160,0,1]]))

#Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
features = [[140, 1, 1], [130, 1, 0],[150, 0, 0], [170, 0, 1]]
lables = [0, 0, 1, 1]
clf = GaussianNB()
clf = clf.fit(features, lables)
print(clf.predict([[160,0,1,1]]))
