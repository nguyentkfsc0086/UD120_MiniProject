#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
features_train = features_train[:int(len(features_train)*0.01)]
labels_train = labels_train[:int(len(labels_train)*0.01)]

#########################################################
### your code goes here ###

#deffine a classification variable, assigned paremeter kernel = "linear"
clf = svm.SVC(C= 10000,kernel="rbf")
#train the date
clf.fit(features_train,labels_train)
#predict the value
pred = clf.predict(features_test)
#score the model
acc = accuracy_score(pred,labels_test)
christ = 0
for i in pred:
    if i == 1:
        christ += 1

print(christ)


#########################################################

#########################################################
'''
You'll be Provided similar code in the Quiz
But the Code provided in Quiz has an Indexing issue
The Code Below solves that issue, So use this one
'''

# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]

#########################################################
