#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
import sklearn.naive_bayes 
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


##############################################################
# Enter Your Code Here
clf = sklearn.naive_bayes.GaussianNB()
#training
t0 = time()
fit = clf.fit(features_train,labels_train)
#record the time of training
print("The time of training is: ", round(time() - t0, 3), "s")
#predicting
t1 = time()
pred = fit.predict(features_test)
print("The time of predicting is: ", round(time() - t1, 3), "s")
#calculate the accuracy
print("The accuracy of the training process is: " + clf.score(pred,labels_test))




##############################################################

##############################################################
'''
You Will be Required to record time for Training and Predicting 
The Code Given on Udacity Website is in Python-2
The Following Code is Python-3 version of the same code
'''

# t0 = time()
# # < your clf.fit() line of code >
# print("Training Time:", round(time()-t0, 3), "s")

# t0 = time()
# # < your clf.predict() line of code >
# print("Predicting Time:", round(time()-t0, 3), "s")

##############################################################