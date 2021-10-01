import numpy as np
import pandas as pd
import pickle as pk
import os, sys
import seaborn as sns
import time
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.model_selection import  GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal, uniform
from sklearn.svm import SVC
from sklearn.metrics import *
from collections import Counter

start_time = time.time()

#Load the pickled data 
file = "ECG_Data"+".pk"
with open(file, "rb") as fn:
    X = pk.load(fn)
    Y = pk.load(fn)

#Randomly split the Data into Train and Test in the ratio 80:20    
r_seed = 1229
data_len = len(X)
np.random.seed(r_seed)
idx = list(range(data_len))
np.random.shuffle(idx)

train_len = int(data_len*0.8)
test_len = int(data_len*0.2)

X_train = X[idx][:train_len]
X_test = X[idx][train_len:]
Y_train = Y[idx][:train_len]
Y_test = Y[idx][train_len:]

print(X_train.shape)
print(X_test.shape)
print(Counter(Y_train))
print(Counter(Y_test))

#Support Vector Machine Classification Model. Fit the model on the training dataset
svm_clf = SVC(gamma="scale")
svm_clf.fit(X_train, Y_train)

#Hyperparameter tuning og C and gamma variable to prevent overfitting
param_distributions = {"gamma": reciprocal(0.001, 0.1), "C": uniform(1, 10)}

#3-fold cross-validation, iterated 10 times
rnd_search_cv = RandomizedSearchCV(svm_clf, param_distributions, n_iter=10, verbose=2, cv=3)
rnd_search_cv.fit(X_train, Y_train)

#Fitting the best estimator on the Training dataset
rnd_search_cv.best_estimator_.fit(X_train, Y_train)

#Fit the trained model on the test data
y_pred = rnd_search_cv.best_estimator_.predict(X_test)

#Evaluation
acc = accuracy_score(Y_test, y_pred)
conf_mat = confusion_matrix(Y_test, y_pred)

#Extracting true positive, true negative, false positive and false negative 
fp = conf_mat.sum(axis=0) - np.diag(conf_mat)
fn = conf_mat.sum(axis=1) - np.diag(conf_mat)
tp = np.diag(conf_mat)
tn = conf_mat.sum() - (fp + fn + tp)

#Converting the values into float values
fp = fp.astype(float)
fn = fn.astype(float)
tp = tp.astype(float)
tn = tn.astype(float)

#Computing sensitivity and specificity from tn, tp, fn, fp
SE = tp/(tp+fn)
SP = tn/(fp+tn)
SE = np.average(SE)
SP = np.average(SP)

print(" SE: %.4f | ACC: %.4f | SP: %4f" %(SE, acc, SP))

print("--- %s seconds ---" % (time.time() - start_time))