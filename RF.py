import pandas as pd
import numpy as np
import pickle as pk
import os, sys
import seaborn as sns
import time
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import *
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
import time

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

#Random Forest Classification Model. Fit the model on the training set
rf = RandomForestClassifier(n_estimators=100)
rf_model = rf.fit(X_train, Y_train)

#Test the trained model on the training dataset
y_pred = rf_model.predict(X_test)

#Evaluation
acc = accuracy_score(Y_test, y_pred)
conf_mat = confusion_matrix(Y_test, y_pred)
print(type(conf_mat))
print(conf_mat)

#Extracting true positive, true negative, false positive and false negative 
#from the confusion matrix
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
