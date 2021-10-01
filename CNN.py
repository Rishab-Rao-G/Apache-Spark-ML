from matplotlib import pyplot
import numpy as np
import pickle as pk
import os, sys
from collections import Counter
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K

from tensorflow.keras import callbacks
from tensorflow.keras.optimizers import Adam


start_time = time.time()

#Load the pickled data 
file = "ECG_Data"+".pk"
with open(file, "rb") as fn:
    X = pk.load(fn)
    Y = pk.load(fn)

#Randomly split the Data into Train and Test in the ratio 70:30    
r_seed = 1229
data_len = len(X)
np.random.seed(r_seed)
idx = list(range(data_len))
np.random.shuffle(idx)

train_len = int(data_len*0.7)
test_len = int(data_len*0.3)

X_train = X[idx][:train_len]
X_test = X[idx][train_len:]
Y_train = Y[idx][:train_len]
Y_test = Y[idx][train_len:]

X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

print(X_train.shape)
print(X_test.shape)
print(Counter(Y_train))
print(Counter(Y_test))

#Input shape and number of classes for classification
f_size = X_train.shape[1]
class_num = 5

#Learning rate and batch size
lr = 0.005
batch_size=128

#Convert the labels inot categorical values
Y_train = tf.keras.utils.to_categorical(Y_train, num_classes=class_num)

#Create the DNN model, using 4 layers and compile it
def make_model():
    model = Sequential()
    model.add(Conv1D(18, 7, activation='relu', input_shape=(f_size,1)))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(18, 7, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(class_num, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr))
    return model

model = make_model()

bin_label = lambda x: min(1,x)

#for e in range(1, 300+1):
for e in range(1, 50+1):
    #Train the model for 50 epochs
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=1, verbose=0)
    
    #Fit the model on the test dataset
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    
    #Evaluation
    acc = np.sum(y_pred==Y_test)/len(Y_test)
    
    #Map the predicted labels and the test labels for evaluation
    y_true = list(map(bin_label, Y_test))
    y_pred = list(map(bin_label, y_pred))
    
    #Extracting true positive, true negative, false positive and false negative 
    #from the confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    SE = tp/(tp+fn)
    SP = tn/(fp+tn)

    print("Epoch: %d | SE: %.4f | ACC %.4f | SP: %.4f"%(e, SE, acc, SP))

#Final Result
print("Final Result: \n")    
print("SE: %.4f | ACC: %.4f | SP: %.4f " %(SE, acc, SP))

print("--- %s seconds ---" % (time.time() - start_time))
